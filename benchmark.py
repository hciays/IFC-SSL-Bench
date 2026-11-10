import os
import shutil
from typing import Dict, Any

import torch

from data_processing.data_loader import build_loaders
from models.model_utils import extract_embeddings, load_trained_model
from utils.benchmark_metrics import evaluate_knn, evaluate_linear_classifier, evaluate_retrieval, \
    roc_auc_visualization, umap_visualization, pca_visualization, run_finetune, evaluate_xgboost
from utils.utils import get_benchmark_dir, get_class_weights, \
    make_subset, setup_logger, get_checkpoint_dir
from utils.results_io import metric_row, save_rows, scalar_summary
from utils.benchmark_metrics import plot_confusion_matrix

def benchmark(
        h_params: Dict[str, Any],
        data_info: Dict[str, Any],
        run_time: str,
        fractions,
        last_ckpt=False
):
    print(
        f"\n============== BENCHMARK ================= \n\n"
        f"Dataset:      {data_info['dataset']} \n"
        f"Model:        {h_params['model'].__name__} \n"
        f"Backbone:     {h_params['backbone'].__name__} \n"
        f"Init Type:    {h_params['init_type']} \n"
        f"Seed:         {h_params['seed']} \n\n"
        f"=========================================\n"
    )

    # Paths for the run-level CSV. Captured BEFORE the fraction loop so the
    # results file does not end up inside the last fraction's folder.
    run_folder, run_root = get_benchmark_dir(data_info, h_params, run_time)

    # get data and data loaders
    if data_info["dataset"] in ["RBC_Dataset", "Immuno_Synapses"]:
        data_path = data_info["data_dir"]
    else:
        data_path = None

    loaders, data_info = build_loaders(h_params, data_info, data_path, eval=True)

    print(f"\nWandb setup...")
    logger = setup_logger(h_params, data_info, run_time, eval=True)

    # ================= Qualitative Analysis (no fractions) =====================

    if h_params["model"].__name__ == "Supervised":
        _, class_weights = get_class_weights(loaders["train"].dataset, subset_indices=None)
    else:
        class_weights = None

    trained_model, ckpt_path = load_trained_model(
        h_params, data_info, class_weights, last_ckpt=last_ckpt)
    print(f"Loaded model from: {ckpt_path}")

    print(f"\nExtract embeddings....\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats_test, labs_test, experiments = extract_embeddings(trained_model, loaders["test"], device)

    umap_path = umap_visualization(
        data_info, h_params, run_root, feats_test, labs_test, experiments
    )
    print(f"UMAP visualization at: {umap_path}")

    pca_path = pca_visualization(
        data_info, h_params, run_root, feats_test, labs_test, experiments
    )
    print(f"PCA visualization at: {pca_path}")

    # ======================== Quantitative eval / Label efficient ================
    rows = []
    class_names = data_info["classes"]
    n_classes = len(class_names)

    def record(metric, metrics, probs=None, folder=None, root=None):
        """Append one uniform row and, if probabilities are given, a confusion matrix."""
        rows.append(metric_row(run_time, h_params, data_info, metric, metrics, class_names))
        if probs is not None:
            cm_path = os.path.join(folder, f"{root}-{metric}-cm")
            plot_confusion_matrix(
                labs_test, probs.argmax(axis=1), class_names, cm_path,
                title=f"{h_params['model'].__name__} · {metric} · "
                      f"{h_params['fraction']}% · seed {h_params['seed']}")
        print(f"{metric} - Metrics: \n{metrics}")

    for fraction in fractions:
        h_params.update({"fraction": fraction})
        if h_params["model"].__name__ == "Supervised" and fraction != 100:
            continue

        print(f"\n\n ========================== {h_params['fraction']} "
              f"========================== \n\n")
        benchm_folder, path_root = get_benchmark_dir(data_info, h_params, run_time)

        # reset loaders for immu_syn dataset - 3 splits from labeled dataset
        if data_info["dataset"] == "Immuno_Synapses":
            loaders, data_info = build_loaders(h_params, data_info, data_path, eval=True)

        # get subset / fraction of labeled data
        if h_params["fraction"] < 100:
            train_dataset, subset_indices = make_subset(
                loaders["train"].dataset, h_params["fraction"], h_params["seed"])
        else:
            train_dataset = loaders["train"].dataset
            subset_indices = None

        trained_model, ckpt_path = load_trained_model(
            h_params, data_info, class_weights, last_ckpt=last_ckpt)
        print(f"Loaded model from: {ckpt_path}")

        print(f"\nExtract embeddings....\n")
        feats_tr, labs_tr, _ = extract_embeddings(trained_model, loaders["train"], device)
        feats_test, labs_test, _ = extract_embeddings(trained_model, loaders["test"], device)
        if subset_indices is not None:
            feats_tr = feats_tr[subset_indices]
            labs_tr = labs_tr[subset_indices]

        # =================== Start Fraction Eval =======================

        print(f"\nEval retrieval.... 1/5")
        retr_metrics = evaluate_retrieval(feats_tr, labs_tr, feats_test, labs_test)
        record("retrieval", retr_metrics)

        print(f"\nEval kNN.... 2/5")
        knn_metrics, knn_label_prob = evaluate_knn(
            feats_tr, labs_tr, feats_test, labs_test, n_classes)
        knn_auroc = roc_auc_visualization(
            trained_model, data_info, h_params, path_root, labs_test, knn_label_prob, "kNN")
        record("knn", {**knn_metrics, **knn_auroc}, knn_label_prob, benchm_folder, path_root)

        print(f"\nEval lin probe.... 3/5")
        lp_metrics, lp_label_prob = evaluate_linear_classifier(
            feats_tr, labs_tr, feats_test, labs_test, n_classes, h_params["max_epochs"])
        lp_auroc = roc_auc_visualization(
            trained_model, data_info, h_params, path_root, labs_test, lp_label_prob, "LP")
        record("lin_probe", {**lp_metrics, **lp_auroc}, lp_label_prob, benchm_folder, path_root)

        print(f"\nEval xgboost.... 4/5")
        xgb_metrics, xgb_label_prob = evaluate_xgboost(
            feats_tr, labs_tr, feats_test, labs_test, n_classes, h_params["seed"])
        xgb_auroc = roc_auc_visualization(
            trained_model, data_info, h_params, path_root, labs_test, xgb_label_prob, "XGB")
        record("xgboost", {**xgb_metrics, **xgb_auroc}, xgb_label_prob, benchm_folder, path_root)

        print(f"\nFine-tune - full.... 5/5\n")
        ft_metrics, ft_label_prob = run_finetune(
            trained_model.backbone, loaders, train_dataset, subset_indices,
            data_info, h_params, logger, frozen=False)
        ft_auroc = roc_auc_visualization(
            trained_model, data_info, h_params, path_root, labs_test, ft_label_prob, "FT")
        record("finetune", {**ft_metrics, **ft_auroc}, ft_label_prob, benchm_folder, path_root)

        # Extract embeddings with fine-tuned backbone & plot UMAP
        _, ft_class_weights = get_class_weights(loaders["train"].dataset, subset_indices)
        ft_trained_model, ft_ckpt_path = load_trained_model(
            h_params, data_info, ft_class_weights, finetune=True)
        print(f"Loaded model from: {ft_ckpt_path}")
        ft_feats_test, ft_labs_test, ft_experiments = extract_embeddings(
            ft_trained_model, loaders["test"], device)

        umap_path_ft = umap_visualization(
            data_info, h_params, path_root, ft_feats_test, ft_labs_test,
            ft_experiments, eval_type="ft_full")
        print(f"\nFT UMAP: {umap_path_ft}")

        try:
            ckpt_dir = get_checkpoint_dir(h_params, finetune=True, create=False)
            if "finetune" in ckpt_dir.parts and ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
                print(f"\n[CLEANUP] Removed finetune checkpoints at {ckpt_dir}")
            else:
                print(f"\n[CLEANUP] Skipping deletion, unexpected path: {ckpt_dir}")
        except Exception as e:
            print(f"Couldn't delete finetune checkpoints: {e}")

    # ======================== STORING & CLEAN-UP ========================

    per_run_csv = os.path.join(
        run_folder, f"{run_root}-{h_params['model'].__name__}-metrics.csv")
    save_rows(rows, per_run_csv)
    print(f"\nSaved results to {per_run_csv}\n")

    if logger:
        logger.experiment.summary.update(scalar_summary(rows))

    return rows