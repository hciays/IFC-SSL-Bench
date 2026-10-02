import os
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from data_processing.data_loader import build_loaders
from models.model_utils import extract_embeddings, load_trained_model
from utils.benchmark_metrics import evaluate_knn, evaluate_linear_classifier, evaluate_retrieval, \
    roc_auc_visualization, umap_visualization, pca_visualization, run_finetune, evaluate_xgboost
from utils.utils import get_benchmark_dir, get_dataset_path, get_checkpoint_path, get_class_weights, \
    make_subset, setup_logger, get_checkpoint_dir


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

    benchm_folder, path_root = get_benchmark_dir(data_info, h_params, run_time)
    # get data and data loaders
    if data_info["dataset"] in ["RBC_Dataset", "Immuno_Synapses"]:
        data_path = get_dataset_path(data_info["dataset"])
    else:
        data_path = None
        
    # get data loaders
    loaders, data_info = build_loaders(h_params, data_info, data_path, eval=True)
    
    # Logger setup
    print(f"\nWandb setup...")
    logger = setup_logger(h_params, data_info, run_time, eval=True)
    
    # ================= Qualitative Analysis (no fractions) =====================

    # for supervised model - calculate class weights
    if h_params["model"].__name__ == "Supervised":
        _, class_weights = get_class_weights(loaders["train"].dataset, subset_indices=None)
    else: 
        class_weights = None
    # load trained SSL model (backbone inside)
    trained_model, ckpt_path = load_trained_model(h_params, data_info, class_weights, last_ckpt=last_ckpt)
    print(f"Loaded model from: {ckpt_path}")

    print(f"\nExtract embeddings....\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Pre-extract full-train/test embeddings
    feats_test, labs_test, experiments = extract_embeddings(trained_model, loaders["test"], device)

    umap_path = umap_visualization(
        data_info, h_params, path_root, feats_test, labs_test, experiments
    )
    print(f"UMAP visualization at: {umap_path}")

    pca_path = pca_visualization(
        trained_model, data_info, h_params, path_root, feats_test, labs_test, experiments
    )
    print(f"PCA visualization at: {pca_path}")

    # ======================== Quantitative eval / Label efficient ========================
    rows = []

    for fraction in fractions:
        h_params.update({"fraction": fraction})
        if h_params["model"].__name__ == "Supervised" and fraction != 100:
            continue

        print(f"\n\n ========================== {h_params['fraction']} ========================== \n\n")
        benchm_folder, path_root = get_benchmark_dir(data_info, h_params, run_time)

        # reset loaders for immu_syn dataset - 3 splits from labeled dataset
        if data_info["dataset"] == "Immuno_Synapses":
            loaders, data_info = build_loaders(h_params, data_info, data_path, eval=True)

        # get subset / fraction of labeled data
        if h_params["fraction"] < 100:
            train_dataset, subset_indices = make_subset(loaders["train"].dataset, h_params["fraction"], h_params["seed"])
        else:
            train_dataset = loaders["train"].dataset
            subset_indices = None

        # load trained SSL model (backbone inside)
        trained_model, ckpt_path = load_trained_model(h_params, data_info, class_weights, last_ckpt=last_ckpt)
        print(f"Loaded model from: {ckpt_path}")

        print(f"\nExtract embeddings....\n")
        # Pre-extract full-train/val/test embeddings once per seed
        feats_tr, labs_tr, _ = extract_embeddings(trained_model, loaders["train"], device)
        feats_test, labs_test, _ = extract_embeddings(trained_model, loaders["test"], device)
        # select the train features of the subset by index
        if subset_indices is not None:
            feats_tr = feats_tr[subset_indices]
            labs_tr = labs_tr[subset_indices]

        # =================== Start Fraction Eval =======================

        # retrieval
        print(f"\nEval retrieval.... 1/5")
        retr_metrics = evaluate_retrieval(feats_tr, labs_tr, feats_test, labs_test)
        rows.append({
            "run_time": run_time,
            "init_type": h_params["init_type"],
            "seed": h_params["seed"],
            "fraction": h_params["fraction"],
            "dataset": data_info["dataset"],
            "method": h_params["model"].__name__,
            "backbone": h_params["backbone"].__name__,
            # "norm": data_info["bio_transf"],
            "metric": "retrieval", 
            **retr_metrics,
        })
        print(f"Retrieval - Metrics: \n{retr_metrics}")

        # kNN
        print(f"\nEval kNN.... 2/5")
        knn_metrics, knn_label_prob = evaluate_knn(
            feats_tr, labs_tr, feats_test, labs_test, len(data_info["classes"])
        )
        knn_auroc = roc_auc_visualization(
            trained_model, data_info, h_params, path_root, labs_test, knn_label_prob, "kNN"
        )
        rows.append({
            "run_time": run_time,
            "init_type": h_params["init_type"],
            "seed": h_params["seed"],
            "fraction": h_params["fraction"],
            "dataset": data_info["dataset"],
            "method": h_params["model"].__name__,
            "backbone": h_params["backbone"].__name__,
            # "norm": data_info["bio_transf"],
            "metric": "knn", 
            **knn_metrics,
            **knn_auroc,
        })
        print(f"kNN - Metrics: \n{knn_metrics}")

        # linear probing (frozen backbone)
        print(f"\nEval lin probe.... 3/5")
        lp_metrics, lp_label_prob = evaluate_linear_classifier(
            feats_tr, labs_tr, feats_test, labs_test, len(data_info["classes"]), h_params["max_epochs"]
        )
        lp_auroc = roc_auc_visualization(
            trained_model, data_info, h_params, path_root, labs_test, lp_label_prob, "LP"
        )
        rows.append({
            "run_time": run_time,
            "init_type": h_params["init_type"],
            "seed": h_params["seed"],
            "fraction": h_params["fraction"],
            "dataset": data_info["dataset"],
            "method": h_params["model"].__name__,
            "backbone": h_params["backbone"].__name__,
            # "norm": data_info["bio_transf"],
            "metric": "lin_probe", 
            **lp_metrics,
            **lp_auroc,
        })
        print(f"Linear Probing - Metrics: \n{lp_metrics}")
        
        # xgboost
        print(f"\nEval xgboost.... 4/5")
        xgb_metrics, xgb_label_prob = evaluate_xgboost(
            feats_tr, labs_tr, feats_test, labs_test, len(data_info["classes"]), h_params["seed"]
        )
        xgb_auroc = roc_auc_visualization(
            trained_model, data_info, h_params, path_root, labs_test, xgb_label_prob, "XGB"
        )
        rows.append({
            "run_time": run_time,
            "init_type": h_params["init_type"],
            "seed": h_params["seed"],
            "fraction": h_params["fraction"],
            "dataset": data_info["dataset"],
            "method": h_params["model"].__name__,
            "backbone": h_params["backbone"].__name__,
            # "norm": data_info["bio_transf"],
            "metric": "xgboost", 
            **xgb_metrics,
            **xgb_auroc,
        })
        print(f"XGBoost - Metrics: \n{xgb_metrics}")

        # full finetuning with unfrozen backbone
        print(f"\nFine-tune - full.... 5/5\n")
        ft_metrics, ft_label_prob = run_finetune(
            trained_model.backbone, loaders, train_dataset, subset_indices, data_info, h_params, logger, frozen=False
        )
        ft_auroc = roc_auc_visualization(
            trained_model, data_info, h_params, path_root, labs_test, ft_label_prob, "FT"
        )
        rows.append({
            "run_time": run_time,
            "init_type": h_params["init_type"],
            "seed": h_params["seed"],
            "fraction": h_params["fraction"],
            "dataset": data_info["dataset"],
            "method": h_params["model"].__name__,
            "backbone": h_params["backbone"].__name__,
            # "norm": data_info["bio_transf"],
            "metric": "finetune", 
            **ft_metrics, 
            **ft_auroc,
        })
        print(f"\nFT full - Metrics: \n{ft_metrics}")
        
        # Extract embeddings with fine-tuned backbone & plot UMAP
        _, class_weights = get_class_weights(loaders["train"].dataset, subset_indices)
        ft_trained_model, ft_ckpt_path = load_trained_model(h_params, data_info, class_weights, finetune=True)
        print(f"Loaded model from: {ft_ckpt_path}")
        ft_feats_test, ft_labs_test, ft_experiments = extract_embeddings(ft_trained_model, loaders["test"], device)
        
        umap_path_ft = umap_visualization(
            data_info, h_params, path_root, ft_feats_test, ft_labs_test, ft_experiments, eval_type="ft_full"
        )
        print(f"\nFT UMAP: {umap_path_ft}")
    
        try:
            ckpt_dir = get_checkpoint_dir(h_params, finetune=True, create=False)
            if "finetune" in ckpt_dir.parts and ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
                print(f"\n[CLEANUP] Removed finetune checkpoints at {ckpt_dir}")
            else:
                print(f"\n[CLEANUP] Skipping deletion, unexpected path: {ckpt_dir}")
        except:
            print(f"Coudn't delete {ft_ckpt_path}")

    # ======================== STORING & CLEAN-UP ========================

    # Save per-run rows and summary
    df = pd.DataFrame(rows)
    
    file_name = f"{path_root}-{h_params['model'].__name__}-metrics.csv"
    per_run_csv = os.path.join(benchm_folder, file_name)
    if os.path.exists(per_run_csv):
        df.to_csv(per_run_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(per_run_csv, index=False)
    print(f"\nSaved results to {per_run_csv}\n")

    if logger:
        # Log results to WandB
        logger.experiment.summary["benchmark"] = rows
