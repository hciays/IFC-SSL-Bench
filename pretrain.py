from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torchinfo import summary
from torch import cat, softmax
from sklearn.metrics import accuracy_score, f1_score

from data_processing.data_loader import build_loaders, make_weighted_loader
from models.model_utils import create_model, load_trained_model
from plot_figures.visualize_loss import visualize_loss
from utils.utils import get_checkpoint_dir, get_class_weights, setup_logger, get_benchmark_dir
from utils.benchmark_metrics import plot_confusion_matrix


def pretrain(
        h_params: dict,
        data_info: dict,
        run_time: str,
        continue_train: bool = False
):
    print(
        f"\n============== TRAINING ================= \n\n"
        f"Dataset:      {data_info['dataset']} \n"
        f"Model:        {h_params['model'].__name__} \n"
        f"Backbone:     {h_params['backbone'].__name__} \n"
        f"Init Type:    {h_params['init_type']} \n"
        f"Seed:         {h_params['seed']} \n"
        f"=========================================\n"
    )

    # Load dataset and create data loaders
    print(f"Get data and setup data loader...")
    if h_params["model"].__name__ != "Supervised" and data_info["dataset"] == "RBC_Dataset":
        data_info["norm_vals"] = data_info["norm_vals-pretraining"]
    loaders, data_info = build_loaders(h_params, data_info)
    if h_params["model"].__name__ == "Supervised":
        labels, class_weights = get_class_weights(loaders["train"].dataset)
    else:
        labels = None
        class_weights = None

    # Logger setup
    if not h_params['test_run']:
        print(f"\nWandb setup...")
        wandb_logger = setup_logger(h_params, data_info, run_time)
    else:
        wandb_logger = False

    print(f"\nCreate model...")
    # Build model
    if not continue_train:  # normal run
        model = create_model(h_params, data_info, class_weights, train=True)
    else:  
        # option to continue training a model from a given checkpoint
        model, _ = load_trained_model(h_params, data_info, class_weights, last_ckpt=continue_train)
    summary(model)

    print("\nCheckpoint and trainer...")
    # Set checkpoint saving strategy
    checkpt_dir = get_checkpoint_dir(h_params)
    print(f"Will save model to: {checkpt_dir}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpt_dir,
        monitor="train_loss",
        save_top_k=1,  # 0 - don't save top-k monitored checkpoints
        save_on_train_epoch_end=True,
        mode="min",
        save_last=True,  # save last checkpoint
        filename="{epoch:02d}-{train_loss:.2f}"
    )
    # Configure PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=h_params["max_epochs"],
        accelerator="auto",  # auto-detect "gpu", "cpu", "mps", or "tpu"
        devices="auto",  # automatically selects available device(s)
        logger=wandb_logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback],
    )

    print(f"\nFinished setup.\ndata_info:\n{data_info}\n\nh_params:\n{h_params}")

    # Train model and log timing
    start_time = datetime.now()
    print(f"\nRunning training... {start_time}\n")
    if h_params["model"].__name__ == "Supervised":
        # for supervised model use weighted dataloader, and validation
        loader_train = make_weighted_loader(h_params, loaders["train"].dataset, class_weights, labels)
        trainer.fit(model, loader_train, loaders["val"])
        pred_batches = trainer.predict(model, loaders["test"], ckpt_path="best")
        logits = cat([b["logits"] for b in pred_batches], dim=0)
        labels = cat([b["labels"] for b in pred_batches], dim=0).cpu().numpy()    
        label_proba = softmax(logits, dim=1).cpu().numpy()
        predictions = label_proba.argmax(axis=1)
        import pandas as pd
        import os
        row = {
            "run_time":  run_time,
            "init_type": h_params["init_type"],
            "seed":      h_params["seed"],
            "fraction":  h_params["fraction"],
            "dataset":   data_info["dataset"],
            "method":    h_params["model"].__name__,
            "backbone":  h_params["backbone"].__name__,
            "norm":      data_info["bio_transf"],
            "metric":    "supervised_baseline",
            "acc":       accuracy_score(labels, predictions),
            "f1_macro":  f1_score(labels, predictions, average="macro", zero_division=0),
        }
        f1c = f1_score(labels, predictions, average=None,
                    labels=list(range(len(class_weights))), zero_division=0)
        row.update({f"f1_{c}": v for c, v in zip(data_info["classes"], f1c)})
        df = pd.DataFrame([row])
        benchm_folder, path_root = get_benchmark_dir(data_info, h_params, run_time)
        cm_path = os.path.join(
            benchm_folder,
            f"{path_root}-{h_params['model'].__name__}-f{h_params['fraction']}-s{h_params['seed']}-cm")
        png = plot_confusion_matrix(
            labels, predictions, data_info["classes"], cm_path,
            title=f"{h_params['model'].__name__} · {h_params['fraction']}% labels · seed {h_params['seed']}")
        file_name = f"{path_root}-{h_params['model'].__name__}-metrics.csv"
        per_run_csv = os.path.join(benchm_folder, file_name)
        if os.path.exists(per_run_csv):
            df.to_csv(per_run_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(per_run_csv, index=False)
        print(f"\nSaved results to {per_run_csv}\n")
    
        if wandb_logger:
            import wandb
            # Log results to WandB
            wandb_logger.experiment.log({"confusion_matrix": wandb.Image(png)})
            wandb_logger.experiment.summary.update(
                {k: v for k, v in row.items() if isinstance(v, (int, float))})
    else:
        # SSL doesn't use validation
        trainer.fit(model, loaders["train"])
    total_time = datetime.now() - start_time
    print(f"\nFinished training. Total time: {total_time}\n")

    # Wrap up logging
    if wandb_logger:
        # Log results to WandB
        wandb_logger.experiment.summary["training_time"] = str(total_time)
        wandb_logger.experiment.summary["hyperparams"] = h_params
        wandb_logger.experiment.summary["data_info"] = data_info

    visualize_loss(model, h_params, data_info, run_time)
