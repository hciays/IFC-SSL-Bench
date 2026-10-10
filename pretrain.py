from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torchinfo import summary

from data_processing.data_loader import build_loaders, make_weighted_loader
from models.model_utils import create_model, load_trained_model
from plot_figures.visualize_loss import visualize_loss
from utils.utils import get_checkpoint_dir, get_dataset_path, get_class_weights, setup_logger


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

    # Use local dataset directory if applicable
    if data_info["dataset"] in ["RBC_Dataset", "Immuno_Synapses"]:
        data_path = get_dataset_path(data_info["dataset"])
    else:  
        # BloodMNIST doesn't need local path
        data_path = None

    # Load dataset and create data loaders
    print(f"Get data and setup data loader...")
    loaders, data_info = build_loaders(h_params, data_info, data_path)
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
