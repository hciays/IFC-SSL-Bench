import lightning as L
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import umap
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import label_binarize, StandardScaler
from torch import cat, softmax
import xgboost as xgb

from data_processing.data_loader import make_weighted_loader
from models.finetune_model import FinetuneModel
from utils.utils import get_params_text, get_checkpoint_dir, get_class_weights


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def evaluate_retrieval(embs_train, labels_train, embs_test, labels_test, k: int = 3):
    # cosine distances via L2 norm
    embs_train = l2_normalize_np(np.asarray(embs_train))
    embs_test = l2_normalize_np(np.asarray(embs_test))

    # unsupervised knn for retrieval
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(embs_train)
    _, idx = nn.kneighbors(embs_test)
    
    top1 = 0
    topk = 0
    # count correct labels
    for i, neighbors in enumerate(idx):
        true_label = labels_test[i]
        neighb_labels = labels_train[neighbors]
        top1 += int(neighb_labels[0] == true_label)
        topk += int(true_label in neighb_labels)

    return {
        "retr_top1": top1 / len(labels_test), 
        f"retr_top{k}": topk / len(labels_test)
    }


def evaluate_knn(embs_train, labels_train, embs_test, labels_test, n_classes, k: int = 5):
    # cosine distances via L2 norm
    embs_train = l2_normalize_np(np.asarray(embs_train))
    embs_test = l2_normalize_np(np.asarray(embs_test))
    
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    # fit knn classifier on train embeddings loaded from trained model
    knn.fit(embs_train, labels_train)
    # test classifier on the test embeddings
    # majority vote from nearest neighbors
    predictions = knn.predict(embs_test) 
    label_proba = knn.predict_proba(embs_test)

    return {
        "acc": accuracy_score(labels_test, predictions),
        "f1_macro": f1_score(labels_test, predictions, average="macro"),
        "f1_class": f1_score(labels_test, predictions, average=None, labels=list(range(n_classes)))
    }, label_proba


def evaluate_linear_classifier(embs_train, labels_train, embs_test, labels_test, n_classes, max_epochs):
    # standardize features before linear probing
    scaler = StandardScaler().fit(embs_train)
    embs_train = scaler.transform(embs_train)
    embs_test = scaler.transform(embs_test)
    
    # build model - with balanced classes weights
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(embs_train, labels_train)
    
    # get predictions
    predictions = clf.predict(embs_test)
    label_proba = clf.predict_proba(embs_test)

    return {
        "acc": accuracy_score(labels_test, predictions),
        "f1_macro": f1_score(labels_test, predictions, average="macro"),
        "f1_class": f1_score(labels_test, predictions, average=None, labels=list(range(n_classes)))
    }, label_proba
    

def evaluate_xgboost(embs_train, labels_train, embs_test, labels_test, n_classes, seed):
    # standardize features before linear probing
    scaler = StandardScaler().fit(embs_train)
    embs_train = scaler.transform(embs_train)
    embs_test = scaler.transform(embs_test)    
    
    params = {
        "objective": "multi:softprob", # Returns probabilities
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "seed": seed,
        "n_jobs": -1,
        "tree_method": "hist", # Fast histogram algorithm
    }

    # Create DMatrices (XGBoost's optimized data format)
    dtrain = xgb.DMatrix(embs_train, label=labels_train)
    dtest = xgb.DMatrix(embs_test, label=labels_test)

        
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        verbose_eval=False
    )

    # predict returns the probability matrix for multi:softprob
    label_proba = model.predict(dtest)
    predictions = np.argmax(label_proba, axis=1)

    # Compute Metrics
    metrics = {
        "acc": accuracy_score(labels_test, predictions),
        "f1_macro": f1_score(labels_test, predictions, average="macro"),
        "f1_class": f1_score(labels_test, predictions, average=None, labels=list(range(n_classes)))
    }

    return metrics, label_proba

def roc_auc_visualization(model, data_info, h_params, path_root, labels_test, probs, metric):
    # binarize labels and compute macro AUC
    test_labels_bin = label_binarize(labels_test, classes=range(len(data_info["classes"])))
    roc_auc = roc_auc_score(test_labels_bin, probs, average='macro', multi_class='ovr')

    fig, ax = plt.subplots(dpi=350)
    for i in range(probs.shape[1]):
        fpr, tpr, _ = roc_curve(test_labels_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=data_info['classes'][i][:20])

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{data_info['dataset']}, {model.__class__.__name__}, {h_params['backbone'].__name__}: "
                 f"ROC (AUC: {roc_auc:.3f}, {metric})")
    
    ax.legend(loc='lower right')

    params_txt = get_params_text(h_params, data_info)
    fig.text(0.05, -0.035, params_txt, ha='left', va='bottom', fontsize=8, family='monospace')
    fig.subplots_adjust(bottom=0.15)  # make space at bottom

    file_name = f"{path_root}-{model.__class__.__name__}-rocauc-{metric}.png"  # -{data_info['bio_transf']}
    fig.savefig(file_name, bbox_inches='tight', dpi=350)
    plt.show(block=False)
    plt.pause(5)
    plt.close()  # Close the plot window
    
    return {"auroc": roc_auc}


def umap_visualization(
        data_info, h_params, path_root, embeddings, labels, experiments, eval_type="", n_neighbors=15, min_dist=0.4,
):
    # Preprocess
    embs = l2_normalize_np(np.asarray(embeddings))
    labs = np.asarray(labels).reshape(-1)  # ensure shape (N,)
    if experiments is not None:
        experiments = np.asarray(experiments)

    # Fit UMAP once with 3 components
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=h_params["seed"],)
    U = reducer.fit_transform(embs)  # (N, 3)

    # Colormap (tab10 up to 10 classes, else tab20)
    n_classes = len(data_info["classes"])
    cmap = plt.cm.get_cmap("tab10", n_classes)

    # Figure
    fig, ax = plt.subplots(figsize=(6, 5), dpi=350)

    title = (f"{data_info['dataset']}, {h_params['model'].__name__}, {h_params['backbone'].__name__}, {eval_type} - UMAP")
    ax.set_title(title)

    # 2D scatter: UMAP1 vs UMAP2
    if experiments is not None:
        exp_markers = {"Experiment_1": "o", "Experiment_2": "x", "Experiment_3": "*", "Experiment_4": "P"}
        for exp, marker in exp_markers.items():
            mask = experiments == exp
            ax.scatter(U[mask, 0], U[mask, 1], c=labs[mask], cmap=cmap, s=10, marker=marker,)
    else:
        ax.scatter(U[:, 0], U[:, 1], c=labs, cmap=cmap, s=10)

    # Legends 
    handles_cls = [
        mpatches.Patch(color=cmap(i), label=str(data_info["classes"][i])[:20])
        for i in range(n_classes)
    ]
    fig.legend(handles=handles_cls, loc="lower center", ncol=min(n_classes, 3), bbox_to_anchor=(0.5, -0.02))
    if experiments is not None:
        handles_exp = [
            Line2D(
                [0], [0], 
                marker=marker, linestyle="None", markersize=7, markerfacecolor="black", 
                markeredgecolor="gray", label=exp,
            ) for exp, marker in exp_markers.items()
        ]
        fig.legend(handles=handles_exp, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.2, wspace=0.0)

    # Save
    file_name = f"{path_root}-{h_params['model'].__name__}-umap.png"  # -{data_info['bio_transf']}
    if eval_type:
        file_name = f"{path_root}-{h_params['model'].__name__}-{eval_type}-umap.png"
    fig.savefig(file_name, bbox_inches="tight", dpi=350)
    plt.close(fig)
    return file_name


def pca_visualization(data_info, h_params, path_root, embeddings, labels, experiments):
    embs = np.asarray(embeddings)
    labs = np.asarray(labels)
    if experiments is not None:
        experiments = np.asarray(experiments)

    # Standardize features before PCA
    embs = StandardScaler().fit_transform(embs)

    # Fit PCA once with 3 components; use first 2 for 2D plot
    pca = PCA(n_components=2, random_state=h_params["seed"])
    Z = pca.fit_transform(embs)  # shape (N, 3)
    var_ratio = pca.explained_variance_ratio_
    var2 = var_ratio[:2].sum() * 100.0

    # Choose a qualitative colormap that covers your class count
    n_classes = len(data_info["classes"])
    cmap = plt.cm.get_cmap("tab10", n_classes)

    # Build the figure with 2D and 3D subplots
    fig, ax = plt.subplots(figsize=(6, 5), dpi=350)

    title = (f"{data_info['dataset']}, {h_params['model'].__name__}, "
             f"{h_params['backbone'].__name__} - PCA (total: {var2:.1f}%)")
    ax.set_title(title)

    # 2D scatter: PC1 vs PC2
    if experiments is not None:
        exp_markers = {"Experiment_1": "o", "Experiment_2": "x", "Experiment_3": "*", "Experiment_4": "P"}
        for exp, marker in exp_markers.items():
            mask = experiments == exp
            ax.scatter(Z[mask, 0], Z[mask, 1], c=labs[mask], cmap=cmap, s=10, marker=marker,)
    else:
        ax.scatter(Z[:, 0], Z[:, 1], c=labs, cmap=cmap, s=10)

    # Legend with class names
    handles = [
        mpatches.Patch(color=cmap(i), label=data_info["classes"][i][:20])
        for i in range(n_classes)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(n_classes, 3), bbox_to_anchor=(0.5, -0.02))
    if experiments is not None:
        handles_exp = [
            Line2D(
                [0], [0], 
                marker=marker, linestyle="None", markersize=7, markerfacecolor="black", 
                markeredgecolor="gray", label=exp,
            ) for exp, marker in exp_markers.items()
        ]
        fig.legend(handles=handles_exp, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.2, wspace=0.0)

    file_name = f"{path_root}-{h_params['model'].__name__}-pca.png"  # -{data_info['bio_transf']}
    fig.savefig(file_name, bbox_inches="tight", dpi=350)
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    return file_name

def run_finetune(
        backbone, loaders, train_dataset, subset_indices, data_info, h_params, logger, frozen
):
    checkpt_dir = get_checkpoint_dir(h_params, finetune=True, create=True)
    n_classes = len(data_info["classes"])
    
    # build weighted dataloader for training set
    labels, class_weights = get_class_weights(loaders["train"].dataset, subset_indices)
    loader_train = make_weighted_loader(h_params, train_dataset, class_weights, labels)

    # create fine-tune model
    ft_model = FinetuneModel(backbone, n_classes, class_weights, freeze=frozen, lr_backbone=1e-4, lr_head=1e-3)
    checkpoint = ModelCheckpoint(
        dirpath=checkpt_dir,
        monitor="ft_val_f1m", 
        mode="max", 
        save_top_k=1, 
        save_last=True,
        filename="{epoch:02d}-{ft_val_loss:.2f}-{ft_val_f1m:.3f}"
    )
    patience = min(5, h_params["max_epochs"])
    early_stopping = EarlyStopping(
        monitor="ft_val_f1m", 
        mode="max", 
        patience=patience,
        min_delta=0.0,
        verbose=True,
    )
    
    trainer = L.Trainer(
        max_epochs=8 if len(loader_train.dataset) < h_params["max_epochs"] else h_params["max_epochs"],
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint, early_stopping],
        enable_progress_bar=False,
    )
    
    # run fine-tuning
    trainer.fit(ft_model, loader_train, loaders["val"])
    
    # get predictions
    pred_batches = trainer.predict(ft_model, loaders["test"])
    logits = cat([b["logits"] for b in pred_batches], dim=0)
    labels = cat([b["labels"] for b in pred_batches], dim=0).cpu().numpy()
    
    label_proba = softmax(logits, dim=1).cpu().numpy()
    predictions = label_proba.argmax(axis=1)

    return {
        "acc": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_class": f1_score(labels, predictions, average=None, labels=list(range(n_classes)))
    }, label_proba
