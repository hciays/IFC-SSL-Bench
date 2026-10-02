# IFC-SSL-Bench

A benchmark of self-supervised learning for high-throughput imaging flow cytometry.


This repository contains the code and configurations for a systematic evaluation of six
self-supervised objectives across four backbones with our without ImageNet-supervised weights initalization, on two diverse imaging
flow cytometry datasets and a range of annotation regimes, with preprocessing and pretraining
scale held fixed so that comparisons isolate the objective and the initialization.

---

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Running experiments](#running-experiments)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [License](#license)

---

## Overview

The pipeline runs in two stages.

**Stage 1 — Pretraining.** A backbone is trained on unannotated images with one of
the self-supervised objectives, or supervised as a baseline.

**Stage 2 — Benchmarking.** The resulting representation is evaluated with three protocols (
*k*-NN classification, linear probing and full fine-tuning) across several
annotation fractions of the data.

| | |
|---|---|
| **Methods** | Supervised · SimCLR · MoCo v2 · MoCo v3 · BYOL · Barlow Twins · DINO |
| **Backbones** | ResNet-50 · ConvNeXt Tiny · ConvNeXt V2 Tiny · ViT |
| **Datasets** | BloodMNIST · RBC morphology · Immunological synapses |
| **Evaluation** | *k*-NN · Linear probe · Fine-tuning |
| **Seeds** | 3 per configuration |

---

## Installation

Requires Python 3.12.9 and a CUDA-capable GPU for pretraining. Evaluation and figure
generation run on CPU.

```bash
git clone <REPO_URL>
cd IFC-SSL-Bench

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Experiment tracking (optional)

Weights & Biases is supported but not required:

```bash
wandb login
export WANDB_PROJECT=ifc-ssl-bench
```

Set `wandb.enabled: false` in `config.yml`, or `WANDB_MODE=disabled`, to turn it off. All
metrics are also written locally to `logs/losses/` and `logs/benchmark/`, so nothing depends on
having an account.

---

## Datasets

| Dataset | Role | Access | Reference |
|---|---|---|---|
| BloodMNIST | Pipeline validation | [`medmnist`](https://github.com/MedMNIST/MedMNIST) Python library | Acevedo et al., *Data in Brief* (2020) |
| RBC morphology | Benchmark | See the data availability statement of the source paper | [Doan et al., *PNAS* 117:21381 (2020)](https://doi.org/10.1073/pnas.2001227117) |
| Immunological synapses | Benchmark | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.ht76hdrk7) | Shetab Boushehri et al., *Nat. Commun.* 14:3620 (2023) |

**BloodMNIST is used only to validate the pipeline**, not as a benchmark dataset. It is small,
public and fast to run, which makes it a convenient end-to-end check that pretraining,
evaluation and logging all work before committing to a full experiment. It is obtained through
the [MedMNIST Python library](https://github.com/MedMNIST/MedMNIST) (`pip install medmnist`),
so no manual download is needed.

### Setup

Dataset locations are set in `config.yml` — there is no separate setup script. Point each
dataset entry at the directory holding its images and split manifests, and the pipeline reads
from there.

### Preprocessing and splits

All datasets pass through identical preprocessing (`src/data/preprocessing.py`), which is what
makes the cross-method comparison valid.

Splits for the downstream analyses are 64 / 16 / 20 % train / validation / test, generated with a fixed seed.
For the RBC dataset, we use the provided subfolders directly as plits.

---

## Running experiments

```bash
python run_local.py

```
Make sure to select appropriate parameters in the scripts

---

## Configuration

`config.yml` holds all set parameters for the  experiments.


---

## Outputs

Each run produces:

- **Checkpoints** — last, best and periodic model weights
- **Loss logs** — per-epoch training loss, as CSV
- **Benchmark results** — accuracy and per-class macro F1 for retrieval, *k*-NN, linear probing
  and fine-tuning, as CSV
- **Embedding visualizations** — UMAP and PCA of the learned representations
- **Resolved configuration** — the full set of parameters the run actually used


## License

[![License: MIT](https://shields.io)](https://opensource.org)

This project is licensed under the MIT License.
