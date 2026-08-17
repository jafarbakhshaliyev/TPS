# Sliding-Window Reordering with Overlap Averaging: A Simple Time-Domain Augmentation for Multivariate Forecasting

Official implementation of **Temporal Patch Shuffle (TPS)** for time series forecasting and classification.

Accepted at **CIKM 2026 (Research Track)**.

- **CIKM 2026 paper:** DOI `10.1145/3799682.3840727`
- **Earlier arXiv version:** [arXiv:2604.09067](https://arxiv.org/abs/2604.09067)
- **Supplementary material:** [CIKM2026_TPS_Supplementary_Material.pdf](supplementary/CIKM2026_TPS_Supplementary_Material.pdf)
- **Full forecasting results:** [results.xlsx](results/results.xlsx)

[![CIKM 2026](https://img.shields.io/badge/CIKM-2026-blue.svg)](https://doi.org/10.1145/3799682.3840727)
[![arXiv](https://img.shields.io/badge/arXiv-2604.09067-B31B1B.svg)](https://arxiv.org/abs/2604.09067)

## Abstract
Augmentation has become a central technique for improving deep forecasting models, but classification-style transformations tend to break the coherence between the look-back window and its continuous future target. We describe a simple procedure that unfolds the joint input–target sequence into overlapping sliding windows, randomly reorders a controlled fraction of them—prioritized by a lightweight variance criterion—and reconstructs the sequence by averaging across the overlaps, producing synthetic samples with controlled variation while limiting temporal distortion. The procedure is model-agnostic, introduces only three interpretable hyperparameters, and achieves strong improvements over a comprehensive set of competing augmentations across nine long-term forecasting benchmarks with five backbone families (TSMixer, DLinear, PatchTST, TiDE, LightTS) and four short-term traffic benchmarks with PatchTST. Component-wise ablations, hyperparameter sensitivity studies, distributional-alignment diagnostics, probabilistic forecasting evaluation, and a transfer experiment to univariate and multivariate time series classification clarify the contribution of each design choice.

## Key Contributions
- **TPS (Temporal Patch Shuffle)**:
  - Time Series Forecasting implementation: `time_series_forecasting/utils/augmentations.py`
  - Univariate classification implementation (MiniRocket): `time_series_classification/minirocket/src/augmentation.py`
  - Multivariate classification implementation (MultiRocket): `time_series_classification/MultiRocket/augmentation.py`

![TPS Architecture](fig/TPS_architecture.png)

## Repository Structure
- `time_series_forecasting/`: forecasting models + augmentation pipeline (TPS and others)
- `time_series_classification/`:
  - `minirocket/`: univariate classification (MiniRocket) + augmentations (TPS and others)
  - `MultiRocket/`: multivariate classification (MultiRocket) + augmentations (TPS and others)
- `results/`: detailed experimental results
- `supplementary/`: CIKM 2026 supplementary material

## Quick Start

### 1) Time Series Forecasting

#### Dataset
Download all forecasting datasets from:
- https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

Create the dataset folder and put the downloaded files inside:
```bash
mkdir -p time_series_forecasting/dataset
```

#### Install & Run
```bash
git clone https://github.com/jafarbakhshaliyev/TPS.git
cd TPS

python3 -m venv .venv_forecasting
source .venv_forecasting/bin/activate

pip install -r time_series_forecasting/requirements.txt

# Install PyTorch (choose the right command for your CUDA/CPU setup):
# https://pytorch.org/get-started/locally/

bash time_series_forecasting/scripts/example.sh
```

Notes:
- `time_series_forecasting/scripts/example.sh` now runs relative to the repo
- The training entrypoint is `time_series_forecasting/run_longExp.py`.

### 2) Univariate Time Series Classification (MiniRocket)

#### Dataset (UCR Archive)
- https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

The MiniRocket code expects TSV files under a folder you set in `time_series_classification/minirocket/src/main.py` via `UCR_PATH`.

#### Install & Run
```bash
cd TPS

python3 -m venv .venv_minirocket
source .venv_minirocket/bin/activate

pip install numpy pandas scikit-learn tqdm

# Run the provided script (SLURM `srun`):
bash time_series_classification/minirocket/scripts/example.sh

# If you're not on SLURM, run directly:
cd time_series_classification/minirocket
python3 ./src/main.py --dataset MiddlePhalanxOutlineAgeGroup
```

### 3) Multivariate Time Series Classification (MultiRocket)

#### Dataset (UEA Multivariate Archive)
- https://www.timeseriesclassification.com/index.php

Place `.ts` files under (recommended):
```bash
mkdir -p time_series_classification/MultiRocket/data/Multivariate_ts
```

#### Install & Run
MultiRocket has its own pinned dependencies in `time_series_classification/MultiRocket/requirements.txt`.

```bash
cd TPS

python3 -m venv .venv_multirocket
source .venv_multirocket/bin/activate

pip install -r time_series_classification/MultiRocket/requirements.txt

# The provided script uses SLURM `srun` and contains cluster-specific paths.
# For a portable local run, prefer calling `main.py` directly:
cd time_series_classification/MultiRocket
python3 main.py --datapath ./data/Multivariate_ts --problem FaceDetection --iter 5 --verbose 1
```

## Augmentation Code (Direct Links)
- Forecasting augmentations (TPS + others):
  - [`time_series_forecasting/utils/augmentations.py`](time_series_forecasting/utils/augmentations.py)
- MiniRocket augmentations (TPS + others):
  - [`time_series_classification/minirocket/src/augmentation.py`](time_series_classification/minirocket/src/augmentation.py)
- MultiRocket augmentations (TPS + others):
  - [`time_series_classification/MultiRocket/augmentation.py`](time_series_classification/MultiRocket/augmentation.py)

## Results

Headline numbers from the tables:
- Long-term forecasting (9 datasets × 4 horizons): TPS is rank-1 in most settings (e.g., **DLinear**: 35/36 wins for MSE; 34/36 wins for MAE).
- Short-term traffic forecasting (PeMS03/04/07/08 with PatchTST): TPS wins on most metrics (e.g., PeMS03: MSE/MAE **0.104/0.216**).
- Classification (mean ± std): TPS improves both univariate MiniRocket (**0.804 ± 0.0098**) and multivariate MultiRocket (**0.643 ± 0.0253**).

## Supplementary Material

The supplementary material contains additional experimental details, including:

- augmentation baseline hyperparameters and TPS hyperparameter selection,
- matched seven-dataset long-term forecasting comparisons,
- long-term forecasting results with standard deviations,
- per-horizon short-term traffic forecasting results,
- extended component-wise ablations,
- augmentation size and augmentation ratio analyses,
- UCR and UEA classification dataset details.

- [Download the CIKM 2026 supplementary material](supplementary/CIKM2026_TPS_Supplementary_Material.pdf)
- [Full per-dataset and per-horizon forecasting results](results/results.xlsx)

This is especially important because your main paper will point readers to this repository.

## Citation

Please cite the CIKM 2026 version of the paper. The earlier arXiv version appears under the original TPS title.

### CIKM 2026

```bibtex
@inproceedings{bakhshaliyev2026tps,
  title={Sliding-Window Reordering with Overlap Averaging:
         A Simple Time-Domain Augmentation for Multivariate Forecasting},
  author={Bakhshaliyev, Jafar and Burchert, Johannes and
          Landwehr, Niels and Schmidt-Thieme, Lars},
  booktitle={Proceedings of the 35th ACM International Conference
             on Information and Knowledge Management},
  year={2026},
  doi={10.1145/3799682.3840727}
}
```

### Earlier arXiv version

```bibtex
@misc{bakhshaliyev2026temporalpatchshuffletps,
      title={Temporal Patch Shuffle (TPS): Leveraging Patch-Level Shuffling to Boost Generalization and Robustness in Time Series Forecasting}, 
      author={Jafar Bakhshaliyev and Johannes Burchert and Niels Landwehr and Lars Schmidt-Thieme},
      year={2026},
      eprint={2604.09067},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.09067}, 
}
```

## Code Attribution and Licenses
- Forecasting code credits: see headers in `time_series_forecasting/` files and references in `time_series_forecasting/`.
- MiniRocket is modified from the official implementation (see `time_series_classification/minirocket/LICENSE`).
- MultiRocket is modified from the official implementation (see `time_series_classification/MultiRocket/LICENSE`).
- Third-party attributions: see `THIRD_PARTY_LICENSES.txt` inside each classification submodule.
