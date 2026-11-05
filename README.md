# MODE

**Multi-Objective Hyperparameter Tuning for Efficient Training at Dynamic Edge Environments**

**MODE** is a multi-objective hyperparameter optimization framework designed for efficient machine-learning training and forecasting in dynamic and resource-constrained edge environments. It uses **Soft-DTW clustering** to summarize resource-usage time series with representative barycenters, allowing an LSTM model to be trained on a much smaller dataset. NSGA-II jointly optimizes prediction accuracy **(validation RMSE)**, training-data compactness **(number of clusters, K)**, and **prediction-error stability (Δσ²)**. An ASF-based selection then chooses one configuration from the resulting Pareto front. The work evaluates MODE on CPU and RAM workload traces collected through the **CODEF** experimentation framework and on an independent public **Raspberry Pi** resource-usage dataset.

## Files and experiments

### MODE implementation and main evaluation

| File | Purpose |
|---|---|
| `MODE_utilities.py` | Core MODE implementation: data loading, Soft-DTW clustering and barycenters, LSTM training/evaluation, NSGA-II search, stability objective, ASF selection, and timing utilities. |
| `Running_MODE.ipynb` | Interactive notebook for running MODE on one selected dataset and inspecting the resulting Pareto front, selected model, and test metrics. |
| `MODE_CODEF_RPi_8_Sequential_Per_Barycenter.ipynb` | Runs the **eight main MODE experiments sequentially**: CODEF Experiments I–III (CPU and RAM) and Raspberry Pi (CPU and RAM). Saves run summaries and raw-scale per-series test RMSE to CSV. |
| `MODE_no_stability.ipynb` | MODE ablation without the stability objective; optimizes RMSE and *K*. |
| `RPi_CPU_MODE_Ablation_CPonly_RMSEonly.ipynb` | RPi CPU ablation experiments for the additional reduced-objective configurations. |
| `MODE_RPi_CPU_Mutation_Sensitivity_Reviewer_2_11.ipynb` | Tests offspring-level mutation probabilities 0.3, 0.5, 0.7, 0.9, and 1.0 on RPi CPU, using repeated seeds. |

## Comparison methods

| File | Purpose |
|---|---|
| `GA.ipynb` | Genetic Algorithm baseline for LSTM hyperparameter tuning. |
| `Benchmark_CCQAROA.ipynb` + `CCQAROA.py` | CCQAROA baseline notebook and supporting optimizer implementation. |
| `MPC.ipynb` | Model-per-cluster baseline: cluster-specific LSTM models and optimization. |
| `CAO_MOO_LSTM.ipynb` + `MOO_2d_iterate_add_neighbor.py` | Adapted multi-objective architecture-search baseline and its neighborhood-search helper. |
| `MOTPE_LP_mirror.ipynb` | MOTPE multi-objective HPO combined with LP-Mirror training-data selection. |
| `MOTPE_LP_scalability.ipynb` | Client-count scalability experiment for MOTPE + LP-Mirror. |

Reference for MO-NAS:
> Q. Cao, S. Liu, A. J. Varghese, J. Darbon, M. S. Triantafyllou, and G. E. Karniadakis, “Automatic selection of the best neural architecture for time series forecasting,” *Nature Communications*, 2026.

Reference for MOTPE:
> Y. Ozaki, Y. Tanigaki, S. Watanabe, M. Nomura, and M. Onishi, “Multiobjective Tree-Structured Parzen Estimator,” *Journal of Artificial Intelligence Research*, vol. 73, pp. 1209–1250, 2022.

### Analysis and figures

| File | Purpose |
|---|---|
| `MODE_scalability_exp.ipynb` | MODE execution-time breakdown and scalability with increasing numbers of profiles or time-series lengths. |
| `MODE_permutations_exp.ipynb` | Tests how permuting complete residual blocks affects the stability objective while preserving order inside each block. |
| `FINAL_PLOTTING.ipynb` | Generates manuscript comparison figures from stored results. |
| `MODE_Comparative_Plots_CODEF_RPi_8.ipynb` | Generates the three main comparison plot types: raw-scale RMSE boxplots, total execution time, and training-data size. |
| `Visiualize copy.ipynb` | Additional visualizations of time series, clusters, barycenters, and predictions. |

*Some repositories or downloaded copies may use slightly different filenames. Adjust the notebook/module names and paths to match your checkout.*

## Datasets

Download the datasets separately:

- **CODEF:** [HEU-101092696-CODECO-CODEF-Resources](https://doi.org/10.5281/zenodo.19005332). Arrange the CSV files under `CODEF/Experiment_1/CPU/`, `CODEF/Experiment_1/RAM/`, and the corresponding `Experiment_2` and `Experiment_3` folders.
- **Raspberry Pi:** [Resource Usage of Applications Running on Raspberry Pi Devices](https://doi.org/10.5683/SP3/GOZAJE). The existing loader uses the folder name `Rasbery_Pi/` (note the spelling).

The main evaluation uses an **80% development / 20% held-out test** split along each time series. Only the development portion is used for clustering and HPO; test observations are reserved for evaluation after selecting the configuration.

---

# LSTM search space

The current MODE implementation tunes the following variables:

| Hyperparameter | Type | Search range |
|---|---|---:|
| Window size | Integer | 5–20 |
| Learning rate | Continuous | 0.0001–0.1 |
| Batch size | Integer | 8–128 |
| Hidden dimension | Integer | 2–10 |
| Epochs | Integer | 50–150 |
| LSTM layers | Integer | 1–5 |
| Number of Soft-DTW clusters \(K\) | Integer | 2–10 |

---
## Setup

Clone the repository, create a Python environment, and install the dependencies used by the notebooks:

```bash
git clone https://github.com/kanasta00/MODE.git
cd MODE
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib tqdm torch pymoo optuna codecarbon openpyxl jupyter
```

The core MODE module also requires its `soft_dtw_fast` backend. Ensure this local module (and its dependencies) is available before running the experiments. Set dataset paths in the notebook configuration cells to match your computer.

## Reproduce the experiments

1. **Prepare the datasets** in the folders described above.
2. **Run the main MODE experiments:** open `MODE_CODEF_RPi_8_Sequential_Per_Barycenter.ipynb`, check its dataset paths and utilities-module import, then run the cells in order. It runs CODEF I/II/III and RPi, CPU and RAM, one experiment at a time.
3. **Run the baselines:** use the corresponding GA, CCQAROA, MPC, MO-NAS/Cao, and MOTPE + LP-Mirror notebooks with the same datasets and evaluation protocol.
4. **Run additional analyses as needed:** use the ablation, scalability, permutation, and mutation-sensitivity notebooks listed above.
5. **Generate figures:** run `FINAL_PLOTTING.ipynb` or `MODE_Comparative_Plots_CODEF_RPi_8.ipynb` using the saved experimental outputs. Plotting does not require repeating HPO.

The sequential MODE notebook saves a run-level summary (`mode_runs_summary.csv`) and raw-scale test RMSE values for individual series (`mode_all_raw_rmse.csv`). Other notebooks may use their own output filenames; check their configuration cells.

**Reproducibility:** Keep the HPO search space, NSGA-II settings, preprocessing, random seeds, and evaluation protocol aligned across compared methods. In particular, the manuscript's polynomial-mutation setting is `PM(prob=0.9, eta=20)`; verify the repository implementation uses this setting before reproducing the reported configuration.

## Manuscript

K. Anastasiou, S. Skaperas, G. Koukis, V. Tsaoussidis, and I. Sakellariou, **“Multi-Objective Hyperparameter Tuning for Efficient Training at Dynamic Edge Environments.”** Manuscript under review.

External datasets remain subject to their original attribution and licensing terms.
are rounded before model evaluation and constrained to their valid ranges.

---
