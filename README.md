# MODE

<p align="center">
  <strong>Multi-Objective Hyperparameter Tuning for Efficient Training at Dynamic Edge Environments</strong>
</p>

<p align="center">
  Multi-objective HPO · Soft-DTW clustering · Barycenter-based data reduction · Change-point stability analysis · Edge workload forecasting
</p>

<p align="center">
  <a href="https://github.com/kanasta00/MODE">Repository</a>
  ·
  <a href="https://doi.org/10.5683/SP3/GOZAJE">Raspberry Pi dataset</a>
  ·
  <a href="https://doi.org/10.5281/zenodo.19005332">CODEF resource dataset</a>
</p>

---

## Overview

**MODE** is a multi-objective hyperparameter optimization framework designed for efficient machine-learning training in dynamic and resource-constrained edge environments.

The framework jointly considers three competing requirements:

1. **Prediction accuracy** — minimize the validation RMSE of the forecasting model.
2. **Training-data compactness** — minimize the number of Soft-DTW clusters \(K\), which directly controls the size of the barycenter-based training representation.
3. **Cross-client predictive stability** — minimize residual variance instability, quantified through a change-point-based stability objective.

MODE combines:

- **Soft-DTW k-means-like clustering** to group temporally similar workload traces.
- **Soft-DTW barycenters** to obtain one representative trajectory per cluster.
- **Training-data reduction** by replacing many client time series with a compact set of barycenters.
- **LSTM workload forecasting** over the reduced representation.
- **Residual-based change-point analysis** to quantify prediction instability across heterogeneous clients.
- **NSGA-II** to search the joint ML/clustering hyperparameter space.
- **ASF-based preference selection** to choose one deployable configuration from the final Pareto front.

The repository contains the MODE implementation, the experimental notebooks used for the paper evaluation, additional multi-objective baselines, ablation/sensitivity experiments, scalability experiments, and plotting utilities.

---

## Paper

This repository accompanies the manuscript:

> **K. Anastasiou, S. Skaperas, G. Koukis, V. Tsaoussidis, and I. Sakellariou,  
> “Multi-Objective Hyperparameter Tuning for Efficient Training at Dynamic Edge Environments.”**

The work evaluates MODE on CPU and RAM workload traces collected through the **CODEF** experimentation framework and on an independent public **Raspberry Pi** resource-usage dataset.

---

## MODE at a glance

Given \(N\) workload time series, MODE performs the following procedure:

```text
N client workload traces
        │
        ▼
Soft-DTW clustering
        │
        ├── Cluster 1 ──► Soft-DTW barycenter 1
        ├── Cluster 2 ──► Soft-DTW barycenter 2
        ├── ...
        └── Cluster K ──► Soft-DTW barycenter K
        │
        ▼
Compact barycenter-based training representation
        │
        ▼
LSTM training + validation
        │
        ├── Objective 1: validation RMSE
        ├── Objective 2: number of clusters K
        └── Objective 3: residual instability Δσ²
        │
        ▼
NSGA-II multi-objective search
        │
        ▼
Pareto-optimal configurations
        │
        ▼
ASF preference selection
        │
        ▼
Selected MODE configuration
```

The central idea is that the forecasting model does **not** need to be trained on every original client trace. Instead, MODE searches for a compact number of representative Soft-DTW barycenters while simultaneously considering prediction quality and robustness.

---

## Optimization objectives

MODE evaluates each candidate configuration using the objective vector

\[
\mathbf{f} =
\left[
f_1,\,
f_2,\,
f_3
\right]
=
\left[
\mathrm{RMSE},\,
K,\,
\Delta \sigma^2
\right].
\]

### Objective 1 — Forecasting error

The first objective is the validation **root mean squared error (RMSE)** of the LSTM:

\[
f_1 = \mathrm{RMSE}.
\]

Lower values indicate better predictive accuracy.

### Objective 2 — Compactness

The second objective is the number of Soft-DTW clusters:

\[
f_2 = K.
\]

Each cluster contributes one barycenter of length \(T\). Therefore, reducing \(K\) reduces the amount of representative data used for model training.

### Objective 3 — Predictive stability

The third objective quantifies changes in residual variance across the participating time series:

\[
f_3 = \Delta \sigma^2.
\]

Residuals are evaluated over the original workload traces. A CUSUM-based change-point procedure is used to identify variance changes and penalize configurations associated with larger instability.

---

## Final Pareto-solution selection

NSGA-II returns a set of non-dominated configurations rather than a single solution.

The repository therefore includes an **Augmented Scalarization Function (ASF)** step for selecting one final configuration according to user-defined preferences.

The default preference used in the main MODE experiments is

```python
weights = [0.6, 0.2, 0.2]
```

corresponding to:

```text
60%  prediction accuracy
20%  compactness
20%  residual stability
```

Before ASF selection, the objectives are normalized using the minimum and maximum values of the final Pareto front so that objectives with different numerical scales can be compared consistently.

---

# Repository contents

The repository is intentionally organized around experiment notebooks plus a shared implementation module.

## Core implementation

### `MODE_utilities.py`

The main implementation of MODE and the shared utility module used by the experiment notebooks.

It contains functionality for:

- loading CODEF CPU/RAM traces;
- loading the Raspberry Pi resource-usage dataset;
- time-series normalization and preparation;
- Soft-DTW computation;
- Soft-DTW divergence;
- Soft-DTW barycenter computation;
- Soft-DTW k-means-like clustering;
- cluster assignment and cluster repair;
- barycenter concatenation;
- LSTM model definition and training;
- validation and per-series evaluation;
- residual collection;
- CUSUM-based variance change-point analysis;
- MODE objective evaluation;
- MODE without the stability objective;
- NSGA-II execution through `pymoo`;
- SBX crossover and polynomial mutation;
- Pareto-front handling;
- ASF-based final solution selection;
- execution-time decomposition;
- helper routines used by scalability and diagnostic experiments.

The main MODE entry point is conceptually:

```python
results, problem, duration = ApplyingMODE(...)
```

followed by final Pareto selection through:

```python
ApplyingASF(...)
```

The module also contains the corresponding **MODE without stability** implementation used for the ablation study.

---

## Main MODE experiments

The main experiment notebooks execute MODE on the workload datasets used in the paper and evaluate the selected configuration on held-out client traces.

The common workflow is:

```text
1. Load CPU or RAM traces
2. Reshape them into client time series
3. Split each time series into offline and online portions
4. Run MODE on the offline portion
5. Obtain the Pareto front
6. Select a final configuration using ASF
7. Evaluate the selected LSTM on held-out online data
8. Store RMSE, timing, selected architecture, and compactness information
```

The experimental evaluation covers:

- **CODEF Experiment 1**
- **CODEF Experiment 2**
- **CODEF Experiment 3**
- **Raspberry Pi CPU**
- **Raspberry Pi RAM**

If the repository contains experiment-specific copies of the main execution notebook, each copy corresponds to one of these dataset/resource combinations while using the shared functions from `MODE_utilities.py`.

---

## `MODE_scalability_exp.ipynb`

Dedicated computational-scalability experiment for MODE.

It studies two independent scaling dimensions.

### Part A — Client-count scalability

The time-series length is fixed to

```text
L = 500
```

while the number of monitored clients is varied:

```text
N = 10, 60, 100, 500, 1000
```

The experiment records the execution-time contribution of:

- Soft-DTW clustering and barycenter computation;
- LSTM training;
- validation;
- stability evaluation;
- remaining NSGA-II/orchestration overhead.

This experiment is used to determine how MODE behaves as the monitored client population grows.

### Part B — Temporal scalability

The number of clients is fixed to

```text
N = 60
```

while the time-series length is varied.

This isolates the cost associated with increasingly long workload histories.

The scalability notebook writes summary files that can be used directly by the plotting notebook.

---

## `MODE_permutations_exp.ipynb`

Residual-order sensitivity experiment.

The trained MODE model, hyperparameters, residual values, and within-series temporal order are kept fixed, while complete residual blocks are permuted.

The experiment:

- collects residuals separately for each time series;
- preserves the internal chronological order of every residual block;
- evaluates many random block permutations;
- recomputes the residual-stability objective \(\Delta\sigma^2\);
- verifies that pooled residual RMSE is unchanged by permutation;
- studies how the ordering affects the stability objective;
- supports the corresponding permutation/Mantel sensitivity analysis reported in the revised evaluation.

The supplied experiment uses **1,000 random permutations**.

---

# Multi-objective baselines

The revised evaluation includes dedicated multi-objective baselines in addition to the original comparison methods.

## `CAO_MOO_LSTM.ipynb`

Implementation/adaptation of the multi-objective time-series architecture-selection approach used as the **Cao et al.** baseline.

This notebook is used to provide a dedicated multi-objective comparison against MODE rather than relying only on conventional single-objective HPO methods.

It evaluates candidate forecasting architectures and extracts Pareto-efficient configurations according to the objectives considered by the baseline.

Reference:

> Q. Cao, S. Liu, A. J. Varghese, J. Darbon, M. S. Triantafyllou, and G. E. Karniadakis,  
> “Automatic selection of the best neural architecture for time series forecasting,”  
> *Nature Communications*, 2026.

---

## `MOO_2d_iterate_add_neighbor.py`

Supporting implementation associated with the Cao-style multi-objective architecture search.

The script operates on architecture candidates and their objective values, identifies Pareto-efficient solutions, and iteratively explores neighboring architecture configurations around promising Pareto candidates.

The architecture representation includes combinations of recurrent/attention/state-space blocks and hidden dimensions. The script is used by the Cao baseline workflow rather than by the core MODE algorithm.

---

## `MOTPE_LP_mirror.ipynb`

Implementation of the **MOTPE + LP-Mirror** comparison pipeline.

It combines:

- **LP-Mirror** for lightweight training-data selection;
- **MOTPE / Optuna TPE** for multi-objective HPO;
- the same general LSTM forecasting setting used for the MODE comparison.

The pipeline first applies LP-Mirror to identify a reduced subset of informative training examples. MOTPE then searches the LSTM hyperparameter space under competing objectives.

The notebook reports:

- original training-set size;
- selected training-set size;
- retention/reduction ratio;
- LP-Mirror selection time;
- MOTPE optimization time;
- total end-to-end execution time;
- final Pareto solution;
- selected LSTM hyperparameters;
- raw-scale online RMSE for every evaluated time series.

Reference for MOTPE:

> Y. Ozaki, Y. Tanigaki, S. Watanabe, M. Nomura, and M. Onishi,  
> “Multiobjective Tree-Structured Parzen Estimator,”  
> *Journal of Artificial Intelligence Research*, vol. 73, pp. 1209–1250, 2022.

---

## `MOTPE_LP_Mirror_client_count_scalability.ipynb`

Client-count scalability experiment for the MOTPE + LP-Mirror baseline.

It uses the **Raspberry Pi CPU** dataset with fixed

```text
T = 500
```

and varies the number of clients.

The experiment follows the same client-scaling principle as the MODE scalability study, enabling a direct comparison of the growth in end-to-end optimization cost.

The notebook records:

- LP-Mirror data-selection time;
- MOTPE HPO time;
- total optimization time;
- total end-to-end time;
- selected training-set size;
- retention percentage;
- validation RMSE;
- online raw-scale RMSE.

---

# Ablation and diagnostic experiments

## MODE without stability

The repository contains the two-objective MODE variant in which the residual-stability objective is removed.

Its objective vector becomes

\[
\mathbf{f}_{\mathrm{no\ stability}}
=
[
\mathrm{RMSE},
K
].
\]

This experiment quantifies the contribution of the third MODE objective to predictive robustness across heterogeneous clients.

The implementation is provided by functions such as:

```python
ApplyingMODE_no_stability(...)
ApplyingASF_no_stability(...)
```

inside `MODE_utilities.py`.

---

# Plotting and visualization

## `FINAL_PLOTTING.ipynb`

Main plotting notebook used to aggregate stored experiment results and generate paper-ready figures.

Typical outputs include comparisons of:

- RMSE;
- total execution time;
- training-set size / retained data;
- MODE vs. baseline performance;
- CPU vs. RAM behavior;
- CODEF vs. Raspberry Pi results;
- scalability trends;
- execution-time breakdowns.

The notebook is intended to separate expensive experiment execution from figure generation: once experiment outputs are saved, plots can be regenerated without rerunning the HPO procedures.

---

## `Visiualize copy.ipynb`

Auxiliary visualization notebook used for inspecting MODE intermediate outputs and generating explanatory figures.

Examples include:

- original client time series;
- cluster memberships;
- Soft-DTW barycenters;
- concatenated barycenters;
- per-cluster barycenter visualizations;
- predictions versus targets;
- diagnostic plots used during manuscript preparation.

---

# Datasets

The repository does **not** need to redistribute the complete external datasets. Download the datasets from their original sources and place them in the expected local folders.

---

## 1. CODEF CPU/RAM dataset

MODE was originally evaluated using CPU and memory workload traces generated through the **CODEF** cloud-edge experimentation framework.

Public dataset:

**HEU-101092696-CODECO-CODEF-Resources**  
DOI: **10.5281/zenodo.19005332**

Link:

https://doi.org/10.5281/zenodo.19005332

The dataset contains real CPU and memory utilization time series from Kubernetes-based cloud-edge infrastructures collected during CODEF benchmarking experiments.

The current loader expects a structure equivalent to:

```text
CODEF/
├── Experiment_1/
│   ├── CPU/
│   │   └── *.csv
│   └── RAM/
│       └── *.csv
├── Experiment_2/
│   ├── CPU/
│   │   └── *.csv
│   └── RAM/
│       └── *.csv
└── Experiment_3/
    ├── CPU/
    │   └── *.csv
    └── RAM/
        └── *.csv
```

CPU values are processed as utilization percentages, while RAM values are converted to MiB where necessary.

---

## 2. Raspberry Pi resource-usage dataset

Independent validation is performed using the public dataset:

> **R. Kain et al., “Resource Usage of Applications Running on Raspberry Pi Devices,”  
> Queen's Telecommunications Research Lab (TRL), Version 1.0, 2022.**

DOI:

**10.5683/SP3/GOZAJE**

Dataset link:

https://doi.org/10.5683/SP3/GOZAJE

The dataset contains dynamic resource-usage measurements collected from **four heterogeneous Raspberry Pi 4 devices** with different RAM capacities and CPU frequencies.

According to the dataset documentation, it contains:

- more than **550,000 measurements**;
- approximately **768 hours** of application execution;
- **74 CSV files**;
- measurements sampled at approximately five-second granularity;
- CPU, memory, network, disk, temperature, GPU, and Wi-Fi-related metrics.

MODE uses the CPU and memory resource-usage columns for the independent forecasting experiments.

### Expected local directory

The current implementation expects the Raspberry Pi CSV files under:

```text
Rasbery_Pi/
```

Note the spelling: **`Rasbery_Pi`** is the folder name currently used by the loader in `MODE_utilities.py`.

The helper

```python
read_raspberry_pi_dataset(
    series_length=500,
    base_folder="Rasbery_Pi",
    column="cpu",
)
```

reads the CSV files in sorted order, concatenates the valid observations, and reshapes the measurements into equal-length time series.

For RAM experiments, use:

```python
column="memory"
```

---

# Experimental data split

A typical experiment represents each client as a time series of length \(T\).

For the Raspberry Pi experiments used in the scalability analysis:

```text
T = 500
```

with:

```text
80% offline  = 400 observations
20% online   = 100 observations
```

The offline portion is used for clustering/HPO/model development, while the online portion is retained for post-selection evaluation.

Always keep the held-out online portion separate from the HPO process.

---

# LSTM search space

The current MODE implementation tunes the following variables:

| Hyperparameter | Type | Search range |
|---|---|---:|
| Window size | Integer | 5–20 |
| Learning rate | Continuous | \(10^{-4}\)–\(10^{-1}\) |
| Batch size | Integer | 8–128 |
| Hidden dimension | Integer | 2–10 |
| Epochs | Integer | 50–150 |
| LSTM layers | Integer | 1–3 |
| Number of Soft-DTW clusters \(K\) | Integer | 3–10 |

Internally, evolutionary operators work on a real-valued representation. Integer-valued parameters are rounded before model evaluation and constrained to their valid ranges.

---

# NSGA-II configuration

MODE uses `pymoo`'s NSGA-II implementation with:

```python
sampling = LHS()
crossover = SBX(prob=0.9, eta=15)
mutation = PM(eta=20)
eliminate_duplicates = True
```

The main paper experiments use a fixed optimization budget so that comparisons are made under the same search budget.

---

# Installation

A Python virtual environment is recommended.

```bash
git clone https://github.com/kanasta00/MODE.git
cd MODE

python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm
pip install torch
pip install pymoo
pip install optuna
pip install codecarbon
pip install openpyxl
```

The core implementation also imports a fast Soft-DTW backend exposing:

```python
_soft_dtw
_soft_dtw_grad
_jacobian_product_sq_euc
```

through `soft_dtw_fast`. Make sure the corresponding Soft-DTW implementation/module used by this repository is available in the active environment.

For fully reproducible execution, using the same package versions and hardware/software environment as the reported experiments is recommended.

---

# Quick start

A minimal MODE workflow is:

```python
import numpy as np
import MODE_utilities as AS

# Example:
# data.shape = (N, T)

results, problem, duration = AS.ApplyingMODE(
    data=data,
    num_population=20,
    num_offsprings=10,
    num_generations=10,
    gamma=1.0,
    max_iter=50,
    barycenter_max_iter=50,
    normalize=True,
    use_divergence=False,
    min_cluster_size=2,
    split=0.70,
    D_star_tp=1.8,
)

best_instance, best_model, best_chromosome, best_info = AS.ApplyingASF(
    results_of_tuning=results,
    Problem_class=problem,
    weights=np.array([0.6, 0.2, 0.2]),
    return_full=True,
)

print("Optimization duration:", duration)
print("Selected configuration:", best_chromosome)
print("Selected objectives:", best_info["objectives"])
```

Parameter values should be aligned with the experiment you intend to reproduce.

---

# Evaluating a selected model on a client trace

The selected MODE model can be evaluated on an individual held-out time series using:

```python
result = AS.evaluate_one_individual_series(
    ML=best_instance,
    series_raw=test_series,
    sdtw_normalize=True,
    return_raw_scale=True,
)

print("RMSE:", result["rmse_raw"])
```

The experiment notebooks repeat this operation for every held-out client trace and store the resulting per-series RMSE values.

---

# Computational-cost accounting

The implementation explicitly records the wall-clock contribution of the main MODE stages:

```text
clustering_sec
training_sec
validation_sec
stability_sec
other_sec
total_mode_sec
```

`total_mode_sec` represents the end-to-end MODE optimization time.

The clustering measurement includes Soft-DTW clustering and barycenter computation. The remaining NSGA-II/orchestration overhead is derived as the difference between total runtime and the explicitly timed stages.

This allows the reported efficiency evaluation to include the **full optimization cost**, rather than only LSTM fitting time.

---

# Reproducing the scalability analysis

## Client-count scalability

Use the Raspberry Pi CPU dataset and keep the series length fixed:

```text
L = 500
```

Evaluate MODE for:

```text
N = 10, 60, 100, 500, 1000
```

The first \(N\) complete time series are used at every scale so that larger experiments extend, rather than replace, the smaller client sets.

## Temporal scalability

Keep:

```text
N = 60
```

fixed and change the time-series length.

The scalability notebook records both total runtime and stage-level runtime decomposition.

---

# Reproducing the permutation analysis

The residual-order experiment should be run only after a fixed MODE model/configuration has been obtained.

The procedure is:

```text
1. Evaluate the fixed model independently on each time series.
2. Preserve the residual sequence belonging to each client as one block.
3. Preserve the chronological order inside every block.
4. Randomly permute only the order of complete blocks.
5. Recompute the stability objective for each permutation.
6. Compare the distribution with the original ordering.
```

Because the residual values themselves do not change, pooled residual RMSE should remain numerically invariant apart from floating-point precision.

---

# Baselines

The complete evaluation compares MODE against multiple classes of alternatives.

### Conventional HPO / optimization baselines

- Genetic Algorithm (GA)
- Chaos Crossover Quantum Attraction-Repulsion Optimization Algorithm (CCQAROA)

### Model-per-cluster baseline

- MPC

### Multi-objective baselines

- Cao et al. multi-objective neural architecture selection
- MOTPE
- MOTPE + LP-Mirror

### MODE ablation

- MODE without the stability objective

This combination allows the experiments to compare MODE against:

- conventional HPO;
- model-per-cluster training;
- dedicated multi-objective search;
- data-reduction + multi-objective HPO;
- MODE itself without its robustness objective.

---

# Saved outputs

Depending on the notebook, experiments may write:

```text
*.txt
*.csv
*.pkl
*.xlsx
*.png
```

These outputs may contain:

- selected hyperparameters;
- Pareto solutions;
- validation objectives;
- per-client online RMSE;
- timing information;
- energy information where enabled;
- cluster counts;
- training-set sizes;
- scalability summaries;
- residual permutation statistics;
- plotting-ready aggregated results.

Expensive experiments should be executed once and saved. The plotting notebooks can then operate on these stored outputs without repeating the full HPO procedure.

---

# Reproducibility notes

Several aspects are important when comparing results:

### Randomness

Evolutionary optimization, model initialization, clustering initialization, and mini-batch training can introduce randomness.

Use fixed seeds when reproducing a specific run.

### GPU timing

CUDA operations are asynchronous. The implementation synchronizes CUDA around timed regions where required so that wall-clock measurements represent completed GPU operations.

### Hardware

Absolute execution time depends on:

- CPU/GPU model;
- number of threads;
- CUDA/cuDNN versions;
- PyTorch version;
- power-management settings;
- background system load.

Relative methodological conclusions should therefore be interpreted together with the reported execution environment.

### Dataset ordering

The Raspberry Pi loader reads CSV files in sorted filename order before concatenating observations. Do not change the file ordering when reproducing the reported experiments.

### No test leakage

The online/held-out portion of each series must remain excluded from model selection and hyperparameter optimization.

---

# Key Python dependencies

The main code uses:

```text
numpy
pandas
scipy
scikit-learn
torch
pymoo
optuna
matplotlib
tqdm
codecarbon        # when energy tracking is enabled
openpyxl          # spreadsheet/result processing
```

A fast Soft-DTW backend is additionally required by `MODE_utilities.py`.

---

# Related methods

MODE builds on the following methodological components:

- **Soft-DTW** — differentiable dynamic time warping for time-series comparison and barycenter estimation.
- **NSGA-II** — elitist multi-objective evolutionary optimization.
- **CUSUM / change-point analysis** — residual-variance stability analysis.
- **ASF** — preference-based selection from a non-dominated Pareto front.
- **LSTM** — forecasting model used in the experimental evaluation.

---

# Citation

If you use MODE, its implementation, or the experimental workflow in your research, please cite the accompanying manuscript.

```bibtex
@article{anastasiou2026mode,
  author  = {Konstantinos Anastasiou and
             Sotiris Skaperas and
             Georgios Koukis and
             Vassilis Tsaoussidis and
             Ilias Sakellariou},
  title   = {Multi-Objective Hyperparameter Tuning for Efficient Training
             at Dynamic Edge Environments},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Please update the bibliographic entry with the final journal volume, issue, pages, and DOI once the paper is published.

---

# Dataset citation

For experiments using the Raspberry Pi data, please also cite the original dataset:

```text
R. Kain et al.,
Resource Usage of Applications Running on Raspberry Pi Devices,
Queen's Telecommunications Research Lab (TRL),
Version 1.0, 2022.
DOI: 10.5683/SP3/GOZAJE
```

For CODEF traces, cite the corresponding CODECO/CODEF dataset record:

```text
HEU-101092696-CODECO-CODEF-Resources
DOI: 10.5281/zenodo.19005332
```

---

# Authors

- **Konstantinos Anastasiou** — University of Macedonia
- **Sotiris Skaperas** — University of Macedonia
- **Georgios Koukis** — Democritus University of Thrace
- **Vassilis Tsaoussidis** — Democritus University of Thrace
- **Ilias Sakellariou** — University of Macedonia

---

# License

Please refer to the license file included in the repository for the licensing terms of the source code.

External datasets remain subject to their **original licenses**. In particular, downloading or using the Raspberry Pi dataset does not transfer ownership of that dataset to this repository; users must follow the licensing and attribution requirements provided by the dataset authors.

---

# Acknowledgement

This repository was created to support transparent and reproducible evaluation of MODE and its comparison baselines.

If you reproduce the experiments, find an issue, or extend MODE to new edge/cloud workload traces, contributions and issue reports are welcome.

