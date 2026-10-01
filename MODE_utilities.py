# ============================================================
# 0. Imports and global device
# ============================================================

# from codecarbon import EmissionsTracker
from scipy.stats import norm
import numpy as np
import time
import numpy as np
import re
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize as scipy_minimize
from pymoo.optimize import minimize as pymoo_minimize
from soft_dtw_fast import _soft_dtw, _soft_dtw_grad, _jacobian_product_sq_euc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import csv
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination import get_termination
from pymoo.decomposition.asf import ASF
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

def _sync_device():
    """
    Synchronize CUDA before/after timing GPU operations.
    Necessary for accurate wall-clock measurements.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ============================================================
# 1. Data loading
# ============================================================


def read_experiment_data(experiment, metric, base_folder="CODEF"):
    """
    Reads all CSV files from a selected experiment/metric folder and returns
    all valid CPU or RAM values as a single flattened NumPy array.

    Expected structure:
        CODEF/
        ├── Experiment_1/
        │   ├── CPU/
        │   └── RAM/
        ├── Experiment_2/
        │   ├── CPU/
        │   └── RAM/
        └── Experiment_3/
            ├── CPU/
            └── RAM/

    Parameters
    ----------
    experiment : int or str
        Experiment number or name.
        Examples:
            1
            2
            "Experiment_1"

    metric : str
        Either "CPU" or "RAM".

    base_folder : str
        Root CODEF folder. Defaults to "CODEF", assuming it is in the
        current working directory.

    Returns
    -------
    numpy.ndarray
        Flattened array containing all valid measurements.

        CPU values are returned as percentages.
        RAM values are returned in MiB.


    Example:
    cpu_exp1 = md.read_experiment_data(1, "CPU")
    ram_exp1 = md.read_experiment_data(1, "RAM")

    cpu_exp2 = md.read_experiment_data(2, "CPU")
    ram_exp2 = md.read_experiment_data(2, "RAM")

    cpu_exp3 = md.read_experiment_data(3, "CPU")
    ram_exp3 = md.read_experiment_data(3, "RAM")
    """

    # Normalize experiment name
    if isinstance(experiment, int):
        experiment = f"Experiment_{experiment}"

    # Normalize metric
    metric = metric.upper()

    if metric not in ("CPU", "RAM"):
        raise ValueError("metric must be either 'CPU' or 'RAM'.")

    # Build folder path
    folder_path = os.path.join(
        base_folder,
        experiment,
        metric
    )

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Folder does not exist: {folder_path}"
        )

    all_values = []

    # Read every CSV file
    for filename in sorted(os.listdir(folder_path)):

        if not filename.lower().endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            with open(
                file_path,
                newline="",
                encoding="utf-8-sig"
            ) as csvfile:

                reader = csv.reader(csvfile, delimiter=",")

                # Skip header
                next(reader, None)

                for row in reader:

                    if len(row) < 2:
                        continue

                    raw_value = row[1].strip()

                    if raw_value == "":
                        continue

                    # CPU
                    if metric == "CPU":

                        value_str = (
                            raw_value
                            .replace("%", "")
                            .strip()
                        )

                    # RAM
                    else:

                        if "GiB" in raw_value:
                            value_str = (
                                raw_value
                                .replace("GiB", "")
                                .strip()
                            )

                            try:
                                value = float(value_str) * 1024

                                if not np.isnan(value):
                                    all_values.append(value)

                            except ValueError:
                                pass

                            continue

                        else:
                            value_str = (
                                raw_value
                                .replace("MiB", "")
                                .strip()
                            )

                    try:
                        value = float(value_str)

                        if not np.isnan(value):
                            all_values.append(value)

                    except (ValueError, TypeError):
                        continue

        except Exception as e:
            print(f"Could not read {filename}: {e}")

    return np.array(all_values, dtype=float)



def read_vmcloud_dataset(metric, series_length=500, csv_path="vmCloud_data.csv"):
    """
    Choices for metric:
        ["vm_id", "timestamp", "cpu_usage", "memory_usage"]
    """
    df = (
        pd.read_csv(csv_path, parse_dates=["timestamp"])
        [[metric]]
        .dropna()
        .reset_index(drop=True)
    )

    values = df[metric].to_numpy(dtype=np.float64)

    n_series_total = len(values) // series_length
    trimmed = values[:n_series_total * series_length]

    all_series = trimmed.reshape((n_series_total, series_length))
    return all_series



def read_raspberry_pi_dataset(series_length=500, base_folder="Rasbery_Pi", column = 'cpu'):
    """
    Reads the 'cpu' or 'memory' column from all CSV files inside the Raspberry_Pi folder,
    concatenates all valid values, and splits them into equal-length time series.

    Expected structure:
        Raspberry_Pi/
        ├── file1.csv
        ├── file2.csv
        ├── file3.csv
        └── ...

    Parameters
    ----------
    series_length : int
        Desired length of each time series.

    base_folder : str
        Folder containing the CSV files.

    Returns
    -------
    numpy.ndarray
        2D NumPy array with shape:

            (n_series, series_length)

        where n_series is determined by the total number of available
        CPU measurements.

    Example
    -------
    cpu_series = read_raspberry_pi_dataset(series_length=500)

    print(cpu_series.shape)
    """

    if not os.path.isdir(base_folder):
        raise FileNotFoundError(
            f"Folder does not exist: {base_folder}"
        )

    all_values = []

    # Read all CSV files in sorted order
    for filename in sorted(os.listdir(base_folder)):

        if not filename.lower().endswith(".csv"):
            continue

        file_path = os.path.join(base_folder, filename)

        try:
            df = pd.read_csv(file_path)

            if column not in df.columns:
                print(f"Skipping {filename}: column {column} not found.")
                continue

            values = (
                pd.to_numeric(df[column], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )

            all_values.extend(values)

        except Exception as e:
            print(f"Could not read {filename}: {e}")

    # Convert everything into one flattened array
    all_values = np.array(all_values, dtype=float)

    if len(all_values) == 0:
        raise ValueError(
            f"No valid {column} values were found in the CSV files."
        )

    # Calculate how many complete time series can be created
    n_series = len(all_values) // series_length

    if n_series == 0:
        raise ValueError(
            f"Not enough data to create a time series of length "
            f"{series_length}. Only {len(all_values)} values were found."
        )

    # Remove remaining values that cannot form a complete time series
    trimmed = all_values[:n_series * series_length]

    # Split into N time series
    all_series = trimmed.reshape(n_series, series_length)

    return all_series




def _natural_sort_key(text):
    """
    Natural sorting helper:
    Experiment2 comes before Experiment10.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


def _parse_codef_value(row, resource_type):
    """
    Parse CPU or RAM value from one CSV row.

    Expected:
        row[1] contains either:
            CPU: "45.2%" or "45.2"
            RAM: "512 MiB", "1.2 GiB", or numeric value
    """

    if len(row) < 2:
        return None

    raw = str(row[1]).strip()

    if raw == "":
        return None

    resource_type = resource_type.upper()

    try:
        if resource_type == "CPU":
            value_str = raw.replace("%", "").strip()
            return float(value_str)

        elif resource_type == "RAM":
            raw_clean = raw.replace(",", "").strip()

            if "GiB" in raw_clean:
                value_str = raw_clean.replace("GiB", "").strip()
                return float(value_str) * 1024.0

            elif "MiB" in raw_clean:
                value_str = raw_clean.replace("MiB", "").strip()
                return float(value_str)

            else:
                # Assume already numeric, probably MiB
                return float(raw_clean)

        else:
            raise ValueError("resource_type must be either 'CPU' or 'RAM'.")

    except Exception:
        return None


def read_codef_dataset(
    root_folder,
    resource_type="CPU",
    series_length=100,
    batch_size=40,
    experiment_names=None,
    recursive=False,
    verbose=True,
):
    """
    Read the CODEF Dynamic Cloud-Edge Resource Demand Dataset and form batches.

    Folder structure expected
    -------------------------
    root_folder/
        Experiment1/
            CPU/
                *.csv
            RAM/
                *.csv
        Experiment2/
            CPU/
                *.csv
            RAM/
                *.csv
        Experiment3/
            CPU/
                *.csv
            RAM/
                *.csv

    Parameters
    ----------
    root_folder : str
        Path to:
        'CODEF Dynamic Cloud-Edge Resource Demand Dataset from Kubernetes Experiments and Stress Testing'

    resource_type : str
        Either 'CPU' or 'RAM'.

    series_length : int
        Length of each individual time series.

    batch_size : int
        Number of time series per batch.

    experiment_names : list[str] or None
        If None, all folders starting with 'Experiment' are used.
        Example:
            ['Experiment1', 'Experiment2', 'Experiment3']

    recursive : bool
        If False, read CSV files directly inside ExperimentX/CPU or ExperimentX/RAM.
        If True, read CSV files recursively inside that folder.

    verbose : bool
        Print dataset summary.

    Returns
    -------
    List_of_batches : list[np.ndarray]
        List of batches. Each batch has shape:
            (batch_size, series_length)

    n_complete_batches : int
        Number of complete batches.

    metadata : dict
        Useful loading information.
    """

    resource_type = resource_type.upper()

    if resource_type not in ["CPU", "RAM"]:
        raise ValueError("resource_type must be either 'CPU' or 'RAM'.")

    if not os.path.isdir(root_folder):
        raise FileNotFoundError(f"Root folder does not exist:\n{root_folder}")

    # ------------------------------------------------------------
    # Find experiments
    # ------------------------------------------------------------

    if experiment_names is None:
        experiment_names = [
            name for name in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, name))
            and name.lower().startswith("experiment")
        ]

        experiment_names = sorted(
            experiment_names,
            key=_natural_sort_key
        )

    if len(experiment_names) == 0:
        raise ValueError(
            "No experiment folders found. Expected folders like "
            "'Experiment1', 'Experiment2', 'Experiment3'."
        )

    all_values = []
    file_records = []

    # ------------------------------------------------------------
    # Read all CSV files from each experiment/resource folder
    # ------------------------------------------------------------

    for experiment_name in experiment_names:

        resource_folder = os.path.join(
            root_folder,
            experiment_name,
            resource_type
        )

        if not os.path.isdir(resource_folder):
            raise FileNotFoundError(
                f"Missing folder for {resource_type}:\n{resource_folder}"
            )

        if recursive:
            csv_files = []
            for current_root, _, files in os.walk(resource_folder):
                for file in files:
                    if file.lower().endswith(".csv"):
                        csv_files.append(os.path.join(current_root, file))

            csv_files = sorted(csv_files, key=_natural_sort_key)

        else:
            csv_files = [
                os.path.join(resource_folder, file)
                for file in os.listdir(resource_folder)
                if file.lower().endswith(".csv")
            ]

            csv_files = sorted(csv_files, key=_natural_sort_key)

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in:\n{resource_folder}")

        for file_path in csv_files:

            file_values = []

            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile, delimiter=",")

                # Skip header
                next(reader, None)

                for row in reader:
                    value = _parse_codef_value(
                        row=row,
                        resource_type=resource_type
                    )

                    if value is not None and not np.isnan(value):
                        file_values.append(value)

            all_values.extend(file_values)

            file_records.append({
                "experiment": experiment_name,
                "resource_type": resource_type,
                "file": file_path,
                "n_values": len(file_values),
            })

    values = np.asarray(all_values, dtype=np.float64)

    if len(values) == 0:
        raise ValueError("No valid numeric values were read.")

    # ------------------------------------------------------------
    # Convert flat values to series and batches
    # ------------------------------------------------------------

    n_series_total = len(values) // series_length

    if n_series_total == 0:
        raise ValueError(
            f"Not enough values to form one time series. "
            f"Read {len(values)} values, but series_length={series_length}."
        )

    trimmed_for_series = values[:n_series_total * series_length]
    all_series = trimmed_for_series.reshape(n_series_total, series_length)

    n_complete_batches = n_series_total // batch_size

    if n_complete_batches == 0:
        raise ValueError(
            f"Not enough time series to form one batch. "
            f"Created {n_series_total} time series, but batch_size={batch_size}."
        )

    usable_series = all_series[:n_complete_batches * batch_size]

    batches = usable_series.reshape(
        n_complete_batches,
        batch_size,
        series_length
    )

    List_of_batches = [
        batches[i]
        for i in range(n_complete_batches)
    ]

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    n_values_used = n_complete_batches * batch_size * series_length

    metadata = {
        "root_folder": root_folder,
        "resource_type": resource_type,
        "experiment_names": experiment_names,
        "series_length": series_length,
        "batch_size": batch_size,
        "n_raw_values": int(len(values)),
        "n_values_used": int(n_values_used),
        "n_values_dropped": int(len(values) - n_values_used),
        "n_series_total_before_batching": int(n_series_total),
        "n_series_used": int(n_complete_batches * batch_size),
        "n_complete_batches": int(n_complete_batches),
        "batch_shape": List_of_batches[0].shape,
        "file_summary": pd.DataFrame(file_records),
    }

    if verbose:
        print("=" * 80)
        print("CODEF dataset loaded")
        print("=" * 80)
        print("Resource:", resource_type)
        print("Experiments:", experiment_names)
        print("Raw values:", len(values))
        print("Values used:", n_values_used)
        print("Values dropped:", len(values) - n_values_used)
        print("Series length:", series_length)
        print("Batch size:", batch_size)
        print("Number of complete batches:", n_complete_batches)
        print("Shape of first batch:", List_of_batches[0].shape)

    return List_of_batches, n_complete_batches, metadata











# ============================================================
# 2. Soft-DTW core
# ============================================================


class SquaredEuclidean:
    """Squared Euclidean ground-cost matrix between two time series."""

    def __init__(self, X, Y):
        self.X = to_2d_ts(X)
        self.Y = to_2d_ts(Y)

    def compute(self):
        return euclidean_distances(self.X, self.Y, squared=True)

    def jacobian_product(self, E):
        G = np.zeros_like(self.X)
        _jacobian_product_sq_euc(self.X, self.Y, E, G)
        return G


class SoftDTW:
    """Soft-DTW dynamic-programming value and gradient wrapper."""

    def __init__(self, D, gamma=1.0):
        self.D = D.compute() if hasattr(D, "compute") else D
        self.D = np.asarray(self.D, dtype=np.float64)
        self.gamma = float(gamma)

    def compute(self):
        m, n = self.D.shape
        self.R_ = np.zeros((m + 2, n + 2), dtype=np.float64)
        _soft_dtw(self.D, self.R_, gamma=self.gamma)
        return self.R_[m, n]

    def grad(self):
        if not hasattr(self, "R_"):
            raise ValueError("Call compute() before grad().")

        m, n = self.D.shape

        D = np.vstack((self.D, np.zeros(n)))
        D = np.hstack((D, np.zeros((m + 1, 1))))

        E = np.zeros((m + 2, n + 2), dtype=np.float64)
        _soft_dtw_grad(D, self.R_, E, gamma=self.gamma)

        return E[1:-1, 1:-1]


def sdtw_barycenter(
    X,
    barycenter_init,
    gamma=1.0,
    weights=None,
    method="L-BFGS-B",
    tol=1e-5,
    max_iter=100,
):
    """
    Compute a Soft-DTW barycenter.

    Parameters
    ----------
    X : list[np.ndarray]
        Time series, each shaped (T,) or (T, 1).
    barycenter_init : np.ndarray
        Initial barycenter, shaped (T,) or (T, 1).
    gamma : float
        Soft-DTW smoothing parameter.
    weights : None or array-like
        Non-negative weights. If None, all weights are 1.
    """

    X = [to_2d_ts(x) for x in X]
    barycenter_init = to_2d_ts(barycenter_init)

    if weights is None:
        weights = np.ones(len(X), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if len(weights) != len(X):
        raise ValueError(f"weights has length {len(weights)}, but X has length {len(X)}")

    def _func(Z_flat):
        Z = Z_flat.reshape(*barycenter_init.shape)
        G = np.zeros_like(Z)
        obj = 0.0

        for x_i, w_i in zip(X, weights):
            D = SquaredEuclidean(Z, x_i)
            sdtw = SoftDTW(D, gamma=gamma)
            value = sdtw.compute()
            E = sdtw.grad()
            G_tmp = D.jacobian_product(E)

            obj += w_i * value
            G += w_i * G_tmp

        return obj, G.ravel()

    res = scipy_minimize(
        _func,
        barycenter_init.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options={"maxiter": max_iter},
    )

    return res.x.reshape(*barycenter_init.shape)


def soft_dtw_value(x, y, gamma=1.0):
    D = SquaredEuclidean(x, y)
    sdtw = SoftDTW(D, gamma=gamma)
    return sdtw.compute()


def soft_dtw_divergence(x, y, gamma=1.0):
    """
    Soft-DTW divergence:

        D_gamma(x, y)
        = sDTW(x, y)
        - 0.5 * sDTW(x, x)
        - 0.5 * sDTW(y, y)
    """
    xy = soft_dtw_value(x, y, gamma=gamma)
    xx = soft_dtw_value(x, x, gamma=gamma)
    yy = soft_dtw_value(y, y, gamma=gamma)

    return xy - 0.5 * xx - 0.5 * yy




# ============================================================
# 3. Normalization and shape utilities
# ============================================================

def to_1d_ts(ts):
    """Convert one time series to shape (T,)."""
    ts = np.asarray(ts, dtype=np.float64)

    if ts.ndim == 2 and ts.shape[1] == 1:
        ts = ts.ravel()

    if ts.ndim != 1:
        raise ValueError(f"Expected shape (T,) or (T, 1), got {ts.shape}")

    return ts


def to_2d_ts(ts):
    """Convert one time series to shape (T, 1)."""
    ts = np.asarray(ts, dtype=np.float64)

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    if ts.ndim != 2 or ts.shape[1] != 1:
        raise ValueError(f"Expected shape (T,) or (T, 1), got {ts.shape}")

    return ts


def normalize_single_ts(ts, eps=1e-8):
    """Per-series min-max normalization for one time series."""
    x = to_1d_ts(ts)
    return (x - x.min()) / (x.max() - x.min() + eps)


def minmax_normalize_ts(X, eps=1e-8):
    """
    Per-series min-max normalization for a batch of time series.

    Input shape:
        (n_series, T) or (n_series, T, 1)

    Output shape:
        (n_series, T)
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 3 and X.shape[2] == 1:
        X = X[:, :, 0]

    if X.ndim != 2:
        raise ValueError(f"Expected batch shape (n_series, T) or (n_series, T, 1), got {X.shape}")

    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)

    return (X - mins) / (maxs - mins + eps)


def prepare_batch(X, normalize=True):
    """
    Prepare a batch for Soft-DTW routines.

    Returns:
        X_used : np.ndarray, shape (n_series, T, 1)

    Important:
        If normalize=True, the returned array is the exact normalized data that
        should be reused later for exact recomputation and cluster-member extraction.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 3 and X.shape[2] == 1:
        X = X[:, :, 0]

    if X.ndim != 2:
        raise ValueError(f"Expected batch shape (n_series, T) or (n_series, T, 1), got {X.shape}")

    if normalize:
        X = minmax_normalize_ts(X)

    return X.reshape(X.shape[0], X.shape[1], 1)


def prepare_single(ts, normalize=True):
    """
    Prepare one time series for comparison/update.

    Use the same normalize flag used in soft_dtw_kmeans().
    """
    if normalize:
        ts = normalize_single_ts(ts)
    return to_2d_ts(ts)



# ============================================================
# 4. Soft-DTW k-means
# ============================================================


def distances_to_barycenters(ts, barycenters, gamma=1.0, use_divergence=True):
    """Compute distances from one time series to all barycenters."""
    ts = to_2d_ts(ts)
    distances = []

    for b in barycenters:
        b = to_2d_ts(b)

        if use_divergence:
            d = soft_dtw_divergence(ts, b, gamma=gamma)
        else:
            d = soft_dtw_value(ts, b, gamma=gamma)

        distances.append(d)

    return np.asarray(distances, dtype=np.float64)


def assign_to_best_cluster(ts, barycenters, gamma=1.0, use_divergence=True):
    """Return best cluster index and distance vector."""
    distances = distances_to_barycenters(
        ts=ts,
        barycenters=barycenters,
        gamma=gamma,
        use_divergence=use_divergence,
    )
    return int(np.argmin(distances)), distances


# Backward-compatible alias, if old notebook cells still call it.
soft_dtw_divergence_distances = distances_to_barycenters


def initialize_barycenters_lloyd_random(X, k):
    """
    Lloyd-style random initialization.

    This initializes the k barycenters by randomly selecting k observed
    time series from the dataset.

    This is the appropriate initialization when we want the Cuturi-style
    soft-DTW k-means variant, i.e. assignment with raw soft-DTW and no
    divergence-based farthest-point initialization.
    """
    X = np.asarray(X, dtype=np.float64)

    if k < 1:
        raise ValueError("k must be >= 1")

    if k > len(X):
        raise ValueError(
            f"k={k} cannot be larger than number of time series={len(X)}"
        )

    rng = np.random.default_rng()
    selected_indices = rng.choice(len(X), size=k, replace=False)

    barycenters = [X[idx].copy() for idx in selected_indices]

    return np.asarray(barycenters, dtype=np.float64)


def initialize_barycenters_farthest(X, k, gamma=1.0):
    """
    Farthest-point initialization under Soft-DTW divergence.

    X must already be shaped (n_series, T, 1), preferably by prepare_batch().
    """
    X = np.asarray(X, dtype=np.float64)

    if k < 1:
        raise ValueError("k must be >= 1")

    if k > len(X):
        raise ValueError(f"k={k} cannot be larger than number of time series={len(X)}")

    rng = np.random.default_rng()

    first_idx = int(rng.integers(len(X)))
    centers = [X[first_idx].copy()]
    selected = {first_idx}

    while len(centers) < k:
        min_dist_to_centers = np.full(len(X), -np.inf, dtype=np.float64)

        for i, x in enumerate(X):
            if i in selected:
                continue

            dists = [soft_dtw_divergence(x, c, gamma=gamma) for c in centers]
            # dists = [soft_dtw_value(x, c, gamma=gamma) for c in centers]
             
            min_dist_to_centers[i] = np.min(dists)

        next_idx = int(np.argmax(min_dist_to_centers))
        centers.append(X[next_idx].copy())
        selected.add(next_idx)

    return np.asarray(centers, dtype=np.float64)


def soft_dtw_kmeans(
    X,
    k,
    gamma=1.0,
    max_iter=100,
    barycenter_max_iter=50,
    normalize=True,
    use_divergence=True,
    min_cluster_size=1,
    return_data=False,
    verbose=True,
):
    """
    Soft-DTW k-means.

    Logic
    -----
    If use_divergence=True:
        - initialize with farthest-point initialization
        - assign clusters using Soft-DTW divergence

    If use_divergence=False:
        - initialize with Lloyd-style random initialization
        - assign clusters using raw Soft-DTW dtw_gamma

    Parameters
    ----------
    X : array-like
        Batch of time series, shape (n_series, T) or (n_series, T, 1).

    k : int
        Number of clusters.

    gamma : float
        Soft-DTW smoothing parameter.

    normalize : bool
        If True, each time series is independently min-max normalized.
        If False, the raw values are used.

    use_divergence : bool
        If True, use Soft-DTW divergence for initialization and assignment.
        If False, use Cuturi-style raw Soft-DTW assignment and Lloyd-style
        random initialization.

    min_cluster_size : int
        If >1, clusters smaller than this are repaired by re-seeding with
        poorly represented points. Use 1 if you do not want to force minimum
        cluster sizes.

    return_data : bool
        If True, also return X_used, the data actually used internally.

    Returns
    -------
    labels, barycenters
        or labels, barycenters, X_used if return_data=True.
    """
    X_used = prepare_batch(X, normalize=normalize)
    n_series = len(X_used)

    # barycenters = initialize_barycenters_farthest(
    #     X_used,
    #     k=k,
    #     gamma=gamma,
    #     random_state=random_state,
    # )
    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    if use_divergence:
        # Your improved version:
        # farthest-point initialization under Soft-DTW divergence
        barycenters = initialize_barycenters_farthest(
            X_used,
            k=k,
            gamma=gamma,
        )

        if verbose:
            print("Initialization: farthest-point using Soft-DTW divergence")

    else:
        # Cuturi/Lloyd-style version:
        # random sample initialization, no divergence in initialization
        
        barycenters = initialize_barycenters_lloyd_random(
            X_used,
            k=k,
        )

        if verbose:
            print("Initialization: Lloyd-style random sample initialization")

    # labels = np.full(n_series, -1, dtype=int)

    labels = np.full(n_series, -1, dtype=int)

    for it in range(max_iter):
        distances = np.zeros((n_series, k), dtype=np.float64)

        for i, x in enumerate(X_used):
            distances[i] = distances_to_barycenters(
                ts=x,
                barycenters=barycenters,
                gamma=gamma,
                use_divergence=use_divergence,
            )

        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, new_labels):
            if verbose:
                print(f"Converged at iteration {it}")
            break

        labels = new_labels.copy()

        if min_cluster_size > 1:
            labels, barycenters = repair_small_clusters(
                X=X_used,
                labels=labels,
                barycenters=barycenters,
                distances=distances,
                min_cluster_size=min_cluster_size,
            )

        for cluster_id in range(k):
            members = [X_used[i] for i in range(n_series) if labels[i] == cluster_id]

            if len(members) > 0:
                barycenters[cluster_id] = sdtw_barycenter(
                    X=members,
                    barycenter_init=barycenters[cluster_id],
                    gamma=gamma,
                    max_iter=barycenter_max_iter,
                )

        if verbose:
            print(f"Iteration {it}: cluster sizes = {np.bincount(labels, minlength=k)}")

    if return_data:
        return labels, barycenters, X_used

    return labels, barycenters


# Backward-compatible alias, if old notebook cells still call it.
soft_dtw_kmeans_improved = soft_dtw_kmeans


def repair_small_clusters(X, labels, barycenters, distances, min_cluster_size=2):
    """
    Repair empty/tiny clusters by reseeding them with poorly represented points.

    This is optional. It prevents immediate singleton collapse but can also force
    clusters that may not truly exist in the data.
    """
    k = len(barycenters)
    cluster_sizes = np.bincount(labels, minlength=k)
    bad_clusters = np.where(cluster_sizes < min_cluster_size)[0]

    if len(bad_clusters) == 0:
        return labels, barycenters

    assigned_distances = distances[np.arange(len(X)), labels]
    farthest_points = np.argsort(assigned_distances)[::-1]

    used = set()

    for cluster_id in bad_clusters:
        for idx in farthest_points:
            if idx not in used:
                labels[idx] = cluster_id
                barycenters[cluster_id] = X[idx].copy()
                used.add(idx)
                break

    return labels, barycenters


def get_cluster_members(X_used, labels, cluster_id):
    """Return the time series belonging to cluster_id from the already-prepared X_used."""
    return [X_used[i] for i in np.where(labels == cluster_id)[0]]


def get_cluster_sizes(labels, k):
    """Return cluster sizes as float array, compatible with incremental updates."""
    return np.bincount(labels, minlength=k).astype(np.float64)



# ============================================================
# Diagnostics
# ============================================================

def compare_barycenters(a, b, gamma=1.0):
    """Return RMSE and Soft-DTW divergence between two barycenters."""
    a = to_2d_ts(a)
    b = to_2d_ts(b)

    rmse = float(np.sqrt(np.mean((a.ravel() - b.ravel()) ** 2)))
    div = float(soft_dtw_divergence(a, b, gamma=gamma))

    return {"rmse": rmse, "soft_dtw_divergence": div}


def reconstruct_old_barycenter_from_selected_pseudo_atoms_sdtw(
    info,
    gamma=1.0,
    max_iter=100,
    tol=1e-5,
    normalize_weights=True,
):
    """
    Diagnostic only.

    Reconstruct the old barycenter from the selected pseudo-atoms used by the
    symmetric FDM update. This helps evaluate whether the pseudo-atoms preserve
    the old barycenter.
    """
    old_barycenter = to_2d_ts(info["old_barycenter"]).copy()

    selected_pseudo_atoms = info["atoms_for_update"][:-1]
    selected_weights = np.asarray(info["weights_for_update"][:-1], dtype=np.float64)

    if normalize_weights:
        selected_weights = selected_weights / selected_weights.sum()

    reconstructed = sdtw_barycenter(
        X=selected_pseudo_atoms,
        barycenter_init=old_barycenter.copy(),
        gamma=gamma,
        weights=selected_weights,
        tol=tol,
        max_iter=max_iter,
    )

    metrics = compare_barycenters(reconstructed, old_barycenter, gamma=gamma)

    return reconstructed, metrics, selected_pseudo_atoms, selected_weights



def barycenter_error_metrics(approx, reference, old_reference=None, gamma=1.0, eps=1e-8):
    """
    Compare an approximate barycenter against a reference barycenter.

    Parameters
    ----------
    approx : array
        Approximate barycenter, e.g. updated_sym_fdm.
    reference : array
        Reference barycenter, e.g. exact_barycenter.
    old_reference : array or None
        Previous barycenter before update. If provided, compute relative update error.
    gamma : float
        Soft-DTW gamma.
    eps : float
        Numerical stability term.

    Returns
    -------
    metrics : dict
        RMSE, normalized RMSE, percentage error, bounded error, and D_gamma.
    """

    approx = to_2d_ts(approx)
    reference = to_2d_ts(reference)

    diff = approx.ravel() - reference.ravel()

    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Normalize by reference dynamic range
    ref_range = float(reference.max() - reference.min())
    nrmse_range = rmse / (ref_range + eps)

    # Percentage version
    nrmse_range_percent = 100.0 * nrmse_range

    # Bounded 0-1 version
    # 0 = perfect, 1 = error at least as large as the reference dynamic range
    bounded_error_0_1 = float(min(1.0, nrmse_range))

    # Soft-DTW divergence
    dgamma = float(
        soft_dtw_divergence(
            approx,
            reference,
            gamma=gamma
        )
    )

    metrics = {
        "rmse": rmse,
        "nrmse_range": float(nrmse_range),
        "nrmse_range_percent": float(nrmse_range_percent),
        "bounded_error_0_1": bounded_error_0_1,
        "soft_dtw_divergence": dgamma,
    }

    if old_reference is not None:
        old_reference = to_2d_ts(old_reference)

        true_update_rmse = float(
            np.sqrt(np.mean((reference.ravel() - old_reference.ravel()) ** 2))
        )

        relative_update_error = rmse / (true_update_rmse + eps)

        metrics["true_update_rmse"] = true_update_rmse
        metrics["relative_update_error"] = float(relative_update_error)
        metrics["relative_update_error_percent"] = float(100.0 * relative_update_error)
        metrics["relative_update_error_0_1"] = float(
            relative_update_error / (1.0 + relative_update_error)
        )

    return metrics

# ============================================================
# 5. FDM barycenter update
# ============================================================


def _analytic_band_signal(X_fft, start_k, end_k):
    N = len(X_fft)
    Z = np.zeros(N, dtype=np.complex128)
    Z[start_k:end_k + 1] = X_fft[start_k:end_k + 1]
    return 2.0 * np.fft.ifft(Z)


def _real_band_component(X_fft, start_k, end_k):
    N = len(X_fft)
    Z = np.zeros(N, dtype=np.complex128)

    Z[start_k:end_k + 1] = X_fft[start_k:end_k + 1]
    Z[N - end_k:N - start_k + 1] = X_fft[N - end_k:N - start_k + 1]

    return np.fft.ifft(Z).real


def _is_monotone_phase(analytic_signal, phase_tol=1e-5, valid_ratio=0.98, amp_eps=1e-10):
    analytic_signal = np.asarray(analytic_signal)

    amp = np.abs(analytic_signal)
    max_amp = amp.max()

    if max_amp < amp_eps:
        return False

    phase = np.unwrap(np.angle(analytic_signal))
    dphi = np.diff(phase)

    valid = amp[:-1] > amp_eps * max_amp

    if valid.sum() == 0:
        return False

    return np.mean(dphi[valid] >= -phase_tol) >= valid_ratio


def fdm_decompose_lth(
    signal,
    phase_tol=1e-5,
    valid_ratio=0.98,
    include_nyquist=True,
    min_component_energy=1e-12,
):
    """
    Practical discrete FDM-style low-to-high decomposition.

    This is used only for the symmetric pseudo-atom barycenter update.
    """
    x = to_1d_ts(signal)
    N = len(x)
    X_fft = np.fft.fft(x)

    mean_value = X_fft[0].real / N

    if N % 2 == 0:
        max_positive = N // 2 - 1
        nyquist_bin = N // 2
    else:
        max_positive = (N - 1) // 2
        nyquist_bin = None

    components = []
    bands = []

    k_start = 1

    while k_start <= max_positive:
        best_end = k_start

        for k_end in range(k_start, max_positive + 1):
            analytic = _analytic_band_signal(X_fft, k_start, k_end)

            if _is_monotone_phase(
                analytic_signal=analytic,
                phase_tol=phase_tol,
                valid_ratio=valid_ratio,
            ):
                best_end = k_end

        comp = _real_band_component(X_fft, k_start, best_end)
        energy = float(np.sum(comp ** 2))

        if energy > min_component_energy:
            components.append(comp)
            bands.append((k_start, best_end))

        k_start = best_end + 1

    if include_nyquist and nyquist_bin is not None:
        Z = np.zeros(N, dtype=np.complex128)
        Z[nyquist_bin] = X_fft[nyquist_bin]

        nyquist_comp = np.fft.ifft(Z).real
        energy = float(np.sum(nyquist_comp ** 2))

        if energy > min_component_energy:
            components.append(nyquist_comp)
            bands.append((nyquist_bin, nyquist_bin))

    if len(components) == 0:
        energies = np.array([], dtype=np.float64)
        reconstruction = np.full_like(x, mean_value)
    else:
        energies = np.array([np.sum(c ** 2) for c in components], dtype=np.float64)
        reconstruction = mean_value + np.sum(np.vstack(components), axis=0)

    return {
        "mean": mean_value,
        "components": components,
        "energies": energies,
        "bands": bands,
        "reconstruction": reconstruction,
    }


def select_dominant_fdm_components(
    components,
    energies,
    energy_keep=0.95,
    max_components=None,
    min_energy_fraction=1e-4,
):
    """Select dominant FDM components by cumulative energy."""
    if len(components) == 0:
        return [], np.array([]), np.array([], dtype=int)

    energies = np.asarray(energies, dtype=np.float64)
    total_energy = energies.sum()

    if total_energy <= 0:
        return [], np.array([]), np.array([], dtype=int)

    energy_frac = energies / total_energy
    order = np.argsort(energy_frac)[::-1]

    selected = []
    cumulative = 0.0

    for idx in order:
        if energy_frac[idx] < min_energy_fraction:
            continue

        selected.append(idx)
        cumulative += energy_frac[idx]

        if max_components is not None and len(selected) >= max_components:
            break

        if cumulative >= energy_keep:
            break

    selected = np.asarray(selected, dtype=int)

    selected_components = [components[i] for i in selected]
    selected_energies = energies[selected]
    selected_weights = selected_energies / selected_energies.sum()

    return selected_components, selected_weights, selected


def build_fdm_symmetric_atoms_no_anchor(
    barycenter,
    energy_keep=0.95,
    max_components=None,
    clip_range=(0.0, 1.0),
    phase_tol=1e-5,
    valid_ratio=0.98,
    beta=1.0,
):
    """
    Build symmetric FDM pseudo-atoms around the old barycenter.

    Atoms:
        a_r+ = z_old + beta * y_r
        a_r- = z_old - beta * y_r

    The old barycenter itself is not included as an atom.
    """
    z = to_1d_ts(barycenter)

    fdm = fdm_decompose_lth(
        signal=z,
        phase_tol=phase_tol,
        valid_ratio=valid_ratio,
    )

    selected_components, selected_component_weights, selected_indices = (
        select_dominant_fdm_components(
            components=fdm["components"],
            energies=fdm["energies"],
            energy_keep=energy_keep,
            max_components=max_components,
            min_energy_fraction=0.0,
        )
    )

    atoms = []
    alpha = []

    for comp, pi_r in zip(selected_components, selected_component_weights):
        a_plus = z + beta * comp
        a_minus = z - beta * comp

        if clip_range is not None:
            lo, hi = clip_range
            a_plus = np.clip(a_plus, lo, hi)
            a_minus = np.clip(a_minus, lo, hi)

        atoms.append(a_plus.reshape(-1, 1))
        alpha.append(pi_r / 2.0)

        atoms.append(a_minus.reshape(-1, 1))
        alpha.append(pi_r / 2.0)

    alpha = np.asarray(alpha, dtype=np.float64)

    if len(alpha) == 0:
        raise ValueError("No FDM components were selected.")

    alpha = alpha / alpha.sum()

    info = {
        "fdm": fdm,
        "selected_indices": selected_indices,
        "selected_component_weights": selected_component_weights,
    }

    return atoms, alpha, info


def Update_barycenter_fdm(
    bar,
    cluster_sizes,
    best_cluster,
    new_series_list,
    gamma=1.0,
    max_iter=25,
    energy_keep=0.95,
    max_components=None,
    clip_range=(0.0, 1.0),
    phase_tol=1e-5,
    valid_ratio=0.98,
    beta=1.0,
    return_info=True,
):
    """
    Batch version of the symmetric FDM pseudo-atom update.

    This updates one cluster barycenter using:
        - symmetric FDM pseudo-atoms representing the old cluster history
        - all newly assigned time series for that cluster

    Unlike the single-series update, this function adds all new series at once.
    """
    old_barycenter = to_2d_ts(bar[best_cluster]).copy()
    new_series_list = [to_2d_ts(x) for x in new_series_list]

    if len(new_series_list) == 0:
        if return_info:
            return old_barycenter, {
                "old_barycenter": old_barycenter,
                "updated_barycenter": old_barycenter,
                "best_cluster": best_cluster,
                "n_old": float(cluster_sizes[best_cluster]),
                "n_new": 0,
                "atoms_for_update": [],
                "weights_for_update": np.array([]),
            }
        return old_barycenter

    cluster_sizes = np.asarray(cluster_sizes, dtype=np.float64)
    n_old = float(cluster_sizes[best_cluster])
    n_new = len(new_series_list)

    historical_atoms, alpha_hist, fdm_info = build_fdm_symmetric_atoms_no_anchor(
        barycenter=old_barycenter,
        energy_keep=energy_keep,
        max_components=max_components,
        clip_range=clip_range,
        phase_tol=phase_tol,
        valid_ratio=valid_ratio,
        beta=beta,
    )

    historical_weights = n_old * alpha_hist
    # Each newly assigned time series has weight 1.
    new_weights = np.ones(n_new, dtype=np.float64)
    atoms_for_update = list(historical_atoms) + list(new_series_list)

    
    weights_for_update = np.concatenate([
        historical_weights,
        new_weights,
    ])

    updated_barycenter = sdtw_barycenter(
        X=atoms_for_update,
        barycenter_init=old_barycenter.copy(),
        gamma=gamma,
        weights=weights_for_update,
        max_iter=max_iter,
    )

    bar[best_cluster] = updated_barycenter
    cluster_sizes[best_cluster] += n_new

    if not return_info:
        return updated_barycenter

    info = {
        "old_barycenter": old_barycenter,
        "updated_barycenter": updated_barycenter,
        "best_cluster": best_cluster,
        "n_old": n_old,
        "n_new": n_new,
        "atoms_for_update": atoms_for_update,
        "weights_for_update": weights_for_update,
        "historical_alpha": alpha_hist,
        "fdm_info": fdm_info,
        "energy_keep": energy_keep,
        "max_components": max_components,
    }

    return updated_barycenter, info

# =====================================
# This is the MODE (no stability block)

class Clustering_centralized_no_stability(ElementwiseProblem):
    def __init__(
        self,
        offline_data,
        gamma=1.0,
        max_iter=100,
        barycenter_max_iter=50,
        normalize=True,
        use_divergence=True,
        min_cluster_size=1,
        split=0.70,
        D_star_tp=1.8,
        ):
        self.K_max = 10
        self.offline_data = np.asarray(offline_data, dtype=np.float64)
        self.gamma = gamma
        self.max_iter = max_iter
        self.barycenter_max_iter = barycenter_max_iter
        self.normalize = normalize
        self.use_divergence = use_divergence
        self.min_cluster_size = min_cluster_size
        self.split = split
        # self.random_state = random_state
        self.D_star_tp = D_star_tp
        super().__init__(
            n_var=7,
            n_obj=2,
                    #    ws,   lr,  bs, nd, ep, ly, k
            xl=np.array([5, 0.0001, 8, 2, 50, 1, 3]),
            xu=np.array([20, 0.1, 128, 10, 150, 3, self.K_max])
        )
        # Cache clustering outputs per k
        self.cluster_cache = {}
        # Keep compatibility with your previous code
        self.trained_models = {}
        self.data_for_k_clusters = {k: [] for k in range(2, self.K_max + 1)}
        self.membership = {k: [] for k in range(2, self.K_max + 1)}
        # ------------------------------------------------------------
        # Timing statistics
        # ------------------------------------------------------------
        self.timing = {
            "clustering_sec": 0.0,
            "training_sec": 0.0,
            "validation_sec": 0.0,
            "stability_sec": 0.0,

            "n_clusterings": 0,
            "n_trainings": 0,
            "n_validations": 0,
            "n_stability_evals": 0,

            "clustering_cache_hits": 0,
            "clustering_by_k_sec": {},
        }
    def _repair_architecture(self, x):
        """
        Convert one MODE vector into a valid architecture dictionary.
        """
        architecture = {
            "window": int(round(x[0])),
            "learning_rate": float(x[1]),
            "batch_size": int(round(x[2])),
            "hidden_dim": int(round(x[3])),
            "epochs": int(round(x[4])),
            "layer_dim": int(round(x[5])),
            "num_clusters": int(round(x[6])),
        }
        architecture["num_clusters"] = int(
            np.clip(architecture["num_clusters"], 2, self.K_max))
        return architecture
    
    def _get_or_create_clustering(self, k):
        """
        Run Soft-DTW k-means once per k and cache:
            labels, barycenters, X_used, concatenated barycenters.

        Also records the total Soft-DTW clustering time.
        """

        if k in self.cluster_cache:
            self.timing["clustering_cache_hits"] += 1
            return self.cluster_cache[k]

        # --------------------------------------------------------
        # Time Soft-DTW clustering
        # --------------------------------------------------------
        start = time.perf_counter()

        try:
            labels, barycenters, X_used = soft_dtw_kmeans(
                X=self.offline_data,
                k=k,
                gamma=self.gamma,
                max_iter=self.max_iter,
                barycenter_max_iter=self.barycenter_max_iter,
                normalize=self.normalize,
                use_divergence=self.use_divergence,
                min_cluster_size=self.min_cluster_size,
                return_data=True,
                verbose=False,
            )

        finally:
            elapsed = time.perf_counter() - start

            self.timing["clustering_sec"] += elapsed
            self.timing["n_clusterings"] += 1

            self.timing["clustering_by_k_sec"][int(k)] = (
                self.timing["clustering_by_k_sec"].get(int(k), 0.0)
                + elapsed
            )

        concat = concatenate_barycenters(barycenters)

        self.cluster_cache[k] = {
            "labels": labels,
            "barycenters": barycenters,
            "X_used": X_used,
            "concat": concat,
        }

        # Backward-compatible storage
        self.data_for_k_clusters[k] = [barycenters]
        self.membership[k] = [labels]

        return self.cluster_cache[k]



    def _evaluate(self, x, out, *args, **kwargs):

        architecture = self._repair_architecture(x)
        k = architecture["num_clusters"]

        try:

            # ----------------------------------------------------
            # 1. Soft-DTW clustering
            # Timing is handled inside _get_or_create_clustering()
            # ----------------------------------------------------
            cache = self._get_or_create_clustering(k)

            barycenters = cache["barycenters"]
            tr_data = cache["concat"]

            # ----------------------------------------------------
            # 2. LSTM training
            # ----------------------------------------------------
            _sync_device()
            start = time.perf_counter()

            try:
                ML_current = train_model_from_barycenters(
                    barycenters=barycenters,
                    architecture=architecture,
                    split=self.split,
                    normalized=self.normalize,
                )

            finally:
                _sync_device()

                elapsed = time.perf_counter() - start

                self.timing["training_sec"] += elapsed
                self.timing["n_trainings"] += 1

            # ----------------------------------------------------
            # 3. Validation
            # ----------------------------------------------------
            _sync_device()
            start = time.perf_counter()

            try:
                val_output = ML_current.validation_phase()

            finally:
                _sync_device()

                elapsed = time.perf_counter() - start

                self.timing["validation_sec"] += elapsed
                self.timing["n_validations"] += 1

            val_rmse = float(val_output[0])

           
            # ----------------------------------------------------
            # 5. Representation-size objective
            # ----------------------------------------------------
            num_clusters_objective = float(k)

            out["F"] = [
                val_rmse,
                num_clusters_objective,
            ]

            self.trained_models[
                tuple(np.asarray(x, dtype=float))
            ] = ML_current

        except Exception as e:

            import traceback

            error_message = traceback.format_exc()

            print("\n" + "=" * 80)
            print(
                "ERROR INSIDE "
                "Clustering_centralized._evaluate()"
            )
            print("Chromosome x:", x)
            print(
                "Architecture:",
                architecture
                if "architecture" in locals()
                else None,
            )
            print(error_message)
            print("=" * 80)

            if not hasattr(self, "evaluation_errors"):
                self.evaluation_errors = []

            self.evaluation_errors.append({
                "x": np.asarray(
                    x,
                    dtype=float
                ).copy(),
                "architecture": (
                    architecture
                    if "architecture" in locals()
                    else None
                ),
                "error": repr(e),
                "traceback": error_message,
            })

            out["F"] = [
                1e6,
                1e6,
            ]

            self.trained_models[
                tuple(np.asarray(x, dtype=float))
            ] = None


    
def ApplyingMODE_no_stability(
    data,
    num_population,
    num_offsprings,
    num_generations,
    gamma=1.0,
    max_iter=100,
    barycenter_max_iter=50,
    normalize=True,
    use_divergence=True,
    min_cluster_size=1,
    split=0.70,
    # random_state=42,
    D_star_tp=1.8,
):
    """
    Hyperparameter tuning with MODE / NSGA-II.

    Returns
    -------
    results_cl_centralized : pymoo result object
    Problem : fitted Clustering_centralized problem instance
    duration : float
        Optimization duration in seconds.
    """
    Problem = Clustering_centralized_no_stability(
        offline_data=data,
        gamma=gamma,
        max_iter=max_iter,
        barycenter_max_iter=barycenter_max_iter,
        normalize=normalize,
        use_divergence=use_divergence,
        min_cluster_size=min_cluster_size,
        split=split,
        # random_state=random_state,
        D_star_tp=D_star_tp,)
    Algorithm = NSGA2(
        pop_size=num_population,
        n_offsprings=num_offsprings,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,)
    Termination_criterion = get_termination(
        "n_gen",
        num_generations)
    # start = time.time()?
    _sync_device()
    start = time.perf_counter()
    results_cl_centralized = pymoo_minimize(
        Problem,
        Algorithm,
        Termination_criterion,
        # seed=random_state,
        save_history=True,
        verbose=True,)
    # duration = time.time() - start
    _sync_device()
    duration = time.perf_counter() - start
    # ------------------------------------------------------------
    # Final timing summary
    # ------------------------------------------------------------
    Problem.timing["total_mode_sec"] = duration

    measured_components = (
        Problem.timing["clustering_sec"]
        + Problem.timing["training_sec"]
        + Problem.timing["validation_sec"]
        + Problem.timing["stability_sec"]
    )

    Problem.timing["other_sec"] = max(
        0.0,
        duration - measured_components
    )

    Problem.timing["measured_components_sec"] = (
        measured_components
    )
    return results_cl_centralized, Problem, duration  


def ApplyingASF_no_stability(
    results_of_tuning,
    Problem_class,
    weights=np.array([0.8, 0.2]),
    return_full=True,
):
    """
    Select one solution from the Pareto front using ASF.

    Important:
    The new framework stores the full trained Instance wrapper.
    Therefore this function returns:

        best_instance   -> full Instance wrapper
        best_model      -> PyTorch model only
        best_chromosome -> architecture dictionary
        best_info       -> useful clustering/model metadata

    Parameters
    ----------
    results_of_tuning : pymoo result
        Output of ApplyingMODE.

    Problem_class : Clustering_centralized
        The fitted problem instance returned by ApplyingMODE.

    weights : np.ndarray
        ASF preference weights for the objectives.

    return_full : bool
        If True, return best_info as fourth output.
    """

    F = np.asarray(results_of_tuning.F, dtype=np.float64)
    X = np.asarray(results_of_tuning.X, dtype=np.float64)

    if F.ndim == 1:
        F = F.reshape(1, -1)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    # ------------------------------------------------------------
    # Normalize objective values
    # ------------------------------------------------------------
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    denom = np.maximum(
        approx_nadir - approx_ideal,
        1e-12
    )

    nF = (F - approx_ideal) / denom

    # ------------------------------------------------------------
    # ASF selection
    # ------------------------------------------------------------
    weights = np.asarray(weights, dtype=np.float64)

    if len(weights) != F.shape[1]:
        raise ValueError(
            f"weights has length {len(weights)}, "
            f"but the problem has {F.shape[1]} objectives."
        )

    decomp = ASF()

    best_idx = decomp.do(
        nF,
        1.0 / weights
    ).argmin()

    best_x = X[best_idx]
    best_F = F[best_idx]

    # ------------------------------------------------------------
    # Convert selected vector to architecture
    # ------------------------------------------------------------
    if hasattr(Problem_class, "_repair_architecture"):
        best_chromosome = Problem_class._repair_architecture(best_x)
    else:
        best_chromosome = {
            "window": int(round(best_x[0])),
            "learning_rate": float(best_x[1]),
            "batch_size": int(round(best_x[2])),
            "hidden_dim": int(round(best_x[3])),
            "epochs": int(round(best_x[4])),
            "layer_dim": int(round(best_x[5])),
            "num_clusters": int(round(best_x[6])),
        }

    # ------------------------------------------------------------
    # Retrieve trained Instance
    # ------------------------------------------------------------
    key = tuple(np.asarray(best_x, dtype=float))

    best_instance = Problem_class.trained_models.get(key, None)

    # Fallback: sometimes float keys may differ slightly
    if best_instance is None:
        available_keys = list(Problem_class.trained_models.keys())

        if len(available_keys) == 0:
            raise ValueError(
                "No trained models were stored in Problem_class.trained_models."
            )

        key_array = np.asarray(available_keys, dtype=np.float64)

        distances = np.linalg.norm(
            key_array - best_x.reshape(1, -1),
            axis=1
        )

        nearest_idx = int(np.argmin(distances))
        nearest_key = available_keys[nearest_idx]

        best_instance = Problem_class.trained_models[nearest_key]

    if best_instance is None:
        raise ValueError(
            "The selected ASF solution has no trained model instance. "
            "Check whether _evaluate() failed for this chromosome."
        )

    # ------------------------------------------------------------
    # Extract PyTorch model from Instance
    # ------------------------------------------------------------
    best_model = best_instance.model

    # ------------------------------------------------------------
    # Retrieve clustering metadata for selected k
    # ------------------------------------------------------------
    k = best_chromosome["num_clusters"]

    best_info = {
        "best_index": int(best_idx),
        "best_x": best_x,
        "best_F": best_F,
        "best_normalized_F": nF[best_idx],
        "weights": weights,
        "best_instance": best_instance,
        "best_model": best_model,
        "best_chromosome": best_chromosome,
    }

    if hasattr(Problem_class, "cluster_cache") and k in Problem_class.cluster_cache:
        cache = Problem_class.cluster_cache[k]

        best_info.update({
            "labels": cache["labels"],
            "barycenters": cache["barycenters"],
            "X_used": cache["X_used"],
            "concat": cache["concat"],
        })

    # print("\nBest model architecture:")
    # print(best_chromosome)

    # print("\nBest objective values:")
    # print(best_F)

    if return_full:
        return best_instance, best_model, best_chromosome, best_info

    return best_instance, best_model, best_chromosome







# =====================================


# ============================================================
# 6. LSTM model and training / MODE
# ============================================================

def train_model_from_full_series(
    series,
    architecture,
    split=0.70,
    normalized=True,
):
    """
    Train exactly the same LSTM used by MODE, but directly on the
    complete original collection of time series rather than on
    Soft-DTW barycenters.

    Intended for external HPO baselines such as Cao-MOO-LSTM.
    """

    series = np.asarray(series, dtype=np.float64)

    if series.ndim != 2:
        raise ValueError(
            f"Expected series with shape (n_series, series_length), "
            f"got {series.shape}"
        )

    # --------------------------------------------------------
    # Use the same per-series normalization as MODE uses
    # before Soft-DTW clustering.
    # --------------------------------------------------------
    if normalized:
        series_used = minmax_normalize_ts(series)
    else:
        series_used = series.copy()

    series_length = series_used.shape[1]

    # Concatenate the original time series.
    Y = series_used.reshape(-1)

    # Same split convention currently used by MODE.
    split_idx = int(split * len(Y))

    Train = Y[:split_idx]
    Val = Y[split_idx:]

    ML = Instance(
        train_data=Train,
        val_data=Val,
        chromosome=architecture,
        NORMALIZED=normalized,
        series_length=series_length,
        train_offset=0,
        val_offset=split_idx,
    )

    ML.train_phase()

    return ML








class Clustering_centralized(ElementwiseProblem):
    def __init__(
        self,
        offline_data,
        gamma=1.0,
        max_iter=100,
        barycenter_max_iter=50,
        normalize=True,
        use_divergence=True,
        min_cluster_size=1,
        split=0.70,
        D_star_tp=1.8,
        ):
        self.K_max = 10
        self.offline_data = np.asarray(offline_data, dtype=np.float64)
        self.gamma = gamma
        self.max_iter = max_iter
        self.barycenter_max_iter = barycenter_max_iter
        self.normalize = normalize
        self.use_divergence = use_divergence
        self.min_cluster_size = min_cluster_size
        self.split = split
        # self.random_state = random_state
        self.D_star_tp = D_star_tp
        super().__init__(
            n_var=7,
            n_obj=3,
                    #    ws,   lr,  bs, nd, ep, ly, k
            xl=np.array([5, 0.0001, 8, 2, 50, 1, 3]),
            xu=np.array([20, 0.1, 128, 10, 150, 3, self.K_max])
        )
        # Cache clustering outputs per k
        self.cluster_cache = {}
        # Keep compatibility with your previous code
        self.trained_models = {}
        self.data_for_k_clusters = {k: [] for k in range(2, self.K_max + 1)}
        self.membership = {k: [] for k in range(2, self.K_max + 1)}
        # ------------------------------------------------------------
        # Timing statistics
        # ------------------------------------------------------------
        self.timing = {
            "clustering_sec": 0.0,
            "training_sec": 0.0,
            "validation_sec": 0.0,
            "stability_sec": 0.0,

            "n_clusterings": 0,
            "n_trainings": 0,
            "n_validations": 0,
            "n_stability_evals": 0,

            "clustering_cache_hits": 0,
            "clustering_by_k_sec": {},
        }
    def _repair_architecture(self, x):
        """
        Convert one MODE vector into a valid architecture dictionary.
        """
        architecture = {
            "window": int(round(x[0])),
            "learning_rate": float(x[1]),
            "batch_size": int(round(x[2])),
            "hidden_dim": int(round(x[3])),
            "epochs": int(round(x[4])),
            "layer_dim": int(round(x[5])),
            "num_clusters": int(round(x[6])),
        }
        architecture["num_clusters"] = int(
            np.clip(architecture["num_clusters"], 2, self.K_max))
        return architecture
    
    # ------------------------------------------------------------
    # def _get_or_create_clustering(self, k):
    #     """
    #     Run Soft-DTW k-means once per k and cache:
    #         labels, barycenters, X_used, concatenated barycenters.
    #     """
    #     if k not in self.cluster_cache:
    #         labels, barycenters, X_used = soft_dtw_kmeans(
    #             X=self.offline_data,
    #             k=k,
    #             gamma=self.gamma,
    #             max_iter=self.max_iter,
    #             barycenter_max_iter=self.barycenter_max_iter,
    #             normalize=self.normalize,
    #             use_divergence=self.use_divergence,
    #             min_cluster_size=self.min_cluster_size,
    #             return_data=True,
    #             verbose=False,
    #         )
    #         concat = concatenate_barycenters(barycenters)
    #         self.cluster_cache[k] = {
    #             "labels": labels,
    #             "barycenters": barycenters,
    #             "X_used": X_used,
    #             "concat": concat,
    #         }
    #         # Backward-compatible storage
    #         self.data_for_k_clusters[k] = [barycenters]
    #         self.membership[k] = [labels]
    #     return self.cluster_cache[k]
    
    def _get_or_create_clustering(self, k):
        """
        Run Soft-DTW k-means once per k and cache:
            labels, barycenters, X_used, concatenated barycenters.

        Also records the total Soft-DTW clustering time.
        """

        if k in self.cluster_cache:
            self.timing["clustering_cache_hits"] += 1
            return self.cluster_cache[k]

        # --------------------------------------------------------
        # Time Soft-DTW clustering
        # --------------------------------------------------------
        start = time.perf_counter()

        try:
            labels, barycenters, X_used = soft_dtw_kmeans(
                X=self.offline_data,
                k=k,
                gamma=self.gamma,
                max_iter=self.max_iter,
                barycenter_max_iter=self.barycenter_max_iter,
                normalize=self.normalize,
                use_divergence=self.use_divergence,
                min_cluster_size=self.min_cluster_size,
                return_data=True,
                verbose=False,
            )

        finally:
            elapsed = time.perf_counter() - start

            self.timing["clustering_sec"] += elapsed
            self.timing["n_clusterings"] += 1

            self.timing["clustering_by_k_sec"][int(k)] = (
                self.timing["clustering_by_k_sec"].get(int(k), 0.0)
                + elapsed
            )

        concat = concatenate_barycenters(barycenters)

        self.cluster_cache[k] = {
            "labels": labels,
            "barycenters": barycenters,
            "X_used": X_used,
            "concat": concat,
        }

        # Backward-compatible storage
        self.data_for_k_clusters[k] = [barycenters]
        self.membership[k] = [labels]

        return self.cluster_cache[k]


    # ------------------------------------------------------------
    def _collect_prediction_errors(self, ML_current):
        """
        Evaluate the trained LSTM on each original time series and collect residuals.

        If self.normalize=True:
            each series is normalized before evaluation, consistent with
            normalized Soft-DTW barycenters.

        The residuals are computed in the same 'used' scale as the model.
        """
        all_errors = []
        for ts in self.offline_data:
            try:
                res = evaluate_one_individual_series(
                    ML=ML_current,
                    series_raw=ts,
                    sdtw_normalize=self.normalize,
                    return_raw_scale=False,)
                errors = res["preds_used"] - res["targets_used"]
                all_errors.append(errors)
            except ValueError:
                # If a series is too short for the selected window, penalize later.
                continue
        if len(all_errors) == 0:
            return None
        all_errors = np.concatenate(all_errors)
        all_errors = all_errors - np.mean(all_errors)
        return all_errors
    
    # ------------------------------------------------------------
    def _variance_instability_score(self, errors):
        """
        Compute the residual-instability objective using your change-point logic.

        Returns
        -------
        difference : float
            Variance difference between high-variance and low-variance residual regimes.
            Larger values indicate less stable residual behavior.
        """
        if errors is None or len(errors) < 30:
            return 1e6
        detector = ChangePointDetector(errors)
        change_points = detector.aL_bs_genCov(self.D_star_tp)
        all_changepoints = [0] + sorted(change_points) + [len(errors)]
        variance_segment_pairs = []
        for i in range(len(all_changepoints) - 1):
            start = all_changepoints[i]
            end = all_changepoints[i + 1]
            segment = errors[start:end]
            if len(segment) <= 1:
                continue
            var = np.var(segment, ddof=1)
            variance_segment_pairs.append((var, segment))
        if len(variance_segment_pairs) == 0:
            return 0.0
        variance_segment_pairs.sort(key=lambda x: x[0])
        sorted_segments = [seg for _, seg in variance_segment_pairs]

        def grow_until_cp(segments, D_star_tp, reverse=False):
            accumulated = []
            iterable = reversed(segments) if reverse else segments
            for seg in iterable:
                accumulated.append(seg)
                combined = np.concatenate(accumulated)
                # if len(combined) < 30:
                #     continue
                detector = ChangePointDetector(combined)
                cps = np.asarray(detector.aL_bs_genCov(D_star_tp))
                valid_cps = cps[(cps > 0) & (cps < len(combined) - 1)]
                if len(valid_cps) > 0:
                    return combined, np.min(valid_cps)
            return np.concatenate(accumulated), None
        L, cp_L = grow_until_cp(
            sorted_segments,
            self.D_star_tp,
            reverse=False,)
        S, cp_S = grow_until_cp(
            sorted_segments,
            self.D_star_tp,
            reverse=True,)
        if cp_L is None or cp_S is None:
            return 0.0
        var_L = np.var(L, ddof=1)
        var_S = np.var(S, ddof=1)
        difference = var_S - var_L
        return float(max(0.0, difference))
    
    # # ------------------------------------------------------------
    # def _evaluate(self, x, out, *args, **kwargs):
    #     architecture = self._repair_architecture(x)
    #     k = architecture["num_clusters"]
    #     try:
    #         # ----------------------------------------------------
    #         # 1. Get cached barycenters for this k
    #         # ----------------------------------------------------
    #         cache = self._get_or_create_clustering(k)
    #         barycenters = cache["barycenters"]
    #         tr_data = cache["concat"]
    #         # ----------------------------------------------------
    #         # 2. Train LSTM from barycenters
    #         # ----------------------------------------------------
    #         ML_current = train_model_from_barycenters(
    #             barycenters=barycenters,
    #             architecture=architecture,
    #             split=self.split,
    #             normalized=self.normalize)
    #         # ----------------------------------------------------
    #         # 3. Internal validation RMSE on concatenated barycenters
    #         # ----------------------------------------------------
    #         val_output = ML_current.validation_phase()
    #         val_rmse = float(val_output[0])
    #         # ----------------------------------------------------
    #         # 4. Residual stability objective
    #         # ----------------------------------------------------
    #         errors = self._collect_prediction_errors(ML_current)
    #         instability_score = self._variance_instability_score(errors)
    #         # ----------------------------------------------------
    #         # 5. Representation size objective
    #         # ----------------------------------------------------
    #         # representation_size = float(len(tr_data))
    #         num_clusters_objective = float(k)
    #         out["F"] = [val_rmse,num_clusters_objective,instability_score,]
    #         self.trained_models[tuple(np.asarray(x, dtype=float))] = ML_current
    #     except Exception as e:
    #         import traceback

    #         error_message = traceback.format_exc()

    #         print("\n" + "=" * 80)
    #         print("ERROR INSIDE Clustering_centralized._evaluate()")
    #         print("Chromosome x:", x)
    #         print("Architecture:", architecture if "architecture" in locals() else None)
    #         print(error_message)
    #         print("=" * 80)

    #         # Store error for later inspection
    #         if not hasattr(self, "evaluation_errors"):
    #             self.evaluation_errors = []

    #         self.evaluation_errors.append({
    #             "x": np.asarray(x, dtype=float).copy(),
    #             "architecture": architecture if "architecture" in locals() else None,
    #             "error": repr(e),
    #             "traceback": error_message,
    #         })

    #         out["F"] = [
    #             1e6,
    #             1e6,
    #             1e6,
    #         ]

    #         self.trained_models[tuple(np.asarray(x, dtype=float))] = None


    def _evaluate(self, x, out, *args, **kwargs):

        architecture = self._repair_architecture(x)
        k = architecture["num_clusters"]

        try:

            # ----------------------------------------------------
            # 1. Soft-DTW clustering
            # Timing is handled inside _get_or_create_clustering()
            # ----------------------------------------------------
            cache = self._get_or_create_clustering(k)

            barycenters = cache["barycenters"]
            tr_data = cache["concat"]

            # ----------------------------------------------------
            # 2. LSTM training
            # ----------------------------------------------------
            _sync_device()
            start = time.perf_counter()

            try:
                ML_current = train_model_from_barycenters(
                    barycenters=barycenters,
                    architecture=architecture,
                    split=self.split,
                    normalized=self.normalize,
                )

            finally:
                _sync_device()

                elapsed = time.perf_counter() - start

                self.timing["training_sec"] += elapsed
                self.timing["n_trainings"] += 1

            # ----------------------------------------------------
            # 3. Validation
            # ----------------------------------------------------
            _sync_device()
            start = time.perf_counter()

            try:
                val_output = ML_current.validation_phase()

            finally:
                _sync_device()

                elapsed = time.perf_counter() - start

                self.timing["validation_sec"] += elapsed
                self.timing["n_validations"] += 1

            val_rmse = float(val_output[0])

            # ----------------------------------------------------
            # 4. Stability component
            # ----------------------------------------------------
            _sync_device()
            start = time.perf_counter()

            try:
                errors = self._collect_prediction_errors(
                    ML_current
                )

                instability_score = (
                    self._variance_instability_score(errors)
                )

            finally:
                _sync_device()

                elapsed = time.perf_counter() - start

                self.timing["stability_sec"] += elapsed
                self.timing["n_stability_evals"] += 1

            # ----------------------------------------------------
            # 5. Representation-size objective
            # ----------------------------------------------------
            num_clusters_objective = float(k)

            out["F"] = [
                val_rmse,
                num_clusters_objective,
                instability_score,
            ]

            self.trained_models[
                tuple(np.asarray(x, dtype=float))
            ] = ML_current

        except Exception as e:

            import traceback

            error_message = traceback.format_exc()

            print("\n" + "=" * 80)
            print(
                "ERROR INSIDE "
                "Clustering_centralized._evaluate()"
            )
            print("Chromosome x:", x)
            print(
                "Architecture:",
                architecture
                if "architecture" in locals()
                else None,
            )
            print(error_message)
            print("=" * 80)

            if not hasattr(self, "evaluation_errors"):
                self.evaluation_errors = []

            self.evaluation_errors.append({
                "x": np.asarray(
                    x,
                    dtype=float
                ).copy(),
                "architecture": (
                    architecture
                    if "architecture" in locals()
                    else None
                ),
                "error": repr(e),
                "traceback": error_message,
            })

            out["F"] = [
                1e6,
                1e6,
                1e6,
            ]

            self.trained_models[
                tuple(np.asarray(x, dtype=float))
            ] = None


def compute_instability_on_batch(model_instance, batch, normalize=True, D_star_tp=1.8):
    """
    Reuse Arsenal's stability logic without running a new MODE search.
    This gives the same residual-instability score used by MODE.
    """
    tmp_problem = Clustering_centralized(
        offline_data=batch,
        normalize=normalize,
        D_star_tp=D_star_tp,
    )
    errors = tmp_problem._collect_prediction_errors(model_instance)
    return tmp_problem._variance_instability_score(errors)



    
def ApplyingMODE(
    data,
    num_population,
    num_offsprings,
    num_generations,
    gamma=1.0,
    max_iter=100,
    barycenter_max_iter=50,
    normalize=True,
    use_divergence=True,
    min_cluster_size=1,
    split=0.70,
    # random_state=42,
    D_star_tp=1.8,
):
    """
    Hyperparameter tuning with MODE / NSGA-II.

    Returns
    -------
    results_cl_centralized : pymoo result object
    Problem : fitted Clustering_centralized problem instance
    duration : float
        Optimization duration in seconds.
    """
    Problem = Clustering_centralized(
        offline_data=data,
        gamma=gamma,
        max_iter=max_iter,
        barycenter_max_iter=barycenter_max_iter,
        normalize=normalize,
        use_divergence=use_divergence,
        min_cluster_size=min_cluster_size,
        split=split,
        # random_state=random_state,
        D_star_tp=D_star_tp,)
    Algorithm = NSGA2(
        pop_size=num_population,
        n_offsprings=num_offsprings,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,)
    Termination_criterion = get_termination(
        "n_gen",
        num_generations)
    # start = time.time()?
    _sync_device()
    start = time.perf_counter()
    results_cl_centralized = pymoo_minimize(
        Problem,
        Algorithm,
        Termination_criterion,
        # seed=random_state,
        save_history=True,
        verbose=True,)
    # duration = time.time() - start
    _sync_device()
    duration = time.perf_counter() - start
    # ------------------------------------------------------------
    # Final timing summary
    # ------------------------------------------------------------
    Problem.timing["total_mode_sec"] = duration

    measured_components = (
        Problem.timing["clustering_sec"]
        + Problem.timing["training_sec"]
        + Problem.timing["validation_sec"]
        + Problem.timing["stability_sec"]
    )

    Problem.timing["other_sec"] = max(
        0.0,
        duration - measured_components
    )

    Problem.timing["measured_components_sec"] = (
        measured_components
    )
    return results_cl_centralized, Problem, duration  


def ApplyingASF(
    results_of_tuning,
    Problem_class,
    weights=np.array([0.6, 0.2, 0.2]),
    return_full=True,
):
    """
    Select one solution from the Pareto front using ASF.

    Important:
    The new framework stores the full trained Instance wrapper.
    Therefore this function returns:

        best_instance   -> full Instance wrapper
        best_model      -> PyTorch model only
        best_chromosome -> architecture dictionary
        best_info       -> useful clustering/model metadata

    Parameters
    ----------
    results_of_tuning : pymoo result
        Output of ApplyingMODE.

    Problem_class : Clustering_centralized
        The fitted problem instance returned by ApplyingMODE.

    weights : np.ndarray
        ASF preference weights for the objectives.

    return_full : bool
        If True, return best_info as fourth output.
    """

    F = np.asarray(results_of_tuning.F, dtype=np.float64)
    X = np.asarray(results_of_tuning.X, dtype=np.float64)

    if F.ndim == 1:
        F = F.reshape(1, -1)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    # ------------------------------------------------------------
    # Normalize objective values
    # ------------------------------------------------------------
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    denom = np.maximum(
        approx_nadir - approx_ideal,
        1e-12
    )

    nF = (F - approx_ideal) / denom

    # ------------------------------------------------------------
    # ASF selection
    # ------------------------------------------------------------
    weights = np.asarray(weights, dtype=np.float64)

    if len(weights) != F.shape[1]:
        raise ValueError(
            f"weights has length {len(weights)}, "
            f"but the problem has {F.shape[1]} objectives."
        )

    decomp = ASF()

    best_idx = decomp.do(
        nF,
        1.0 / weights
    ).argmin()

    best_x = X[best_idx]
    best_F = F[best_idx]

    # ------------------------------------------------------------
    # Convert selected vector to architecture
    # ------------------------------------------------------------
    if hasattr(Problem_class, "_repair_architecture"):
        best_chromosome = Problem_class._repair_architecture(best_x)
    else:
        best_chromosome = {
            "window": int(round(best_x[0])),
            "learning_rate": float(best_x[1]),
            "batch_size": int(round(best_x[2])),
            "hidden_dim": int(round(best_x[3])),
            "epochs": int(round(best_x[4])),
            "layer_dim": int(round(best_x[5])),
            "num_clusters": int(round(best_x[6])),
        }

    # ------------------------------------------------------------
    # Retrieve trained Instance
    # ------------------------------------------------------------
    key = tuple(np.asarray(best_x, dtype=float))

    best_instance = Problem_class.trained_models.get(key, None)

    # Fallback: sometimes float keys may differ slightly
    if best_instance is None:
        available_keys = list(Problem_class.trained_models.keys())

        if len(available_keys) == 0:
            raise ValueError(
                "No trained models were stored in Problem_class.trained_models."
            )

        key_array = np.asarray(available_keys, dtype=np.float64)

        distances = np.linalg.norm(
            key_array - best_x.reshape(1, -1),
            axis=1
        )

        nearest_idx = int(np.argmin(distances))
        nearest_key = available_keys[nearest_idx]

        best_instance = Problem_class.trained_models[nearest_key]

    if best_instance is None:
        raise ValueError(
            "The selected ASF solution has no trained model instance. "
            "Check whether _evaluate() failed for this chromosome."
        )

    # ------------------------------------------------------------
    # Extract PyTorch model from Instance
    # ------------------------------------------------------------
    best_model = best_instance.model

    # ------------------------------------------------------------
    # Retrieve clustering metadata for selected k
    # ------------------------------------------------------------
    k = best_chromosome["num_clusters"]

    best_info = {
        "best_index": int(best_idx),
        "best_x": best_x,
        "best_F": best_F,
        "best_normalized_F": nF[best_idx],
        "weights": weights,
        "best_instance": best_instance,
        "best_model": best_model,
        "best_chromosome": best_chromosome,
    }

    if hasattr(Problem_class, "cluster_cache") and k in Problem_class.cluster_cache:
        cache = Problem_class.cluster_cache[k]

        best_info.update({
            "labels": cache["labels"],
            "barycenters": cache["barycenters"],
            "X_used": cache["X_used"],
            "concat": cache["concat"],
        })

    # print("\nBest model architecture:")
    # print(best_chromosome)

    # print("\nBest objective values:")
    # print(best_F)

    if return_full:
        return best_instance, best_model, best_chromosome, best_info

    return best_instance, best_model, best_chromosome



def local_q_bounds(previous_q, neigh_per=0.50, q_min=3, q_max=10):
    """Integer lower/upper bounds for Q around previous_q."""
    lo = int(np.ceil(max(q_min, previous_q * (1.0 - neigh_per))))
    hi = int(np.floor(min(q_max, previous_q * (1.0 + neigh_per))))

    if lo > hi:
        lo = hi = int(np.clip(previous_q, q_min, q_max))

    return lo, hi



class WSQMLModeProblem(ElementwiseProblem):
    """
    MODE-style warm-start problem for WS-Q-ML and WS-All.

    Tuned variables:
        ML hyperparameters and Q.

    Domains:
        local ML intervals around the previous accepted ML hyperparameters,
        and local Q interval around Q_{m-1}.

    Objectives:
        f1 = validation RMSE,
        f2 = Q,
        f3 = residual-instability score.
    """

    def __init__(
        self,
        offline_data,
        previous_chromosome,
        neigh_per=0.50,
        q_min=3,
        q_max=10,
        gamma=1.0,
        max_iter=50,
        barycenter_max_iter=50,
        normalize=True,
        use_divergence=False,
        min_cluster_size=1,
        split=0.70,
        # random_state=42,
        D_star_tp=1.8,
    ):
        self.offline_data = np.asarray(offline_data, dtype=np.float64)
        self.previous_chromosome = previous_chromosome

        self.previous_q = int(previous_chromosome["num_clusters"])
        self.q_min = int(q_min)
        self.q_max = int(q_max)
        self.local_q_min, self.local_q_max = local_q_bounds(
            previous_q=self.previous_q,
            neigh_per=neigh_per,
            q_min=self.q_min,
            q_max=self.q_max,
        )

        ml_xl, ml_xu = build_local_bounds_from_chromosome(
            previous_chromosome=previous_chromosome,
            neigh_per=neigh_per,
        )

        self.gamma = gamma
        self.max_iter = max_iter
        self.barycenter_max_iter = barycenter_max_iter
        self.normalize = normalize
        self.use_divergence = use_divergence
        self.min_cluster_size = min_cluster_size
        self.split = split
        # self.random_state = random_state
        self.D_star_tp = D_star_tp

        self.cluster_cache = {}
        self.trained_models = {}
        self.evaluation_errors = []

        xl = np.concatenate([ml_xl, np.array([self.local_q_min], dtype=float)])
        xu = np.concatenate([ml_xu, np.array([self.local_q_max], dtype=float)])

        super().__init__(
            n_var=7,
            n_obj=3,
            xl=xl,
            xu=xu,
        )

    def _repair_architecture(self, x):
        x = np.asarray(x, dtype=float).ravel()
        architecture = {
            "window": int(round(x[0])),
            "learning_rate": float(x[1]),
            "batch_size": int(round(x[2])),
            "hidden_dim": int(round(x[3])),
            "epochs": int(round(x[4])),
            "layer_dim": int(round(x[5])),
            "num_clusters": int(round(x[6])),
        }

        architecture["window"] = int(np.clip(architecture["window"], 5, 20))
        architecture["learning_rate"] = float(np.clip(architecture["learning_rate"], 0.0001, 0.1))
        architecture["batch_size"] = int(np.clip(architecture["batch_size"], 8, 128))
        architecture["hidden_dim"] = int(np.clip(architecture["hidden_dim"], 2, 10))
        architecture["epochs"] = int(np.clip(architecture["epochs"], 50, 150))
        architecture["layer_dim"] = int(np.clip(architecture["layer_dim"], 1, 3))
        architecture["num_clusters"] = int(np.clip(architecture["num_clusters"], self.local_q_min, self.local_q_max))

        return architecture

    def _get_or_create_clustering(self, q):
        if q not in self.cluster_cache:
            labels, barycenters, X_used = soft_dtw_kmeans(
                X=self.offline_data,
                k=q,
                gamma=self.gamma,
                max_iter=self.max_iter,
                barycenter_max_iter=self.barycenter_max_iter,
                # random_state=self.random_state,
                normalize=self.normalize,
                use_divergence=self.use_divergence,
                min_cluster_size=self.min_cluster_size,
                return_data=True,
                verbose=False,
            )
            self.cluster_cache[q] = {
                "labels": labels,
                "barycenters": barycenters,
                "X_used": X_used,
                "concat": concatenate_barycenters(barycenters),
            }
        return self.cluster_cache[q]

    def _evaluate(self, x, out, *args, **kwargs):
        architecture = self._repair_architecture(x)
        q = int(architecture["num_clusters"])

        try:
            cache = self._get_or_create_clustering(q)
            barycenters = cache["barycenters"]

            model_instance = train_model_from_barycenters(
                barycenters=barycenters,
                architecture=architecture,
                split=self.split,
                normalized=self.normalize,
                # seed=self.random_state,
            )

            val_rmse = float(model_instance.validation_phase()[0])
            instability = float(compute_instability_on_batch(
                model_instance=model_instance,
                batch=self.offline_data,
                normalize=self.normalize,
                D_star_tp=self.D_star_tp,
            ))

            out["F"] = [val_rmse, float(q), instability]
            self.trained_models[tuple(np.asarray(x, dtype=float))] = model_instance

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("\n" + "=" * 80)
            print("ERROR INSIDE WSQMLModeProblem._evaluate()")
            print("Chromosome x:", x)
            print("Architecture:", architecture)
            print(tb)
            print("=" * 80)

            self.evaluation_errors.append({
                "x": np.asarray(x, dtype=float).copy(),
                "architecture": architecture,
                "error": repr(e),
                "traceback": tb,
            })
            out["F"] = [1e6, 1e6, 1e6]
            self.trained_models[tuple(np.asarray(x, dtype=float))] = None


def ApplyingMODE_WS_Q_ML(
    data,
    previous_chromosome,
    num_population,
    num_offsprings,
    num_generations,
    neigh_per=0.50,
    q_min=3,
    q_max=10,
    gamma=1.0,
    max_iter=50,
    barycenter_max_iter=50,
    normalize=True,
    use_divergence=False,
    min_cluster_size=1,
    split=0.70,
    # random_state=42,
    D_star_tp=1.8,):
    """Run NSGA-II/MODE-style WS-Q-ML or WS-All over local ML and local Q domains."""

    Problem = WSQMLModeProblem(
        offline_data=data,
        previous_chromosome=previous_chromosome,
        neigh_per=neigh_per,
        q_min=q_min,
        q_max=q_max,
        gamma=gamma,
        max_iter=max_iter,
        barycenter_max_iter=barycenter_max_iter,
        normalize=normalize,
        use_divergence=use_divergence,
        min_cluster_size=min_cluster_size,
        split=split,
        # random_state=random_state,
        D_star_tp=D_star_tp,
    )

    Algorithm = NSGA2(
        pop_size=num_population,
        n_offsprings=num_offsprings,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    Termination_criterion = get_termination("n_gen", num_generations)

    start = time.time()
    results = pymoo_minimize(
        Problem,
        Algorithm,
        Termination_criterion,
        # seed=random_state,
        save_history=True,
        verbose=True,
    )
    duration = time.time() - start

    return results, Problem, duration



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# def create_sequences(data, seq_length):
#     x = []
#     y = []
#     for i in range(len(data)-seq_length):
#         x.append(data[i:(i+seq_length)])
#         y.append(data[i+seq_length])
#     return np.array(x), np.array(y)

def create_sequences(data, seq_length, series_length=None, offset=0):
    """
    Create one-step-ahead LSTM samples.

    If series_length is provided, samples crossing boundaries between
    concatenated barycenters are excluded.

    Parameters
    ----------
    data : array-like
        Current data segment (training or validation).

    seq_length : int
        LSTM input-window length.

    series_length : int or None
        Length of each original barycenter.

    offset : int
        Global starting index of `data` within the full concatenated
        barycenter sequence. For training this is normally 0; for
        validation it is the train/validation split index.
    """

    data = np.asarray(data)

    x = []
    y = []

    for i in range(len(data) - seq_length):

        target_idx = i + seq_length

        if series_length is not None:

            # Convert local positions to positions in the complete
            # concatenated barycenter sequence.
            global_start = offset + i
            global_target = offset + target_idx

            # Identify which barycenter contains the first input point
            # and which barycenter contains the prediction target.
            start_barycenter = global_start // series_length
            target_barycenter = global_target // series_length

            # If they differ, the sample crosses an artificial boundary.
            if start_barycenter != target_barycenter:
                continue

        x.append(data[i:i + seq_length])
        y.append(data[target_idx])

    return np.array(x), np.array(y)


# class Instance:
    # def __init__(self, train_data, val_data, chromosome, NORMALIZED = True):
class Instance:
    def __init__(
        self,
        train_data,
        val_data,
        chromosome,
        NORMALIZED=True,
        series_length=None,
        train_offset=0,
        val_offset=0,
    ):

        self.NORMAL = NORMALIZED
        self.chromosome = chromosome

        self.series_length = series_length
        self.train_offset = train_offset
        self.val_offset = val_offset

        if self.NORMAL == True:
            self.Y_tr = train_data
            self.Y_va = val_data
        else:
            self.scaler = MinMaxScaler()

            self.Y_tr = self.scaler.fit_transform(
                np.array(train_data).reshape(-1, 1)
            ).flatten()

            self.Y_va = self.scaler.transform(
                np.array(val_data).reshape(-1, 1)
            ).flatten()

        self.model = LSTMModel(
            input_dim=1,
            hidden_dim=self.chromosome["hidden_dim"],
            layer_dim=self.chromosome["layer_dim"],
            output_dim=1
        ).to(device)

        self.train_losses = []
        self.test_preds = []
        self.test_targets = []

    def get_dataloader(self, train_stage = True):
        batch = self.chromosome['batch_size']
        window = self.chromosome['window']
        # data = self.Y_tr if train_stage else self.Y_va
        # X, y = create_sequences(data, window)
        if train_stage:
            data = self.Y_tr
            offset = self.train_offset
        else:
            data = self.Y_va
            offset = self.val_offset

        X, y = create_sequences(
            data=data,
            seq_length=window,
            series_length=self.series_length,
            offset=offset,
        )
        if len(X) == 0:
            raise ValueError(
                f"Not enough data to create sequences. "
                f"Data length={len(data)}, window={window}."
            )
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset=dataset,batch_size=batch,shuffle=train_stage,drop_last=False)
        return dataloader

    def train_phase(self):
        epochs = self.chromosome['epochs']
        lr = self.chromosome['learning_rate']
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        train_loader = self.get_dataloader(train_stage=True)
        self.train_losses = []
        self.val_losses = []
        for _ in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X, y in train_loader:
                x_batch = X.unsqueeze(-1).to(device)
                y_batch = y.unsqueeze(1).to(device)
                y_pred = self.model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
    def validation_phase(self):
        self.test_preds = []
        self.test_targets = []
        loader = self.get_dataloader(train_stage=False)
        self.model.eval()
        with torch.inference_mode():
            for X, y in loader:
                x_batch = X.unsqueeze(-1).to(device)
                y_batch = y.unsqueeze(1).to(device)
                preds = self.model(x_batch)
                self.test_preds.append(preds.cpu().numpy())
                self.test_targets.append(y_batch.cpu().numpy())
        self.test_preds = np.concatenate(self.test_preds).flatten()
        self.test_targets = np.concatenate(self.test_targets).flatten()
        if self.NORMAL:
            # Already in normalized barycenter space.
            preds_original = self.test_preds
            targets_original = self.test_targets
        else:
            # Model predictions are in scaler space, return to raw space.
            preds_original = self.scaler.inverse_transform(self.test_preds.reshape(-1, 1)).flatten()
            targets_original = self.scaler.inverse_transform(self.test_targets.reshape(-1, 1)).flatten()
        rmse = np.sqrt(np.mean((preds_original - targets_original) ** 2))
        ss_res = np.sum((targets_original - preds_original) ** 2)
        ss_tot = np.sum((targets_original - targets_original.mean()) ** 2)
        r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)
        return rmse, r2, preds_original, targets_original, self.test_preds, self.test_targets         
           
def concatenate_barycenters(barycenters):
    """
    Convert list/array of barycenters into one long 1D sequence.
    Works whether barycenters is:
        - list of arrays
        - numpy array of shape (K, T, 1)
        - numpy array of shape (K, T)
    """

    if isinstance(barycenters, np.ndarray):
        return barycenters.reshape(-1)

    return np.concatenate([
        np.asarray(b).ravel()
        for b in barycenters
    ])






# def train_model_from_barycenters(barycenters,architecture,split=0.7,normalized=True):
#     # set_seed(seed)
#     Y = concatenate_barycenters(barycenters)
#     split_idx = int(split * len(Y))
#     Train = Y[:split_idx]
#     Val = Y[split_idx:]
#     ML = Instance(train_data=Train,val_data=Val,chromosome=architecture,NORMALIZED=normalized)
#     ML.train_phase()
#     return ML

def train_model_from_barycenters(
    barycenters,
    architecture,
    split=0.7,
    normalized=True,
):
    """
    Train the LSTM on concatenated barycenters while excluding
    windows that cross artificial barycenter boundaries.
    """

    # Every barycenter has the same known length.
    series_length = len(
        np.asarray(barycenters[0]).ravel()
    )

    # Concatenate exactly as before.
    Y = concatenate_barycenters(barycenters)

    # Same train/validation split.
    split_idx = int(split * len(Y))

    Train = Y[:split_idx]
    Val = Y[split_idx:]

    ML = Instance(
        train_data=Train,
        val_data=Val,
        chromosome=architecture,
        NORMALIZED=normalized,
        series_length=series_length,
        train_offset=0,
        val_offset=split_idx,
    )

    ML.train_phase()

    return ML



def evaluate_one_individual_series(
    ML,
    series_raw,
    sdtw_normalize=True,
    return_raw_scale=True,
    eps=1e-8
):
    """
    Evaluate a trained LSTM model on one individual time series.

    Meaning of ML.NORMAL
    --------------------
    ML.NORMAL = True:
        The LSTM was trained on already-normalized barycenters.
        No additional LSTM scaler was used.

    ML.NORMAL = False:
        The LSTM was trained on raw barycenters, but internally scaled
        them using ML.scaler for numerical stability.

    Meaning of sdtw_normalize
    -------------------------
    sdtw_normalize = True:
        Normalize this individual raw time series independently before
        prediction, consistent with Soft-DTW normalized barycenters.

    sdtw_normalize = False:
        Use this individual time series in raw scale.
    """

    series_raw = np.asarray(series_raw, dtype=np.float64).ravel()
    if len(series_raw) == 0:
        raise ValueError("series_raw is empty.")

    # ------------------------------------------------------------
    # 1. Soft-DTW-style per-series normalization
    # ------------------------------------------------------------
    if sdtw_normalize:
        s_min = series_raw.min()
        s_max = series_raw.max()
        s_range = s_max - s_min

        series_used = (series_raw - s_min) / (s_range + eps)
    else:
        s_min = None
        s_max = None
        s_range = None

        series_used = series_raw.copy()

    # ------------------------------------------------------------
    # 2. Apply LSTM internal scaler only if ML.NORMAL=False
    # ------------------------------------------------------------
    if ML.NORMAL:
        # Already in the scale used by the LSTM.
        series_model_scale = series_used.copy()

    else:
        # Raw-space model: the LSTM used an internal MinMaxScaler.
        if not hasattr(ML, "scaler") or ML.scaler is None:
            raise ValueError(
                "ML.NORMAL=False, but ML.scaler does not exist. "
                "The model must have a fitted scaler for raw-scale evaluation."
            )
        series_model_scale = ML.scaler.transform(series_used.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------
    # 3. Create supervised sequences
    # ------------------------------------------------------------
    window = ML.chromosome["window"]
    X, y = create_sequences(series_model_scale, window)
    if len(X) == 0:
        raise ValueError(
            f"Series too short for evaluation. "
            f"Length={len(series_model_scale)}, window={window}"
        )
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)

    # ------------------------------------------------------------
    # 4. Predict
    # ------------------------------------------------------------
    ML.model.eval()
    with torch.inference_mode():
        preds_model_scale = ML.model(X_tensor).cpu().numpy().flatten()
    targets_model_scale = y.flatten()

    # ------------------------------------------------------------
    # 5. Return from LSTM model scale to evaluation-used scale
    # ------------------------------------------------------------
    if ML.NORMAL:
        # No internal scaler was applied.
        preds_used = preds_model_scale.copy()
        targets_used = targets_model_scale.copy()
    else:
        preds_used = ML.scaler.inverse_transform(
            preds_model_scale.reshape(-1, 1)
        ).flatten()

        targets_used = ML.scaler.inverse_transform(
            targets_model_scale.reshape(-1, 1)
        ).flatten()

    # ------------------------------------------------------------
    # 6. Metrics in used scale
    # ------------------------------------------------------------
    rmse_used = np.sqrt(np.mean((preds_used - targets_used) ** 2))
    used_range = targets_used.max() - targets_used.min()
    nrmse_used = rmse_used / (used_range + eps)
    results = {
        "rmse_used": float(rmse_used),
        "nrmse_percent_used": float(100 * nrmse_used),
        "preds_used": preds_used,
        "targets_used": targets_used,
        "preds_model_scale": preds_model_scale,
        "targets_model_scale": targets_model_scale,
        "window": window,
        "prediction_time_index": np.arange(
            window,
            window + len(targets_used)
        ),
    }

    # ------------------------------------------------------------
    # 7. Optional conversion back to raw original scale
    # ------------------------------------------------------------
    if return_raw_scale and sdtw_normalize:
        preds_raw = preds_used * (s_range + eps) + s_min
        targets_raw = targets_used * (s_range + eps) + s_min
        rmse_raw = np.sqrt(np.mean((preds_raw - targets_raw) ** 2))
        raw_range = targets_raw.max() - targets_raw.min()
        nrmse_raw = rmse_raw / (raw_range + eps)
        results.update({
            "rmse_raw": float(rmse_raw),
            "nrmse_percent_raw": float(100 * nrmse_raw),
            "preds_raw": preds_raw,
            "targets_raw": targets_raw,
        })

    elif return_raw_scale and not sdtw_normalize:
        # The used scale is already raw scale.
        results.update({
            "rmse_raw": float(rmse_used),
            "nrmse_percent_raw": float(100 * nrmse_used),
            "preds_raw": preds_used,
            "targets_raw": targets_used,
        })

    return results


def get_eval_rows_from_batches(
    batches,
    batch_indices
):
    """
    Return full time series from selected batches for evaluation.
    No train/eval split is used.
    """

    rows = []

    for batch_id in batch_indices:

        for series_id, ts in enumerate(batches[batch_id]):

            rows.append({
                "batch_id": batch_id,
                "series_id": series_id,
                "global_id": f"b{batch_id}_s{series_id}",
                "segment": np.asarray(ts).ravel(),
            })

    return rows



def evaluate_model_on_eval_rows(
    ML,
    eval_rows,
    rmse_col,
    normalized=True
):
    """
    Evaluate model on full individual time series.

    If normalized=True:
        each raw series is normalized before evaluation,
        because the barycenters/model live in normalized space.

    If normalized=False:
        raw series are evaluated in raw space.
    """

    rows = []

    for row in eval_rows:

        try:
            res = evaluate_one_individual_series(
                ML=ML,
                series_raw=row["segment"],
                sdtw_normalize=normalized,
                return_raw_scale=False
            )

            rows.append({
                "batch_id": row["batch_id"],
                "series_id": row["series_id"],
                "global_id": row["global_id"],
                rmse_col: res["rmse_used"],
            })

        except ValueError as e:
            rows.append({
                "batch_id": row["batch_id"],
                "series_id": row["series_id"],
                "global_id": row["global_id"],
                rmse_col: np.nan,
                f"error_{rmse_col}": str(e),
            })

    return pd.DataFrame(rows)




def evaluate_on_new_data(trained_model,
                          new_data,
                            chromosome):
    new_instance = Instance(new_data, chromosome)
    min = np.min(new_data)
    max = np.max(new_data)
    
    # new_instance.model = trained_model
    new_instance.model = trained_model.to(device)
    new_instance.test_data = (new_data - min) / (max - min)
    _, _, pred, targ = new_instance.validation_phase()
    preds = pred * (max - min) + min
    targs = targ * (max - min) + min
    rmse = np.sqrt(np.mean((preds - targs) ** 2))
    ss_res = np.sum((targs - preds) ** 2)
    ss_tot = np.var(targs) * len(targs)
    r2 = 1 - (ss_res / ss_tot)
    return rmse, r2, new_instance, preds, targs


# ============================================================
# Warm - start
# ============================================================

def build_local_bounds_from_chromosome(
    previous_chromosome,
    neigh_per=0.20,
):
    """
    Build local warm-start bounds around the previous best architecture.

    neigh_per = 0.20 means +/-20% around the previous value.

    The local bounds are clipped to the original global search space.
    """

    if not (0.0 < neigh_per <= 1.0):
        raise ValueError("neigh_per must be in the interval (0, 1].")

    # Original global bounds for:
    # [window, learning_rate, batch_size, hidden_dim, epochs, layer_dim]
    global_xl = np.array([5, 0.0001, 8,   2,  50, 1], dtype=float)
    global_xu = np.array([20, 0.1,   128, 10, 150, 3], dtype=float)

    x_star = np.array([
        previous_chromosome["window"],
        previous_chromosome["learning_rate"],
        previous_chromosome["batch_size"],
        previous_chromosome["hidden_dim"],
        previous_chromosome["epochs"],
        previous_chromosome["layer_dim"],
    ], dtype=float)

    local_xl = x_star * (1.0 - neigh_per)
    local_xu = x_star * (1.0 + neigh_per)

    # Clip to original global bounds
    local_xl = np.maximum(local_xl, global_xl)
    local_xu = np.minimum(local_xu, global_xu)

    # Integer-like variables:
    # window, batch_size, hidden_dim, epochs, layer_dim
    integer_idx = [0, 2, 3, 4, 5]

    local_xl[integer_idx] = np.floor(local_xl[integer_idx])
    local_xu[integer_idx] = np.ceil(local_xu[integer_idx])

    # Final safety clipping
    local_xl = np.maximum(local_xl, global_xl)
    local_xu = np.minimum(local_xu, global_xu)

    # Avoid zero-width intervals
    for idx in range(len(local_xl)):
        if local_xl[idx] == local_xu[idx]:
            if idx in integer_idx:
                local_xl[idx] = max(global_xl[idx], local_xl[idx] - 1)
                local_xu[idx] = min(global_xu[idx], local_xu[idx] + 1)
            else:
                local_xl[idx] = max(global_xl[idx], local_xl[idx] * 0.9)
                local_xu[idx] = min(global_xu[idx], local_xu[idx] * 1.1)

    return local_xl, local_xu


class WarmStartAMLMProblem(ElementwiseProblem):
    def __init__(
        self,
        fixed_barycenters,
        evaluation_data,
        previous_chromosome,
        neigh_per=0.20,
        split=0.70,
        normalize=True,
        # random_state=42,
        D_star_tp=1.8,
    ):
        """
        Warm-start MODE problem after an FDM retraining step.

        Differences from Clustering_centralized
        ---------------------------------------
        1. No k is optimized.
        2. No Soft-DTW k-means is run.
        3. The updated barycenters are fixed.
        4. Only LSTM hyperparameters are tuned locally.
        5. Objectives:
            f1 = validation RMSE on concatenated barycenters
            f2 = residual-instability difference
        """

        self.fixed_barycenters = [
            np.asarray(b).copy()
            for b in fixed_barycenters
        ]

        self.evaluation_data = np.asarray(evaluation_data, dtype=np.float64)

        self.previous_chromosome = previous_chromosome
        self.split = split
        self.normalize = normalize
        # self.random_state = random_state
        self.D_star_tp = D_star_tp

        xl, xu = build_local_bounds_from_chromosome(
            previous_chromosome=previous_chromosome,
            neigh_per=neigh_per,
        )

        super().__init__(
            n_var=6,
            n_obj=2,
            xl=xl,
            xu=xu,
        )

        self.trained_models = {}
        self.evaluation_errors = []

    # ------------------------------------------------------------
    def _repair_architecture(self, x):
        architecture = {
            "window": int(round(x[0])),
            "learning_rate": float(x[1]),
            "batch_size": int(round(x[2])),
            "hidden_dim": int(round(x[3])),
            "epochs": int(round(x[4])),
            "layer_dim": int(round(x[5])),
        }

        architecture["window"] = int(np.clip(architecture["window"], 5, 20))
        architecture["batch_size"] = int(np.clip(architecture["batch_size"], 8, 128))
        architecture["hidden_dim"] = int(np.clip(architecture["hidden_dim"], 2, 10))
        architecture["epochs"] = int(np.clip(architecture["epochs"], 50, 150))
        architecture["layer_dim"] = int(np.clip(architecture["layer_dim"], 1, 3))

        return architecture

    # ------------------------------------------------------------
    def _collect_prediction_errors(self, ML_current):
        all_errors = []

        for ts in self.evaluation_data:

            try:
                res = evaluate_one_individual_series(
                    ML=ML_current,
                    series_raw=ts,
                    sdtw_normalize=self.normalize,
                    return_raw_scale=False,
                )

                errors = res["preds_used"] - res["targets_used"]
                all_errors.append(errors)

            except ValueError:
                continue

        if len(all_errors) == 0:
            return None

        all_errors = np.concatenate(all_errors)
        all_errors = all_errors - np.mean(all_errors)

        return all_errors

    # ------------------------------------------------------------
    def _variance_instability_score(self, errors):
        if errors is None or len(errors) < 30:
            return 1e6

        detector = ChangePointDetector(errors)
        change_points = detector.aL_bs_genCov(self.D_star_tp)

        all_changepoints = [0] + sorted(change_points) + [len(errors)]

        variance_segment_pairs = []

        for i in range(len(all_changepoints) - 1):
            start = all_changepoints[i]
            end = all_changepoints[i + 1]

            segment = errors[start:end]

            if len(segment) <= 1:
                continue

            var = np.var(segment, ddof=1)
            variance_segment_pairs.append((var, segment))

        if len(variance_segment_pairs) == 0:
            return 0.0

        variance_segment_pairs.sort(key=lambda x: x[0])
        sorted_segments = [seg for _, seg in variance_segment_pairs]

        def grow_until_cp(segments, D_star_tp, reverse=False):
            accumulated = []
            iterable = reversed(segments) if reverse else segments

            for seg in iterable:
                accumulated.append(seg)
                combined = np.concatenate(accumulated)

                detector = ChangePointDetector(combined)
                cps = np.asarray(detector.aL_bs_genCov(D_star_tp))

                valid_cps = cps[
                    (cps > 0) &
                    (cps < len(combined) - 1)
                ]

                if len(valid_cps) > 0:
                    return combined, np.min(valid_cps)

            return np.concatenate(accumulated), None

        L, cp_L = grow_until_cp(
            sorted_segments,
            self.D_star_tp,
            reverse=False,
        )

        S, cp_S = grow_until_cp(
            sorted_segments,
            self.D_star_tp,
            reverse=True,
        )

        if cp_L is None or cp_S is None:
            return 0.0

        var_L = np.var(L, ddof=1)
        var_S = np.var(S, ddof=1)

        difference = var_S - var_L

        return float(max(0.0, difference))

    # ------------------------------------------------------------
    def _evaluate(self, x, out, *args, **kwargs):
        architecture = self._repair_architecture(x)

        try:
            ML_current = train_model_from_barycenters(
                barycenters=self.fixed_barycenters,
                architecture=architecture,
                split=self.split,
                normalized=self.normalize,
                # seed=self.random_state,
            )

            val_output = ML_current.validation_phase()
            val_rmse = float(val_output[0])

            errors = self._collect_prediction_errors(ML_current)
            instability_score = self._variance_instability_score(errors)

            out["F"] = [
                val_rmse,
                instability_score,
            ]

            self.trained_models[tuple(np.asarray(x, dtype=float))] = ML_current

        except Exception as e:
            import traceback

            tb = traceback.format_exc()

            print("\n" + "=" * 80)
            print("ERROR INSIDE WarmStartAMLMProblem._evaluate()")
            print("Chromosome x:", x)
            print("Architecture:", architecture)
            print(tb)
            print("=" * 80)

            self.evaluation_errors.append({
                "x": np.asarray(x, dtype=float).copy(),
                "architecture": architecture,
                "error": repr(e),
                "traceback": tb,
            })

            out["F"] = [1e6, 1e6]
            self.trained_models[tuple(np.asarray(x, dtype=float))] = None

def ApplyingMODE_WarmStart(
    fixed_barycenters,
    evaluation_data,
    previous_chromosome,
    num_population,
    num_offsprings,
    num_generations,
    neigh_per=0.20,
    split=0.70,
    normalize=True,
    # random_state=42,
    D_star_tp=1.8,
):
    """
    Warm-start MODE tuning after FDM retraining.

    Uses:
        - fixed updated barycenters
        - local neighborhood around previous best architecture
        - reduced population / offspring / generations
    """

    Problem = WarmStartAMLMProblem(
        fixed_barycenters=fixed_barycenters,
        evaluation_data=evaluation_data,
        previous_chromosome=previous_chromosome,
        neigh_per=neigh_per,
        split=split,
        normalize=normalize,
        # random_state=random_state,
        D_star_tp=D_star_tp,
    )

    Algorithm = NSGA2(
        pop_size=num_population,
        n_offsprings=num_offsprings,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    Termination_criterion = get_termination(
        "n_gen",
        num_generations,
    )

    start = time.time()

    results_warm = pymoo_minimize(
        Problem,
        Algorithm,
        Termination_criterion,
        # seed=random_state,
        save_history=True,
        verbose=True,
    )

    duration = time.time() - start

    return results_warm, Problem, duration

def ApplyingASF_WarmStart(
    results_of_tuning,
    Problem_class,
    previous_chromosome,
    weights=np.array([0.6, 0.4]),
    return_full=True,
):
    """
    Select best warm-start solution using ASF.

    Returns:
        best_instance
        best_model
        best_chromosome
        best_info
    """

    valid_models = {
        k: v for k, v in Problem_class.trained_models.items()
        if v is not None
    }

    if len(valid_models) == 0:
        msg = (
            "No valid trained models were found in warm-start tuning. "
            "Check Problem_class.evaluation_errors."
        )

        if hasattr(Problem_class, "evaluation_errors") and len(Problem_class.evaluation_errors) > 0:
            msg += "\n\nFirst warm-start error:\n"
            msg += Problem_class.evaluation_errors[0]["traceback"]

        raise ValueError(msg)

    F = np.asarray(results_of_tuning.F, dtype=np.float64)
    X = np.asarray(results_of_tuning.X, dtype=np.float64)

    if F.ndim == 1:
        F = F.reshape(1, -1)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    denom = np.maximum(
        approx_nadir - approx_ideal,
        1e-12,
    )

    nF = (F - approx_ideal) / denom

    weights = np.asarray(weights, dtype=np.float64)

    if len(weights) != F.shape[1]:
        raise ValueError(
            f"weights has length {len(weights)}, "
            f"but warm-start problem has {F.shape[1]} objectives."
        )

    decomp = ASF()

    best_idx = decomp.do(
        nF,
        1.0 / weights,
    ).argmin()

    best_x = X[best_idx]
    best_F = F[best_idx]

    best_chromosome_6d = Problem_class._repair_architecture(best_x)

    # Preserve previous K because warm-start does not tune k
    best_chromosome = dict(best_chromosome_6d)
    best_chromosome["num_clusters"] = int(previous_chromosome["num_clusters"])

    key = tuple(np.asarray(best_x, dtype=float))

    best_instance = Problem_class.trained_models.get(key, None)

    if best_instance is None:
        available_keys = list(Problem_class.trained_models.keys())
        key_array = np.asarray(available_keys, dtype=np.float64)

        distances = np.linalg.norm(
            key_array - best_x.reshape(1, -1),
            axis=1,
        )

        nearest_idx = int(np.argmin(distances))
        nearest_key = available_keys[nearest_idx]

        best_instance = Problem_class.trained_models[nearest_key]

    if best_instance is None:
        raise ValueError(
            "The selected warm-start ASF solution has no trained model instance."
        )

    best_model = best_instance.model

    best_info = {
        "best_index": int(best_idx),
        "best_x": best_x,
        "best_F": best_F,
        "best_normalized_F": nF[best_idx],
        "weights": weights,
        "best_instance": best_instance,
        "best_model": best_model,
        "best_chromosome": best_chromosome,
        "fixed_barycenters": Problem_class.fixed_barycenters,
    }

    print("\nWarm-start best architecture:")
    print(best_chromosome)

    print("\nWarm-start best objective values:")
    print(best_F)

    if return_full:
        return best_instance, best_model, best_chromosome, best_info

    return best_instance, best_model, best_chromosome

def run_with_energy_tracking(func, label="task", enable=True):
    """
    Run a function while tracking duration, energy consumption and emissions
    with CodeCarbon.

    Returns
    -------
    result : object
        Output of func()

    energy_info : dict
        duration_sec, energy_kwh, energy_wh, emissions_kg
    """

    import time
    import numpy as np

    tracker = None
    start = time.time()

    energy_info = {
        "label": label,
        "duration_sec": np.nan,
        "energy_kwh": np.nan,
        "energy_wh": np.nan,
        "emissions_kg": np.nan,
        "tracker_available": False,
    }

    try:
        if enable:
            from codecarbon import EmissionsTracker

            tracker = EmissionsTracker(
                save_to_file=False,
                log_level="error",
            )

            tracker.start()
            energy_info["tracker_available"] = True

        result = func()

    finally:
        duration = time.time() - start
        energy_info["duration_sec"] = duration

        if tracker is not None:
            try:
                emissions = tracker.stop()
                data = tracker.final_emissions_data

                if data is not None:
                    energy_kwh = getattr(data, "energy_consumed", np.nan)
                    emissions_kg = getattr(data, "emissions", emissions)

                    energy_info["energy_kwh"] = float(energy_kwh)
                    energy_info["energy_wh"] = float(energy_kwh) * 1000.0
                    energy_info["emissions_kg"] = float(emissions_kg)

            except Exception as e:
                energy_info["tracker_error"] = repr(e)

    return result, energy_info

# ============================================================
# Change-point
# ============================================================



class ChangePointDetector:
    ''' This is the Change point mechanism on 
    Variance which is used in the stability component of MODE'''
    def __init__(self, Yt):
        self.Yt = np.array(Yt).flatten()
        self.cp = []
        self.num_of_cp = 0
        self.M_V = []
        self.LL = []

    
    def aL_bs_genCov(self, D_star):
        """Main function (equivalent to aL_bs_genMean_Monroe)"""
        self.cp = []
        self.num_of_cp = 0
        self.M_V = []

        self._aL_bs_Cov(0, len(self.Yt) - 1, D_star)

        if self.num_of_cp == 0:
            return [0]

        self._aL_Elimi2017(D_star)
        self.cp = sorted(self.cp)

        # Remove changepoints too close to start or end
        self.cp = [c for c in self.cp if c >= 20 and c <= len(self.Yt) - 20]

        # Remove changepoints that are too close together
        if len(self.cp) > 2:
            # cleaned_cp = []
            for i in range(1, len(self.cp)):
                if self.cp[i] - self.cp[i - 1] < 20:
                    self.cp[i - 1] = 0  # Flag for removal
            self.cp = sorted(set(filter(lambda x: x > 0, self.cp)))

        return self.cp

    def _aL_bs_Cov(self, st, en, D_star):
        if abs(en - st) < 20:
            return
        
        M_value, loc = self._aL_Cal_max2017(st, en)
        if M_value >= D_star:
            self._aL_bs_Cov(st, loc, D_star)
            self._aL_bs_Cov(loc + 1, en, D_star)
            self.num_of_cp += 1
            self.cp.append(loc)
            self.M_V.append(M_value)
            

    def _aL_Cal_max2017(self, st, en):

        if abs(en - st) > 15:
            extr_a = self.Yt[st:en + 1]
            l = len(extr_a)
            H = int(np.floor(np.log10(l)))

            Et = extr_a - np.mean(extr_a)
            Vech = np.array([self._vech(Et[i]) for i in range(l)]).T

            sk = np.cumsum(Vech, axis=1)
            SK = (1 / np.sqrt(l)) * (sk - (np.arange(1, l + 1) / l) * sk[:, -1, np.newaxis])

            SS = self._covnw(Vech.T, H, demean=True)
            SS_inv = np.linalg.pinv(SS)

            L = np.array([(SK[:, i].T @ SS_inv @ SK[:, i]) for i in range(l)])
            M_value = np.max(np.abs(L))
            loc = st + np.argmax(np.abs(L))
            self.LL = L
            # M_value = np.max(L)
            # index = np.argmax(L)
            # loc = st + index
        else:
            loc = 0
            M_value = 0
        return M_value, loc

    def _aL_Elimi2017(self, D_star):
        if self.num_of_cp != len(self.cp):
            self.cp = [0]
            return

        tmp_cp = [0] + self.cp + [len(self.Yt)]
        tmp_cp = sorted(set(tmp_cp))

        if tmp_cp[-1] < tmp_cp[-2]:
            tmp_cp[-2] = 0
            tmp_cp = sorted(set(tmp_cp))
            self.num_of_cp -= 1

        flag = True
        Cal_times = 0

        while flag:
            tmp_cp2 = []
            Cal_times += 1

            for i in range(1, self.num_of_cp + 1):
                M_value, loc = self._aL_Cal_max2017(tmp_cp[i - 1], tmp_cp[i + 1])
                if M_value >= D_star:
                    tmp_cp2.append(loc)

            tmp_cp2 = sorted(set(tmp_cp2))
            num_of_cp2 = len(tmp_cp2)

            if self.num_of_cp == num_of_cp2:
                n = sum(abs(tmp_cp[i] - tmp_cp2[i - 1]) <= 30 for i in range(1, self.num_of_cp + 1))
                if n == self.num_of_cp:
                    flag = False

            if Cal_times >= 7:
                flag = False

            if flag:
                self.num_of_cp = num_of_cp2
                tmp_cp = [0] + tmp_cp2 + [len(self.Yt)]

        self.cp = tmp_cp2


    def _covnw(self, data, nlag=None, demean=True):
        data = np.array(data)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        T, _ = data.shape

        if nlag is None:
            nlag = min(int(np.floor(1.2 * T ** (1 / 3))), T)

        if demean:
            data = data - np.mean(data, axis=0)

        V = data.T @ data / T
        w = (nlag + 1 - np.arange(nlag + 1)) / (nlag + 1)

        for i in range(1, nlag + 1):
            Gammai = data[i:T, :].T @ data[0:T - i, :] / T
            V += w[i] * (Gammai + Gammai.T)

        return V

    def _vech(self, scalar):
        """In univariate case, vech is just the squared deviation."""
        return np.array([scalar ** 2])
    


# ============================================================
# AMLM
# ============================================================


# ============================================================
# Cluster-gated FDM update utilities
# ============================================================
def evaluate_incoming_batch_failures(
    model_instance,
    new_batch,
    normalize=True,
    D_star_tp=1.8,
    failure_threshold_percent=5.0,
    count_errors_as_failures=True,
):
    """
    Evaluate the current model on an incoming batch and decide whether
    the batch requires retraining.

    Logic
    -----
    Each time series in the incoming batch is evaluated independently.
    A time series is marked as "failure" if the residual sequence contains
    at least one detected change point.

    The global failure ratio is:

        failure_ratio_percent =
            100 * number_of_failed_series / total_number_of_series_in_batch

    No cluster-level ratios are computed here.

    Parameters
    ----------
    model_instance : Instance
        Trained LSTM wrapper returned by ApplyingASF / train_model_from_barycenters.

    new_batch : np.ndarray
        Incoming batch, shape (n_series, T).

    normalize : bool
        If True, each incoming series is normalized before evaluation,
        consistent with normalized Soft-DTW barycenters.

    D_star_tp : float
        Threshold used by the change-point detector.

    failure_threshold_percent : float
        If global failure ratio is greater than or equal to this value,
        retraining is required.

    count_errors_as_failures : bool
        If True, evaluation errors are counted as failures.

    Returns
    -------
    eval_df : pd.DataFrame
        One row per time series.

    failed_indices : np.ndarray
        Indices of failed time series.

    summary : dict
        Global batch-level failure statistics and retraining decision.
    """

    new_batch = np.asarray(new_batch, dtype=np.float64)

    rows = []

    for series_id, ts in enumerate(new_batch):

        try:
            res = evaluate_one_individual_series(
                ML=model_instance,
                series_raw=ts,
                sdtw_normalize=normalize,
                return_raw_scale=False,
            )

            preds = res["preds_used"]
            targets = res["targets_used"]

            error = preds - targets
            error = error - np.mean(error)

            detector = ChangePointDetector(error)
            change_points = detector.aL_bs_genCov(D_star_tp)

            n_change_points = int(
                np.sum(np.asarray(change_points) > 0)
            )

            failed = n_change_points > 0
            status = "failure" if failed else "OK"

            rows.append({
                "series_id": int(series_id),
                "rmse": float(res["rmse_used"]),
                "status": status,
                "failed": bool(failed),
                "n_change_points": n_change_points,
            })

        except ValueError as e:

            failed = bool(count_errors_as_failures)

            rows.append({
                "series_id": int(series_id),
                "rmse": np.nan,
                "status": "error",
                "failed": failed,
                "n_change_points": np.nan,
                "error": str(e),
            })

    eval_df = pd.DataFrame(rows)

    total_series = len(eval_df)
    n_failed = int(eval_df["failed"].sum())

    failure_ratio = n_failed / total_series if total_series > 0 else 1.0
    failure_ratio_percent = 100.0 * failure_ratio

    retraining_required = (
        failure_ratio_percent >= failure_threshold_percent
    )

    failed_indices = (
        eval_df.loc[eval_df["failed"], "series_id"]
        .astype(int)
        .to_numpy()
    )

    summary = {
        "total_series": int(total_series),
        "n_failed": int(n_failed),
        "n_ok": int(np.sum(eval_df["status"] == "OK")),
        "n_error": int(np.sum(eval_df["status"] == "error")),
        "failure_ratio": float(failure_ratio),
        "failure_ratio_percent": float(failure_ratio_percent),
        "failure_threshold_percent": float(failure_threshold_percent),
        "retraining_required": bool(retraining_required),
    }

    return eval_df, failed_indices, summary


def initialize_fdm_memory_state(
    best_info,
    best_chromosome,
):
    """
    Initialize the barycenter/FDM memory state after MODE + ASF.

    This stores:
        - current barycenters
        - initial cluster sizes
        - historical members per cluster

    These objects are needed for future FDM barycenter updates.
    """

    K = int(best_chromosome["num_clusters"])

    current_barycenters = [
        np.asarray(b).copy()
        for b in best_info["barycenters"]
    ]

    labels = np.asarray(best_info["labels"], dtype=int)
    X_used = np.asarray(best_info["X_used"], dtype=np.float64)

    cluster_sizes = get_cluster_sizes(
        labels=labels,
        k=K,
    )

    cluster_members_by_cluster = {
        cluster_id: get_cluster_members(
            X_used=X_used,
            labels=labels,
            cluster_id=cluster_id,
        )
        for cluster_id in range(K)
    }

    state = {
        "K": K,
        "current_barycenters": current_barycenters,
        "cluster_sizes": cluster_sizes,
        "cluster_members_by_cluster": cluster_members_by_cluster,
    }

    return state


def assign_failed_series_to_barycenters(
    new_batch,
    failed_indices,
    current_barycenters,
    gamma=1.0,
    normalize=True,
    use_divergence=True,
):
    """
    Assign only the failed incoming time series to the closest barycenter.

    OK time series are ignored and do not affect the barycenter update.

    Parameters
    ----------
    new_batch : np.ndarray
        Incoming batch, shape (n_series, T).

    failed_indices : array-like
        Indices of failed time series from evaluate_incoming_batch_failures().

    current_barycenters : list[np.ndarray]
        Current barycenters.

    Returns
    -------
    assignment_df : pd.DataFrame
        One row per failed time series.

    failed_members_by_cluster : dict
        Prepared failed series grouped by assigned cluster.

    affected_clusters : list[int]
        Clusters that received at least one failed series.
    """

    new_batch = np.asarray(new_batch, dtype=np.float64)
    failed_indices = np.asarray(failed_indices, dtype=int)

    K = len(current_barycenters)

    failed_members_by_cluster = {
        cluster_id: []
        for cluster_id in range(K)
    }

    assignment_rows = []

    for series_id in failed_indices:

        ts_used = prepare_single(
            new_batch[series_id],
            normalize=normalize,
        )

        best_cluster, distances = assign_to_best_cluster(
            ts=ts_used,
            barycenters=current_barycenters,
            gamma=gamma,
            use_divergence=use_divergence,
        )

        failed_members_by_cluster[best_cluster].append(ts_used)

        assignment_rows.append({
            "series_id": int(series_id),
            "assigned_cluster": int(best_cluster),
            "min_distance": float(np.min(distances)),
        })

        for cluster_id, dist in enumerate(distances):
            assignment_rows[-1][f"distance_to_cluster_{cluster_id}"] = float(dist)

    assignment_df = pd.DataFrame(assignment_rows)

    affected_clusters = [
        cluster_id
        for cluster_id, members in failed_members_by_cluster.items()
        if len(members) > 0
    ]

    return assignment_df, failed_members_by_cluster, affected_clusters


def update_barycenters_from_failed_assignments_fdm(
    current_barycenters,
    cluster_sizes,
    cluster_members_by_cluster,
    failed_members_by_cluster,
    gamma=1.0,
    normalize=True,
    max_iter=50,
    energy_keep=0.99,
    max_components=5,
    exact_diagnostic=False,
):
    """
    Update only the barycenters that received failed time series.

    Clusters with no assigned failed series remain untouched.
    """

    bar_current = [
        np.asarray(b).copy()
        for b in current_barycenters
    ]

    cluster_sizes = np.asarray(cluster_sizes, dtype=np.float64).copy()

    cluster_members_by_cluster = {
        int(k): list(v)
        for k, v in cluster_members_by_cluster.items()
    }

    K = len(bar_current)

    update_rows = []
    fdm_updated_by_cluster = {}
    exact_updated_by_cluster = {}
    metrics_by_cluster = {}

    for cluster_id in range(K):

        new_members = failed_members_by_cluster.get(cluster_id, [])

        if len(new_members) == 0:
            update_rows.append({
                "cluster_id": int(cluster_id),
                "updated": False,
                "old_cluster_size": int(cluster_sizes[cluster_id]),
                "n_failed_assigned": 0,
                "new_cluster_size": int(cluster_sizes[cluster_id]),
                "status": "untouched",
            })
            continue

        old_cluster_size = int(cluster_sizes[cluster_id])

        updated_barycenter, fdm_info = Update_barycenter_fdm(
            bar=bar_current,
            cluster_sizes=cluster_sizes,
            best_cluster=cluster_id,
            new_series_list=new_members,
            gamma=gamma,
            max_iter=max_iter,
            energy_keep=energy_keep,
            max_components=max_components,
            clip_range=(0.0, 1.0) if normalize else None,
            return_info=True,
        )

        bar_current[cluster_id] = updated_barycenter.copy()

        row = {
            "cluster_id": int(cluster_id),
            "updated": True,
            "old_cluster_size": old_cluster_size,
            "n_failed_assigned": int(len(new_members)),
            "new_cluster_size": int(cluster_sizes[cluster_id]),
            "status": "updated",
        }

        if exact_diagnostic:

            old_members = cluster_members_by_cluster[cluster_id]

            exact_barycenter = sdtw_barycenter(
                X=old_members + new_members,
                barycenter_init=fdm_info["old_barycenter"].copy(),
                gamma=gamma,
                weights=np.ones(len(old_members) + len(new_members)),
                max_iter=max_iter,
            )

            update_metrics = barycenter_error_metrics(
                approx=updated_barycenter,
                reference=exact_barycenter,
                old_reference=fdm_info["old_barycenter"],
                gamma=gamma,
            )

            fdm_updated_by_cluster[cluster_id] = updated_barycenter.copy()
            exact_updated_by_cluster[cluster_id] = exact_barycenter.copy()
            metrics_by_cluster[cluster_id] = update_metrics

            row.update({
                "fdm_vs_exact_rmse": update_metrics["rmse"],
                "fdm_vs_exact_nrmse_percent": update_metrics["nrmse_range_percent"],
                "relative_update_error_percent": update_metrics.get(
                    "relative_update_error_percent",
                    np.nan,
                ),
            })

        cluster_members_by_cluster[cluster_id].extend(new_members)

        update_rows.append(row)

    update_df = pd.DataFrame(update_rows)

    updated_concat = concatenate_barycenters(bar_current)

    return {
        "bar_current": bar_current,
        "updated_concat": updated_concat,
        "cluster_sizes": cluster_sizes,
        "cluster_members_by_cluster": cluster_members_by_cluster,
        "update_df": update_df,
        "fdm_updated_by_cluster": fdm_updated_by_cluster,
        "exact_updated_by_cluster": exact_updated_by_cluster,
        "metrics_by_cluster": metrics_by_cluster,
    }

def Incremental_Barycenter_Update(
    H_new,
    drifted_indices,
    current_barycenters,
    cluster_sizes,
    gamma=1.0,
    normalize=True,
    use_divergence=False,
    max_iter=50,
    energy_keep=0.99,
    max_components=5,
):
    """
    Assign drifted time series to their closest existing barycenter and
    update only the clusters receiving new members.

    Historical time series are not required. The old cluster is represented
    through its current barycenter and cluster size.

    Parameters
    ----------
    H_new : np.ndarray
        Current horizon-set, shape (N, T).

    drifted_indices : array-like
        Indices of time series flagged by the change-point detector.

    current_barycenters : list[np.ndarray]
        Currently accepted cluster barycenters.

    cluster_sizes : array-like
        Number of historical members represented by every barycenter.

    gamma : float
        Soft-DTW smoothing parameter.

    normalize : bool
        Whether incoming time series should be independently normalized.

    use_divergence : bool
        False uses raw Soft-DTW distance.
        True uses Soft-DTW divergence.

    Returns
    -------
    result : dict
        Updated barycenters, cluster sizes, assignments, affected clusters,
        and the new concatenated time series.
    """

    H_new = np.asarray(H_new, dtype=np.float64)
    drifted_indices = np.asarray(drifted_indices, dtype=int)

    barycenters = [
        np.asarray(barycenter, dtype=np.float64).copy()
        for barycenter in current_barycenters
    ]

    cluster_sizes = np.asarray(
        cluster_sizes,
        dtype=np.float64,
    ).copy()

    number_of_clusters = len(barycenters)

    if len(cluster_sizes) != number_of_clusters:
        raise ValueError(
            "cluster_sizes must have one value per barycenter."
        )

    # Store the newly assigned members for each cluster.
    new_members_by_cluster = {
        cluster_id: []
        for cluster_id in range(number_of_clusters)
    }

    assignment_rows = []

    # ============================================================
    # 1. Assign every drifted series to its closest barycenter
    # ============================================================

    for series_id in drifted_indices:

        if series_id < 0 or series_id >= len(H_new):
            raise IndexError(
                f"Invalid drifted series index: {series_id}"
            )

        prepared_series = prepare_single(
            H_new[series_id],
            normalize=normalize,
        )

        best_cluster, distances = assign_to_best_cluster(
            ts=prepared_series,
            barycenters=barycenters,
            gamma=gamma,
            use_divergence=use_divergence,
        )

        new_members_by_cluster[best_cluster].append(
            prepared_series
        )

        assignment_rows.append({
            "series_id": int(series_id),
            "assigned_cluster": int(best_cluster),
            "soft_dtw_distance": float(
                distances[best_cluster]
            ),
        })

    # ============================================================
    # 2. Update only clusters that received new members
    # ============================================================

    affected_clusters = []

    for cluster_id in range(number_of_clusters):

        new_members = new_members_by_cluster[cluster_id]

        if len(new_members) == 0:
            # This barycenter remains exactly as it was.
            continue

        affected_clusters.append(cluster_id)

        updated_barycenter, _ = Update_barycenter_fdm(
            bar=barycenters,
            cluster_sizes=cluster_sizes,
            best_cluster=cluster_id,
            new_series_list=new_members,
            gamma=gamma,
            max_iter=max_iter,
            energy_keep=energy_keep,
            max_components=max_components,
            clip_range=(0.0, 1.0) if normalize else None,
            return_info=True,
        )

        barycenters[cluster_id] = (
            updated_barycenter.copy()
        )

    # ============================================================
    # 3. Construct the new reduced training representation
    # ============================================================

    concatenated_series = concatenate_barycenters(
        barycenters
    )

    assignment_df = pd.DataFrame(assignment_rows)

    return {
        "barycenters": barycenters,
        "cluster_sizes": cluster_sizes,
        "concatenated_series": concatenated_series,
        "assignment_df": assignment_df,
        "affected_clusters": affected_clusters,
        "new_members_by_cluster": new_members_by_cluster,
    }
# ============================================================
# Catastrophic forgetting / assimilation metrics
# ============================================================

def evaluate_model_rmse_on_batches(
    model_instance,
    List_of_batches,
    batch_indices,
    normalize=True,
    rmse_col="rmse",
):
    """
    Evaluate a model on selected full batches and return one RMSE row
    per time series.

    Parameters
    ----------
    model_instance : Instance
        Trained LSTM wrapper.

    List_of_batches : list[np.ndarray]
        Full list of time-series batches.

    batch_indices : list[int]
        Which batches to evaluate.

    normalize : bool
        If True, each series is normalized before evaluation.

    rmse_col : str
        Name of the RMSE column to create.

    Returns
    -------
    df : pd.DataFrame
        Columns:
            batch_id, series_id, global_id, rmse_col
    """

    rows = []

    for batch_id in batch_indices:

        batch = List_of_batches[batch_id]

        for series_id, ts in enumerate(batch):

            try:
                res = evaluate_one_individual_series(
                    ML=model_instance,
                    series_raw=ts,
                    sdtw_normalize=normalize,
                    return_raw_scale=False,
                )

                rows.append({
                    "batch_id": int(batch_id),
                    "series_id": int(series_id),
                    "global_id": f"b{batch_id}_s{series_id}",
                    rmse_col: float(res["rmse_used"]),
                })

            except ValueError as e:
                rows.append({
                    "batch_id": int(batch_id),
                    "series_id": int(series_id),
                    "global_id": f"b{batch_id}_s{series_id}",
                    rmse_col: np.nan,
                    f"error_{rmse_col}": str(e),
                })

    return pd.DataFrame(rows)

def compute_catastrophic_forgetting_metrics(
    old_before_df,
    old_after_df,
    new_before_df,
    new_after_df,
    old_reference_df,
    step_id,
    batch_id,
    method_name="AMLM-FDM",
    eps=1e-8,
):
    """
    Compute RMSE-based catastrophic forgetting and assimilation metrics.

    Definitions
    -----------
    F_RMSE:
        RMSE_old_after - RMSE_old_before.
        Positive means forgetting.

    A_RMSE:
        RMSE_new_before - RMSE_new_after.
        Positive means assimilation.

    omega_base:
        M0 on batch 0 vs Mi on batch 0.

    omega_previous:
        M_{i-1} on batch i-1 vs Mi on batch i-1.

    omega_memory_bj:
        M_j on batch j vs Mi on batch j.

    omega_memory_avg:
        Overall cumulative memory retention.

    omega_new:
        M_{i-1} on batch i vs Mi on batch i.

    omega_balance:
        Harmonic mean of omega_memory_avg and omega_new.
    """

    # ------------------------------------------------------------
    # Old batches: before vs after
    # ------------------------------------------------------------

    old_df = old_before_df.merge(
        old_after_df,
        on=["batch_id", "series_id", "global_id"],
        how="inner",
    )

    old_df["forgetting_rmse"] = (
        old_df["rmse_old_after"] - old_df["rmse_old_before"]
    )

    old_df["relative_forgetting"] = (
        old_df["forgetting_rmse"] /
        (old_df["rmse_old_before"] + eps)
    )

    old_valid = old_df.dropna(
        subset=["rmse_old_before", "rmse_old_after"]
    )

    # ------------------------------------------------------------
    # New batch: before vs after
    # ------------------------------------------------------------

    new_df = new_before_df.merge(
        new_after_df,
        on=["batch_id", "series_id", "global_id"],
        how="inner",
    )

    new_df["assimilation_rmse"] = (
        new_df["rmse_new_before"] - new_df["rmse_new_after"]
    )

    new_df["relative_assimilation"] = (
        new_df["assimilation_rmse"] /
        (new_df["rmse_new_before"] + eps)
    )

    new_valid = new_df.dropna(
        subset=["rmse_new_before", "rmse_new_after"]
    )

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------

    def score_from_rmse(rmse):
        return 1.0 / (1.0 + np.asarray(rmse, dtype=np.float64))

    def omega_from_before_after(before, after):
        before = np.asarray(before, dtype=np.float64)
        after = np.asarray(after, dtype=np.float64)

        if len(before) == 0 or len(after) == 0:
            return np.nan

        s_before = score_from_rmse(before)
        s_after = score_from_rmse(after)

        return float(np.mean(s_after / (s_before + eps)))

    # ------------------------------------------------------------
    # omega_previous: M_{i-1} on B_{i-1} vs Mi on B_{i-1}
    # ------------------------------------------------------------

    previous_subset = old_valid[
        old_valid["batch_id"] == (batch_id - 1)
    ]

    if len(previous_subset) > 0:
        omega_previous = omega_from_before_after(
            before=previous_subset["rmse_old_before"],
            after=previous_subset["rmse_old_after"],
        )
    else:
        omega_previous = np.nan

    # ------------------------------------------------------------
    # omega_new: M_{i-1} on Bi vs Mi on Bi
    # ------------------------------------------------------------

    if len(new_valid) > 0:
        omega_new = omega_from_before_after(
            before=new_valid["rmse_new_before"],
            after=new_valid["rmse_new_after"],
        )
    else:
        omega_new = np.nan

    # ------------------------------------------------------------
    # Cumulative memory: M_j on Bj vs Mi on Bj
    # ------------------------------------------------------------

    omega_base = np.nan
    omega_memory_avg = np.nan
    omega_memory_by_batch = {}

    if old_reference_df is not None and len(old_reference_df) > 0:

        ref_compare_df = old_reference_df.merge(
            old_after_df[
                ["batch_id", "series_id", "global_id", "rmse_old_after"]
            ],
            on=["batch_id", "series_id", "global_id"],
            how="inner",
        ).dropna(
            subset=["rmse_reference", "rmse_old_after"]
        )

        for past_batch_id in sorted(ref_compare_df["batch_id"].unique()):

            batch_subset = ref_compare_df[
                ref_compare_df["batch_id"] == past_batch_id
            ]

            omega_value = omega_from_before_after(
                before=batch_subset["rmse_reference"],
                after=batch_subset["rmse_old_after"],
            )

            omega_memory_by_batch[
                f"omega_memory_b{int(past_batch_id)}"
            ] = omega_value

        if len(ref_compare_df) > 0:
            omega_memory_avg = omega_from_before_after(
                before=ref_compare_df["rmse_reference"],
                after=ref_compare_df["rmse_old_after"],
            )

        omega_base = omega_memory_by_batch.get("omega_memory_b0", np.nan)

    # ------------------------------------------------------------
    # omega_balance
    # ------------------------------------------------------------

    if np.isnan(omega_memory_avg) or np.isnan(omega_new):
        omega_balance = np.nan
    else:
        omega_balance = float(
            2.0 * omega_memory_avg * omega_new /
            (omega_memory_avg + omega_new + eps)
        )

    # ------------------------------------------------------------
    # Summary row
    # ------------------------------------------------------------

    summary_row = {
        "step": int(step_id),
        "new_batch_id": int(batch_id),
        "method": method_name,

        "n_old_eval_series": int(len(old_valid)),
        "n_new_eval_series": int(len(new_valid)),

        "old_rmse_before_mean": old_valid["rmse_old_before"].mean(),
        "old_rmse_after_mean": old_valid["rmse_old_after"].mean(),
        "F_RMSE": old_valid["forgetting_rmse"].mean(),
        "RF_avg": old_valid["relative_forgetting"].mean(),

        "new_rmse_before_mean": new_valid["rmse_new_before"].mean(),
        "new_rmse_after_mean": new_valid["rmse_new_after"].mean(),
        "A_RMSE": new_valid["assimilation_rmse"].mean(),
        "RA_avg": new_valid["relative_assimilation"].mean(),

        "omega_base": omega_base,
        "omega_previous": omega_previous,
        "omega_memory_avg": omega_memory_avg,
        "omega_new": omega_new,
        "omega_balance": omega_balance,
    }

    summary_row.update(omega_memory_by_batch)

    return summary_row, old_df, new_df

def make_reference_from_new_after(new_after_df):
    """
    After batch i has been accepted/learned, store Mi on Bi as the
    reference performance for future omega_memory_bi calculations.
    """

    return new_after_df[
        ["batch_id", "series_id", "global_id", "rmse_new_after"]
    ].rename(
        columns={"rmse_new_after": "rmse_reference"}
    )



def compute_squared_residuals_from_batch(
    model,
    batch,
    normalize=True,
    return_raw_scale=False
):
    """
    Evaluate a trained AMLM/LSTM Instance on each time series of a batch
    and compute squared residuals.

    Parameters
    ----------
    model_instance : ar.Instance
        Trained model instance, e.g. best_instance or current_instance.

    batch : np.ndarray
        Batch of time series with shape (N, T).

    normalize : bool
        If True, each time series is independently min-max normalized before
        prediction, consistent with normalized Soft-DTW barycenters.

    return_raw_scale : bool
        If False, residuals are computed in the normalized/model-used scale.
        This is recommended if your model was trained with normalize=True.

    Returns
    -------
    results : list[dict]
        One dictionary per time series.
    """

    results = []

    for series_id, ts in enumerate(batch):

        res = evaluate_one_individual_series(
            ML=model,
            series_raw=ts,
            sdtw_normalize=normalize,
            return_raw_scale=return_raw_scale
        )

        if return_raw_scale:
            preds = res["preds_raw"]
            targets = res["targets_raw"]
        else:
            preds = res["preds_used"]
            targets = res["targets_used"]

        residuals = preds - targets
        squared_residuals = residuals**2

        results.append({
            "series_id": series_id,
            "preds": preds,
            "targets": targets,
            "residuals": residuals,
            "squared_residuals": squared_residuals,
            "prediction_time_index": res["prediction_time_index"],
            "rmse_used": res["rmse_used"],
            })
        # print(f"{res["rmse_used"]}\n")
    return results



def Criterion(H_ref, H_new, model):
    H_ref_results = compute_squared_residuals_from_batch(model=model,
        batch=H_ref,normalize=True,return_raw_scale=False)
    H_new_results = compute_squared_residuals_from_batch(model=model,
        batch=H_new,normalize=True,return_raw_scale=False)
    Length = len(H_ref)
    Ratio = 0
    for series_id in range(len(H_ref_results)):
        sq_res_1 = np.asarray(
            H_ref_results[series_id]["squared_residuals"],
            dtype=float
        ).reshape(-1)

        sq_res_2 = np.asarray(
            H_new_results[series_id]["squared_residuals"],
            dtype=float
        ).reshape(-1)

        concatenated_signal = np.concatenate([sq_res_1, sq_res_2])
        boundary_index = len(sq_res_1)
        detector = ChangePointDetector(concatenated_signal)
        cps = detector.detect_mean_change(2.4)
        # Remove fake/no-change output and sort
        cps = sorted([int(cp) for cp in cps if cp > 0])
        if len(cps) == 0:
            continue
        last_cp = cps[-1]
        # CP must be after the reference/new boundary
        if last_cp < boundary_index:
            continue
        before_cp = concatenated_signal[:last_cp]
        after_cp = concatenated_signal[last_cp:]
        if len(before_cp) == 0 or len(after_cp) == 0:
            continue
        rmse_before_cp = np.sqrt(np.mean(before_cp))
        rmse_after_cp = np.sqrt(np.mean(after_cp))
        # Drift only if RMSE gets worse after the last CP
        if rmse_after_cp > rmse_before_cp*1.05:
            Ratio += 1
    return Ratio/Length

    # return {
    #     "threshold": retune_thresh,
    #     "RMSEs": list_of_RMSEs,
    #     "ratios": list_of_ratios,
    #     "retuned_batches": list_of_retuned_batches,
    #     "durations_retune_only": list_of_durations,
    #     "energies_retune_only": list_of_energies,
    #     "batch_durations": list_of_batch_durations,
    #     "batch_energies": list_of_batch_energies,
    #     "rmse_rows": all_rmse_rows,
    # }
from pymoo.indicators.hv import HV

def _hv_indicator(ref_point):
    return HV(ref_point=np.asarray(ref_point, dtype=float))
  


def _get_F_from_generation(hist, use_opt=True):
    """
    Extract objective values from one pymoo generation.
    use_opt=True uses the current non-dominated front.
    use_opt=False uses the whole population.
    """
    F = hist.opt.get("F") if use_opt else hist.pop.get("F")
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    F = F[np.all(np.isfinite(F), axis=1)]
    return F


def _estimate_ideal_nadir_from_history(results):
    """
    Estimate ideal and nadir points from the whole optimization history.
    This is used to normalize objectives to [0, 1].
    """
    all_F = []
    for hist in results.history:
        F = _get_F_from_generation(hist, use_opt=False)
        if len(F) > 0:
            all_F.append(F)
    if len(all_F) == 0:
        return None, None
    all_F = np.vstack(all_F)
    ideal = np.min(all_F, axis=0)
    nadir = np.max(all_F, axis=0)
    return ideal, nadir


def _normalize_minimization_objectives(F, ideal, nadir):
    """
    Normalize minimized objectives to [0, 1].
    0 = best observed value
    1 = worst observed value
    """
    eps = 1e-12
    denom = np.where(np.abs(nadir - ideal) < eps, 1.0, nadir - ideal)
    F_norm = (F - ideal) / denom
    F_norm = np.clip(F_norm, 0.0, 1.0)
    return F_norm


def collect_relative_hypervolume_history(results, ideal=None, nadir=None, use_opt=True):
    """
    Compute relative hypervolume values in [0, 1].

    Since the objectives are normalized to [0, 1] and the reference point is
    [1, 1, 1], the maximum possible hypervolume is 1.

    Parameters
    ----------
    results : pymoo result
        Result object with save_history=True.

    ideal, nadir : np.ndarray or None
        If None, they are estimated from the optimization history.
        For fair comparison across runs, pass the same ideal/nadir.

    use_opt : bool
        If True, HV is computed using the non-dominated front per generation.
        If False, HV is computed using the whole population.

    Returns
    -------
    relative_hv_history : list[float]
        Hypervolume values in [0, 1].

    ideal : np.ndarray
        Ideal point used for normalization.

    nadir : np.ndarray
        Nadir point used for normalization.
    """

    if ideal is None or nadir is None:
        ideal, nadir = _estimate_ideal_nadir_from_history(results)

    if ideal is None or nadir is None:
        return [], None, None

    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)

    n_obj = len(ideal)
    ref_point = np.ones(n_obj)

    hv = _hv_indicator(ref_point)

    relative_hv_history = []

    for hist in results.history:
        F = _get_F_from_generation(hist, use_opt=use_opt)

        if len(F) == 0:
            relative_hv_history.append(np.nan)
            continue

        F_norm = _normalize_minimization_objectives(F, ideal, nadir)

        # Keep only points strictly inside or on the unit reference box
        F_norm = F_norm[np.all(F_norm <= ref_point, axis=1)]

        if len(F_norm) == 0:
            relative_hv_history.append(0.0)
            continue

        try:
            hv_value = float(hv(F_norm))
        except TypeError:
            hv_value = float(hv.do(F_norm))

        # Because the reference box volume is 1, HV is already relative.
        hv_value = float(np.clip(hv_value, 0.0, 1.0))

        relative_hv_history.append(hv_value)

    return relative_hv_history, ideal, nadir


def evaluate_model_score_on_batch(
    model_instance,
    batch,
    normalize=True,
):
    """
    Compute the accuracy-like score:

        S = mean_n [1 / (1 + RMSE_n)]

    for one model evaluated on one complete horizon-set.
    """

    scores = []

    for ts in batch:
        try:
            result = evaluate_one_individual_series(
                ML=model_instance,
                series_raw=ts,
                sdtw_normalize=normalize,
                return_raw_scale=False,
            )

            rmse = float(result["rmse_used"])
            score = 1.0 / (1.0 + rmse)

            scores.append(score)

        except ValueError:
            continue

    if len(scores) == 0:
        return np.nan

    return float(np.mean(scores))


def compute_signed_omega_metrics(
    model_before,
    model_after,
    List_of_batches,
    batch_id,
    diagonal_scores,
    normalize=True,
):
    """
    Compute the signed Omega metrics at incremental step i.

    Parameters
    ----------
    model_before : Instance
        Model M_{i-1}, deployed before processing batch B_i.

    model_after : Instance
        Final model M_i after:
            - no update,
            - incremental FDM adaptation, or
            - warm-start retuning.

    List_of_batches : list
        All horizon-sets B_0, ..., B_i.

    batch_id : int
        Current incremental step i.

    diagonal_scores : dict
        Stores S_{j,j}, the score of model M_j on batch B_j.

        Initialize with:
            diagonal_scores[0] = S_{0,0}

    Returns
    -------
    metrics : dict
        Signed Omega metrics for the current step.

    Notes
    -----
    All Omega values lie theoretically in [-1, 1].

    Positive:
        improvement relative to the reference.

    Zero:
        stable performance.

    Negative:
        degradation relative to the reference.
    """

    i = int(batch_id)

    if i < 1:
        raise ValueError("Omega metrics start from batch_id=1.")

    missing_references = [
        j for j in range(i)
        if j not in diagonal_scores
    ]

    if missing_references:
        raise ValueError(
            "Missing diagonal reference scores for batches: "
            f"{missing_references}"
        )

    # ============================================================
    # 1. New-batch assimilation
    # ============================================================

    # S_{i-1,i}: previous model on incoming batch
    score_before_new = evaluate_model_score_on_batch(
        model_instance=model_before,
        batch=List_of_batches[i],
        normalize=normalize,
    )

    # S_{i,i}: final model on incoming batch
    score_after_new = evaluate_model_score_on_batch(
        model_instance=model_after,
        batch=List_of_batches[i],
        normalize=normalize,
    )

    omega_new = score_after_new - score_before_new

    # ============================================================
    # 2. Memory retention on every previous batch
    # ============================================================

    current_scores_on_old_batches = {}
    omega_memory_by_batch = {}

    for j in range(i):

        # S_{i,j}: current final model evaluated on old batch B_j
        score_current = evaluate_model_score_on_batch(
            model_instance=model_after,
            batch=List_of_batches[j],
            normalize=normalize,
        )

        current_scores_on_old_batches[j] = score_current

        # Omega_memory,bj = S_{i,j} - S_{j,j}
        omega_memory_by_batch[
            f"omega_memory_b{j}"
        ] = score_current - diagonal_scores[j]

    # ============================================================
    # 3. Base retention
    # ============================================================

    omega_base = omega_memory_by_batch["omega_memory_b0"]

    # ============================================================
    # 4. Previous-horizon retention
    # ============================================================

    omega_previous = (
        current_scores_on_old_batches[i - 1]
        - diagonal_scores[i - 1]
    )

    # ============================================================
    # 5. Average memory retention
    # ============================================================

    memory_values = np.asarray(
        list(omega_memory_by_batch.values()),
        dtype=np.float64,
    )

    omega_memory_avg = float(
        np.mean(memory_values)
    )

    # ============================================================
    # 6. Conservative memory-adaptation balance
    # ============================================================

    if (
        np.isfinite(omega_memory_avg)
        and np.isfinite(omega_new)
    ):
        omega_balance = float(
            min(omega_memory_avg, omega_new)
        )
    else:
        omega_balance = np.nan

    metrics = {
        "step": i,
        "new_batch_id": i,

        "score_before_new": score_before_new,
        "score_after_new": score_after_new,

        "omega_base": float(omega_base),
        "omega_previous": float(omega_previous),
        "omega_memory_avg": omega_memory_avg,
        "omega_new": float(omega_new),
        "omega_balance": omega_balance,
    }

    metrics.update(omega_memory_by_batch)

    # Store S_{i,i}; this becomes the reference score for batch B_i
    # during all future incremental steps.
    diagonal_scores[i] = score_after_new

    return metrics
# ============================================================
# AMLM FRAMEWORK!!!!!
# ============================================================

def get_ml_hyperparameters(chromosome):
    return {
        "window": chromosome["window"],
        "learning_rate": chromosome["learning_rate"],
        "batch_size": chromosome["batch_size"],
        "hidden_dim": chromosome["hidden_dim"],
        "epochs": chromosome["epochs"],
        "layer_dim": chromosome["layer_dim"],
    }



def RunAMLM(
    List_of_batches,
    num_population,
    num_offsprings,
    num_generations,

    # Soft-DTW / clustering parameters
    gamma=1.0,
    max_iter=100,
    barycenter_max_iter=50,
    normalize=True,
    use_divergence=True,
    min_cluster_size=1,

    # LSTM train split over concatenated barycenters
    split=0.70,

    # Failure detection
    D_star_tp=1.8,
    failure_threshold_percent=5.0,
    post_failure_threshold_percent=5.0,

    # FDM update parameters
    energy_keep=0.99,
    max_components=5,
    exact_diagnostic=False,

    # Warm-start retuning fallback
    allow_retuning=True,
    warm_num_population=2,
    warm_num_offsprings=2,
    warm_num_generations=1,
    warm_neigh_per=0.20,

    # ASF weights
    asf_weights=None,
    warm_asf_weights=None,

    # Reproducibility / output
    # random_state=42,
    save_txt_path=None,
    track_energy=True,
):
    """
    Run the AMLM framework with failed-only FDM barycenter updates.

    Workflow
    --------
    1. Run full MODE on the initial batch B0.
    2. Select the best model/architecture with ASF.
    3. For every incoming batch Bi:
        a. Evaluate the current model on all time series in Bi.
        b. Compute global failure ratio: failed series / total batch size.
        c. If the ratio is below threshold, do nothing.
        d. If the ratio exceeds threshold:
            - assign only failed series to the closest barycenters,
            - update only barycenters receiving failed series,
            - retrain the LSTM on the updated barycenters,
            - re-evaluate Bi.
        e. If the update still fails, optionally run warm-start MODE:
            - fixed updated barycenters,
            - no k optimization,
            - local search around previous best architecture.

    Energy tracking
    ---------------
    CodeCarbon energy measurements are stored for:
        - initial MODE tuning,
        - FDM barycenter update,
        - LSTM retraining,
        - warm-start MODE retuning.

    Catastrophic-forgetting metrics
    -------------------------------
    Stores:
        - F_RMSE, RF_avg
        - A_RMSE, RA_avg
        - omega_base
        - omega_previous
        - omega_memory_bj
        - omega_memory_avg
        - omega_new
        - omega_balance
    """

    import time
    import numpy as np
    import pandas as pd
    from pprint import pformat

    if asf_weights is None:
        asf_weights = np.array([0.6, 0.2, 0.2])

    if warm_asf_weights is None:
        warm_asf_weights = np.array([0.6, 0.4])

    if len(List_of_batches) < 2:
        raise ValueError("RunAMLM requires at least two batches.")

    warm_num_population = int(warm_num_population)
    warm_num_offsprings = int(warm_num_offsprings)
    warm_num_generations = int(warm_num_generations)

    results = {
        "initial_hpo": {},
        "batches": {},
        "metrics": {},
        "final_state": {},
    }

    # ============================================================
    # 1. Initial MODE / HPO on batch 0
    # ============================================================

    print("\n" + "=" * 80)
    print("Initial MODE tuning on batch 0")
    print("=" * 80)

    hpo_start = time.time()

    hpo_output, hpo_energy = run_with_energy_tracking(
        func=lambda: ApplyingMODE(
            data=List_of_batches[0],
            num_population=num_population,
            num_offsprings=num_offsprings,
            num_generations=num_generations,
            gamma=gamma,
            max_iter=max_iter,
            barycenter_max_iter=barycenter_max_iter,
            normalize=normalize,
            use_divergence=use_divergence,
            min_cluster_size=min_cluster_size,
            split=split,
            # random_state=random_state,
            D_star_tp=D_star_tp,
        ),
        label="initial_MODE_tuning",
        enable=track_energy,
    )

    results_cl_centralized, solved_Problem, hpo_duration = hpo_output

    best_instance, best_model, best_chromosome, best_info = ApplyingASF(
        results_of_tuning=results_cl_centralized,
        Problem_class=solved_Problem,
        weights=asf_weights,
        return_full=True,
    )

    hpo_total_duration = time.time() - hpo_start

    results["initial_hpo"] = {
        "best_chromosome": best_chromosome,
        "best_F": best_info.get("best_F", None),
        "mode_duration": hpo_duration,
        "total_duration": hpo_total_duration,
        "energy": hpo_energy,
    }

    # ============================================================
    # 2. Initialize FDM memory state
    # ============================================================

    current_instance = best_instance
    current_chromosome = best_chromosome

    fdm_state = initialize_fdm_memory_state(
        best_info=best_info,
        best_chromosome=best_chromosome,
    )

    print("\nInitial online state")
    print("K:", fdm_state["K"])
    print("Cluster sizes:", fdm_state["cluster_sizes"].astype(int))

    def _architecture_without_k(chromosome):
        return {
            "window": chromosome["window"],
            "learning_rate": chromosome["learning_rate"],
            "batch_size": chromosome["batch_size"],
            "hidden_dim": chromosome["hidden_dim"],
            "epochs": chromosome["epochs"],
            "layer_dim": chromosome["layer_dim"],
        }

    # ============================================================
    # 3. Initialize catastrophic-forgetting metric storage
    # ============================================================

    metric_summary_rows = []
    old_metric_detail_tables = []
    new_metric_detail_tables = []

    # Reference table:
    # batch 0 reference = M0 evaluated on B0.
    base_reference_df = evaluate_model_rmse_on_batches(
        model_instance=current_instance,
        List_of_batches=List_of_batches,
        batch_indices=[0],
        normalize=normalize,
        rmse_col="rmse_reference",
    )

    reference_tables = [base_reference_df]

    def _compute_and_store_metrics(
        batch_idx,
        old_before_df,
        new_before_df,
        final_model_instance,
        decision_name,
        batch_log,
    ):
        """
        Compute F/A/Omega metrics for one incremental round and update
        metric storage.
        """

        old_batch_indices = list(range(batch_idx))

        old_after_df = evaluate_model_rmse_on_batches(
            model_instance=final_model_instance,
            List_of_batches=List_of_batches,
            batch_indices=old_batch_indices,
            normalize=normalize,
            rmse_col="rmse_old_after",
        )

        new_after_df = evaluate_model_rmse_on_batches(
            model_instance=final_model_instance,
            List_of_batches=List_of_batches,
            batch_indices=[batch_idx],
            normalize=normalize,
            rmse_col="rmse_new_after",
        )

        old_reference_df = pd.concat(
            reference_tables,
            ignore_index=True,
        )

        summary_row, old_details_df, new_details_df = (
            compute_catastrophic_forgetting_metrics(
                old_before_df=old_before_df,
                old_after_df=old_after_df,
                new_before_df=new_before_df,
                new_after_df=new_after_df,
                old_reference_df=old_reference_df,
                step_id=batch_idx,
                batch_id=batch_idx,
                method_name=decision_name,
            )
        )

        metric_summary_rows.append(summary_row)

        old_details_df.insert(0, "step", batch_idx)
        new_details_df.insert(0, "step", batch_idx)

        old_metric_detail_tables.append(old_details_df)
        new_metric_detail_tables.append(new_details_df)

        # Store Mi on Bi as future reference for omega_memory_bi.
        reference_tables.append(
            make_reference_from_new_after(new_after_df)
        )

        batch_log["metrics"] = summary_row

        return summary_row

    # ============================================================
    # 4. Online loop over incoming batches
    # ============================================================

    for batch_idx in range(1, len(List_of_batches)):

        print("\n" + "=" * 80)
        print(f"Incoming batch {batch_idx}/{len(List_of_batches) - 1}")
        print("=" * 80)

        new_batch = List_of_batches[batch_idx]
        old_batch_indices = list(range(batch_idx))

        batch_log = {
            "batch_id": batch_idx,
            "pre_update": {},
            "assignment": {},
            "fdm_update": {},
            "post_update": {},
            "retuning": {},
            "metrics": {},
        }

        # --------------------------------------------------------
        # A0. Metric evaluation before update
        # --------------------------------------------------------

        old_before_df = evaluate_model_rmse_on_batches(
            model_instance=current_instance,
            List_of_batches=List_of_batches,
            batch_indices=old_batch_indices,
            normalize=normalize,
            rmse_col="rmse_old_before",
        )

        new_before_df = evaluate_model_rmse_on_batches(
            model_instance=current_instance,
            List_of_batches=List_of_batches,
            batch_indices=[batch_idx],
            normalize=normalize,
            rmse_col="rmse_new_before",
        )

        # --------------------------------------------------------
        # A. Evaluate current model on incoming batch for failure detection
        # --------------------------------------------------------

        eval_before_df, failed_indices, failure_summary = (
            evaluate_incoming_batch_failures(
                model_instance=current_instance,
                new_batch=new_batch,
                normalize=normalize,
                D_star_tp=D_star_tp,
                failure_threshold_percent=failure_threshold_percent,
            )
        )

        batch_log["pre_update"] = {
            "eval_df": eval_before_df,
            "failed_indices": failed_indices,
            "summary": failure_summary,
        }

        print("Pre-update failure summary:")
        print(failure_summary)

        # --------------------------------------------------------
        # B. If failure ratio is below threshold, do nothing
        # --------------------------------------------------------

        if not failure_summary["retraining_required"]:

            print("Failure ratio below threshold. No retraining performed.")

            batch_log["decision"] = "no_update"
            batch_log["current_chromosome"] = current_chromosome
            batch_log["cluster_sizes"] = fdm_state["cluster_sizes"].copy()

            _compute_and_store_metrics(
                batch_idx=batch_idx,
                old_before_df=old_before_df,
                new_before_df=new_before_df,
                final_model_instance=current_instance,
                decision_name=batch_log["decision"],
                batch_log=batch_log,
            )

            results["batches"][f"batch_{batch_idx}"] = batch_log

            continue

        # --------------------------------------------------------
        # C. Assign only failed series to closest barycenters
        # --------------------------------------------------------

        print("Retraining required.")
        print(f"Failed series: {len(failed_indices)} / {len(new_batch)}")

        assignment_df, failed_members_by_cluster, affected_clusters = (
            assign_failed_series_to_barycenters(
                new_batch=new_batch,
                failed_indices=failed_indices,
                current_barycenters=fdm_state["current_barycenters"],
                gamma=gamma,
                normalize=normalize,
                use_divergence=use_divergence,
            )
        )

        batch_log["assignment"] = {
            "assignment_df": assignment_df,
            "affected_clusters": affected_clusters,
        }

        print("Affected clusters:", affected_clusters)

        if len(affected_clusters) == 0:

            print("No affected clusters after assignment. No update performed.")

            batch_log["decision"] = "failure_detected_but_no_assignment"
            batch_log["current_chromosome"] = current_chromosome
            batch_log["cluster_sizes"] = fdm_state["cluster_sizes"].copy()

            _compute_and_store_metrics(
                batch_idx=batch_idx,
                old_before_df=old_before_df,
                new_before_df=new_before_df,
                final_model_instance=current_instance,
                decision_name=batch_log["decision"],
                batch_log=batch_log,
            )

            results["batches"][f"batch_{batch_idx}"] = batch_log

            continue

        # --------------------------------------------------------
        # D. Update only barycenters receiving failed series
        # --------------------------------------------------------

        update_result, update_energy = run_with_energy_tracking(
            func=lambda: update_barycenters_from_failed_assignments_fdm(
                current_barycenters=fdm_state["current_barycenters"],
                cluster_sizes=fdm_state["cluster_sizes"],
                cluster_members_by_cluster=fdm_state["cluster_members_by_cluster"],
                failed_members_by_cluster=failed_members_by_cluster,
                gamma=gamma,
                normalize=normalize,
                max_iter=barycenter_max_iter,
                energy_keep=energy_keep,
                max_components=max_components,
                exact_diagnostic=exact_diagnostic,
            ),
            label=f"batch_{batch_idx}_fdm_update",
            enable=track_energy,
        )

        update_duration = update_energy["duration_sec"]

        fdm_state["current_barycenters"] = update_result["bar_current"]
        fdm_state["cluster_sizes"] = update_result["cluster_sizes"]
        fdm_state["cluster_members_by_cluster"] = update_result[
            "cluster_members_by_cluster"
        ]

        batch_log["fdm_update"] = {
            "duration": update_duration,
            "energy": update_energy,
            "update_df": update_result["update_df"],
        }

        print("FDM update table:")
        print(update_result["update_df"])

        # --------------------------------------------------------
        # E. Retrain LSTM on updated barycenters
        # --------------------------------------------------------

        updated_instance, retrain_energy = run_with_energy_tracking(
            func=lambda: train_model_from_barycenters(
                barycenters=fdm_state["current_barycenters"],
                architecture=_architecture_without_k(current_chromosome),
                split=split,
                normalized=normalize,
                # seed=random_state,
            ),
            label=f"batch_{batch_idx}_lstm_retraining",
            enable=track_energy,
        )

        retrain_duration = retrain_energy["duration_sec"]

        # --------------------------------------------------------
        # F. Evaluate updated model on the same incoming batch
        # --------------------------------------------------------

        eval_after_df, failed_indices_after, failure_summary_after = (
            evaluate_incoming_batch_failures(
                model_instance=updated_instance,
                new_batch=new_batch,
                normalize=normalize,
                D_star_tp=D_star_tp,
                failure_threshold_percent=post_failure_threshold_percent,
            )
        )

        batch_log["post_update"] = {
            "retrain_duration": retrain_duration,
            "retrain_energy": retrain_energy,
            "eval_df": eval_after_df,
            "failed_indices": failed_indices_after,
            "summary": failure_summary_after,
        }

        print("Post-update failure summary:")
        print(failure_summary_after)

        current_instance = updated_instance

        # --------------------------------------------------------
        # G. If still bad, optionally run warm-start MODE
        # --------------------------------------------------------

        if failure_summary_after["retraining_required"] and allow_retuning:

            print("Post-update failure ratio still above threshold.")
            print("Running warm-start MODE on the updated barycenter representation...")

            warm_output, warm_energy = run_with_energy_tracking(
                func=lambda: ApplyingMODE_WarmStart(
                    fixed_barycenters=fdm_state["current_barycenters"],
                    evaluation_data=new_batch,
                    previous_chromosome=current_chromosome,
                    num_population=warm_num_population,
                    num_offsprings=warm_num_offsprings,
                    num_generations=warm_num_generations,
                    neigh_per=warm_neigh_per,
                    split=split,
                    normalize=normalize,
                    # random_state=random_state,
                    D_star_tp=D_star_tp,
                ),
                label=f"batch_{batch_idx}_warm_start_retuning",
                enable=track_energy,
            )

            results_warm, warm_problem, warm_duration = warm_output

            warm_best_instance, warm_best_model, warm_best_chromosome, warm_best_info = (
                ApplyingASF_WarmStart(
                    results_of_tuning=results_warm,
                    Problem_class=warm_problem,
                    previous_chromosome=current_chromosome,
                    weights=warm_asf_weights,
                    return_full=True,
                )
            )

            retune_total_duration = warm_energy["duration_sec"]

            # Warm-start retuning changes only the predictive model/hyperparameters.
            # The FDM barycenters and cluster memory remain unchanged.
            current_instance = warm_best_instance
            current_chromosome = warm_best_chromosome

            eval_after_warm_df, failed_indices_after_warm, failure_summary_after_warm = (
                evaluate_incoming_batch_failures(
                    model_instance=current_instance,
                    new_batch=new_batch,
                    normalize=normalize,
                    D_star_tp=D_star_tp,
                    failure_threshold_percent=post_failure_threshold_percent,
                )
            )

            batch_log["retuning"] = {
                "triggered": True,
                "type": "warm_start_MODE",
                "warm_num_population": warm_num_population,
                "warm_num_offsprings": warm_num_offsprings,
                "warm_num_generations": warm_num_generations,
                "warm_neigh_per": warm_neigh_per,
                "warm_duration": warm_duration,
                "total_duration": retune_total_duration,
                "energy": warm_energy,
                "new_best_chromosome": current_chromosome,
                "new_best_F": warm_best_info.get("best_F", None),
                "eval_after_warm_df": eval_after_warm_df,
                "failed_indices_after_warm": failed_indices_after_warm,
                "summary_after_warm": failure_summary_after_warm,
                "K_preserved": current_chromosome["num_clusters"],
            }

            if failure_summary_after_warm["retraining_required"]:
                batch_log["decision"] = "warm_start_retuned_but_still_problematic"
            else:
                batch_log["decision"] = "warm_start_retuned_successful"

        else:

            batch_log["retuning"] = {
                "triggered": False,
            }

            if failure_summary_after["retraining_required"]:
                batch_log["decision"] = "fdm_update_done_but_still_problematic"
            else:
                batch_log["decision"] = "fdm_update_successful"

        batch_log["current_chromosome"] = current_chromosome
        batch_log["cluster_sizes"] = fdm_state["cluster_sizes"].copy()

        # --------------------------------------------------------
        # H. Catastrophic-forgetting metrics after final model
        # --------------------------------------------------------

        _compute_and_store_metrics(
            batch_idx=batch_idx,
            old_before_df=old_before_df,
            new_before_df=new_before_df,
            final_model_instance=current_instance,
            decision_name=batch_log["decision"],
            batch_log=batch_log,
        )

        results["batches"][f"batch_{batch_idx}"] = batch_log

    # ============================================================
    # 5. Final metric tables
    # ============================================================

    metrics_summary_df = pd.DataFrame(metric_summary_rows)

    old_metrics_details_df = (
        pd.concat(old_metric_detail_tables, ignore_index=True)
        if len(old_metric_detail_tables) > 0
        else pd.DataFrame()
    )

    new_metrics_details_df = (
        pd.concat(new_metric_detail_tables, ignore_index=True)
        if len(new_metric_detail_tables) > 0
        else pd.DataFrame()
    )

    reference_rmse_df = (
        pd.concat(reference_tables, ignore_index=True)
        if len(reference_tables) > 0
        else pd.DataFrame()
    )

    results["metrics"] = {
        "summary_df": metrics_summary_df,
        "old_details_df": old_metrics_details_df,
        "new_details_df": new_metrics_details_df,
        "reference_rmse_df": reference_rmse_df,
    }

    # ============================================================
    # 6. Final state
    # ============================================================

    results["final_state"] = {
        "current_instance": current_instance,
        "current_chromosome": current_chromosome,
        "fdm_state": fdm_state,
    }

    if save_txt_path is not None:
        with open(save_txt_path, "w", encoding="utf-8") as f:
            f.write("AMLM = ")
            f.write(pformat(results, width=120, sort_dicts=False))

    print("\nAMLM ALL DONE!")

    return results



# =========================================
# Referensh-Ancored closed-end change-point
# =========================================

def Referensh_Ancored_Criterion(
    H_ref,
    H_new,
    model,
    alpha_p=0.05,
    normalize=True,
    return_indices=False,
):
    """
    Reference-anchored closed-end change-point detector.

    Parameters
    ----------
    H_ref : array-like, shape (N, T_ref)
        Reference horizon-set accepted with the deployed model.

    H_new : array-like, shape (N, T_new)
        Fully observed current horizon-set.

    model : Instance
        Currently deployed forecasting model.

    alpha_p : float
        Per-profile significance level.

    normalize : bool
        Whether each time series is normalized before prediction.

    Returns
    -------
    ratio : float
        Fraction of profiles for which degradation was detected:

            ratio = number_of_flagged_profiles / total_profiles
    """

    

    H_ref = np.asarray(H_ref, dtype=np.float64)
    H_new = np.asarray(H_new, dtype=np.float64)

    if H_ref.ndim != 2 or H_new.ndim != 2:
        raise ValueError(
            "H_ref and H_new must have shape (number_of_profiles, series_length)."
        )

    if len(H_ref) != len(H_new):
        raise ValueError(
            "H_ref and H_new must contain the same number of profiles."
        )

    if not 0.0 < alpha_p < 1.0:
        raise ValueError("alpha_p must be between 0 and 1.")

    eps = 1e-12
    drifted_indices = []
    number_of_profiles = len(H_ref)

    for profile_id in range(number_of_profiles):

        # --------------------------------------------------------
        # 1. Generate reference squared prediction errors
        # --------------------------------------------------------
        ref_result = evaluate_one_individual_series(
            ML=model,
            series_raw=H_ref[profile_id],
            sdtw_normalize=normalize,
            return_raw_scale=False,
        )

        reference_residuals = (
            ref_result["preds_used"] -
            ref_result["targets_used"]
        )

        Z_ref = np.asarray(
            reference_residuals ** 2,
            dtype=np.float64,
        )

        # --------------------------------------------------------
        # 2. Generate current squared prediction errors
        # --------------------------------------------------------
        new_result = evaluate_one_individual_series(
            ML=model,
            series_raw=H_new[profile_id],
            sdtw_normalize=normalize,
            return_raw_scale=False,
        )

        current_residuals = (
            new_result["preds_used"] -
            new_result["targets_used"]
        )

        Z_new = np.asarray(
            current_residuals ** 2,
            dtype=np.float64,
        )

        T_ref = len(Z_ref)
        T_new = len(Z_new)

        if T_ref < 5 or T_new < 1:
            raise ValueError(
                f"Insufficient residuals for profile {profile_id}."
            )

        # --------------------------------------------------------
        # 3. Fixed reference squared-error mean
        # --------------------------------------------------------
        reference_mean = float(np.mean(Z_ref))

        # --------------------------------------------------------
        # 4. Reference-only AR(1)-prewhitened QS-HAC LRV
        # --------------------------------------------------------

        # Recursive demeaning for the AR(1) prefilter
        recursive_mean = (
            np.cumsum(Z_ref) /
            np.arange(1, T_ref + 1)
        )

        recursive_centered = Z_ref - recursive_mean

        ar_denominator = np.dot(
            recursive_centered[:-1],
            recursive_centered[:-1],
        )

        if ar_denominator > eps:
            ar_coefficient = (
                np.dot(
                    recursive_centered[:-1],
                    recursive_centered[1:],
                )
                / ar_denominator
            )
        else:
            ar_coefficient = 0.0

        stability_limit = 1.0 - 1.0 / T_ref

        ar_coefficient = float(
            np.clip(
                ar_coefficient,
                -stability_limit,
                stability_limit,
            )
        )

        # Prewhiten the ordinarily centered reference sequence
        centered_reference = Z_ref - reference_mean

        prewhitened = (
            centered_reference[1:] -
            ar_coefficient * centered_reference[:-1]
        )

        prewhitened = prewhitened - np.mean(prewhitened)

        n_pw = len(prewhitened)

        # Auxiliary AR(1) coefficient for automatic QS bandwidth
        auxiliary_denominator = np.dot(
            prewhitened[:-1],
            prewhitened[:-1],
        )

        if auxiliary_denominator > eps:
            auxiliary_ar = (
                np.dot(
                    prewhitened[:-1],
                    prewhitened[1:],
                )
                / auxiliary_denominator
            )
        else:
            auxiliary_ar = 0.0

        auxiliary_limit = 1.0 - 1.0 / max(n_pw, 2)

        auxiliary_ar = float(
            np.clip(
                auxiliary_ar,
                -auxiliary_limit,
                auxiliary_limit,
            )
        )

        alpha_2 = (
            4.0 * auxiliary_ar ** 2
            / max((1.0 - auxiliary_ar) ** 4, eps)
        )

        bandwidth = max(
            n_pw ** (1.0 / 5.0),
            1.3221 * (max(alpha_2, 0.0) * n_pw) ** (1.0 / 5.0),
        )

        # Reference-only autocovariances
        autocovariances = np.empty(n_pw, dtype=np.float64)

        autocovariances[0] = (
            np.dot(prewhitened, prewhitened) / n_pw
        )

        for lag in range(1, n_pw):
            autocovariances[lag] = (
                np.dot(
                    prewhitened[lag:],
                    prewhitened[:-lag],
                )
                / n_pw
            )

        # Quadratic-spectral kernel
        lags = np.arange(1, n_pw, dtype=np.float64)
        x = lags / bandwidth
        a = 6.0 * np.pi * x / 5.0

        qs_weights = (
            25.0
            / (12.0 * np.pi ** 2 * x ** 2)
            * (np.sin(a) / a - np.cos(a))
        )

        prewhitened_lrv = (
            autocovariances[0]
            + 2.0 * np.dot(
                qs_weights,
                autocovariances[1:],
            )
        )

        # Recolor to obtain the reference LRV
        lrv_variance = (
            prewhitened_lrv
            / max((1.0 - ar_coefficient) ** 2, eps)
        )

        # Numerical fallback
        if not np.isfinite(lrv_variance) or lrv_variance <= eps:
            lrv_variance = max(
                np.var(Z_ref, ddof=1),
                eps,
            )

        reference_lrv = np.sqrt(lrv_variance)

        # --------------------------------------------------------
        # 5. Reference-anchored backward current-horizon scan
        # --------------------------------------------------------
        centered_current = Z_new - reference_mean

        suffix_sums = np.cumsum(
            centered_current[::-1]
        )[::-1]

        suffix_lengths = np.arange(
            T_new,
            0,
            -1,
            dtype=np.float64,
        )

        statistics = (
            suffix_sums
            /
            (
                reference_lrv
                * np.sqrt(T_ref)
                * (1.0 + suffix_lengths / T_ref)
            )
        )

        maximum_statistic = float(np.max(statistics))

        # --------------------------------------------------------
        # 6. Analytical one-sided critical value
        # --------------------------------------------------------
        critical_value = (
            np.sqrt(T_new / (T_ref + T_new))
            * norm.ppf(1.0 - alpha_p / 2.0)
        )

        if maximum_statistic > critical_value:
            drifted_indices.append(profile_id)

    # ------------------------------------------------------------
    # 7. Final horizon-level degradation ratio
    # ------------------------------------------------------------
    ratio = (
        len(drifted_indices) /
        number_of_profiles
    )
    if return_indices:
        return float(ratio), np.asarray(drifted_indices, dtype=int)

    return float(ratio)



