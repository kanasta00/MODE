# '''Ming-Wei Li, Xiang-Yang Li, Yu-Tian Wang, Zhong-Yi Yang, Wei-Chiang Hong,
# Chaos crossover quantum attraction-repulsion optimization algorithm,
# Swarm and Evolutionary Computation,
# Volume 92,
# 2025,
# 101811,
# ISSN 2210-6502,
# https://doi.org/10.1016/j.swevo.2024.101811.
# (https://www.sciencedirect.com/science/article/pii/S2210650224003493)
# Abstract: The Attraction-Repulsion Optimization Algorithm (AROA) is a novel optimization algorithm that balances exploration and exploitation by mimicking the natural equilibrium associated with attraction-repulsion phenomena. While AROA exhibits strong optimization capabilities and fast convergence speed, its global search ability is relatively weak, making it prone to local optima in later iterations. To address these issues, this paper proposes a chaotic crossover quantum attraction-repulsion optimization algorithm (CCQAROA). First, a two-dimensional Hénon-Sine hyperchaotic map is introduced for population initialization to achieve a more uniformly distributed initial population, thereby improving convergence speed. Second, a crossover strategy inspired by differential evolution is applied, probabilistically retaining dimensional information from parent individuals to balance exploration and exploitation. Finally, a novel quantum mutation strategy is introduced, which perturbs the global best solution when the algorithm stagnates, helping the algorithm escape from local optima. The CCQAROA was tested using the CEC2014, CEC2017, and CEC2022 benchmark test suites. The performance of CCQAROA was compared with nine advanced algorithms, and the results demonstrated that CCQAROA can attain competitive or even better results. Furthermore, CCQAROA was applied to solve three engineering problems, and the results confirmed that the proposed improvements to AROA are both feasible and effective.
# Keywords: Aroa; Chaoitc initialization; Crossover operation; Quantum mutation'''
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore
import time
import csv
# import sys


def henon_sine_hyperchaotic_init(N, Dim, param_ranges, 
                                 alpha=1.4, beta=0.3,
                                   gamma_range=(0,4),
                                     iterations=20):
    """
    N: population size
    Dim: number of dimensions
    param_ranges: list of tuples [(min1,max1), (min2,max2), ...] for each dimension
    iterations: number of chaotic iterations to generate sequence
    """
    X = np.zeros((N, Dim), dtype=float)
    for i in range(N):
        # Initialize x and y randomly in (0,1)
        x = np.random.rand()
        y = np.random.rand()
        gamma = np.random.uniform(*gamma_range)
        # Iterate to generate chaotic sequence
        for _ in range(iterations):
            x_new = (1 - alpha*x**2 + beta*y*101) % 1
            y_new = (gamma * np.sin(np.pi * x * y) * 101) % 1
            x, y = x_new, y_new
        # Assign the final x value to all dimensions of agent i
        # X[i,:] = x

        # Map chaotic x to integer ranges
        for d in range(Dim):
            min_d, max_d = param_ranges[d]
            if d == Dim-1:
                X[i, d] = min_d + x * (max_d - min_d)        
            else:
                X[i, d] = int(round(min_d + x * (max_d - min_d)))
        # X[i, 5] = float(round(min_d + x * (max_d - min_d)))
        
    return X

# def min_max_normalize(ts, how, MIN, MAX):
#     if how == "alone":    
#         min = np.min(ts)
#         max = np.max(ts)
#         return (ts - min) / (max - min), min, max
#     else:
#         return (ts - min) / (max - min)

# def reverse_min_max(ts, min, max):
#     return ts * (max - min) + min

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        x.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)




class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# def calculate_r2_score(y_true, y_pred):
#     ss_total = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8 
#     ss_residual = np.sum((y_true - y_pred) ** 2)
#     return 1 - (ss_residual / ss_total)



def fitness(individual, Train_Data): 
    
    # print("hello")
    split_idx = int(len(Train_Data) * 0.75)
    X_train_split = Train_Data[:split_idx]
    X_val_split = Train_Data[split_idx:]

    ws, nd, ep, nl, bs, lr = individual
    
    window_size = int(ws)
    # print(window_size)
    nodes = int(nd)
    num_epochs = int(ep)
    num_layers = int(nl)
    learning_rate = float(lr)
    batch_size = int(bs)
    # print(learning_rate)
    minTr = np.min(X_train_split)
    maxTr = np.max(X_train_split)

    X_train_normalized = (X_train_split- minTr)/(maxTr-minTr)
    X_val_normalized = (X_val_split- minTr)/(maxTr-minTr)
    
    
    x_train, y_train = create_sequences(X_train_normalized, window_size)
    # x_train = x_train.float().unsqueeze(-1)  # Add channel dim
    # y_train = y_train.float()

    # Convert validation data to input-output sequences
    x_val, y_val = create_sequences(X_val_normalized, window_size)
    # x_val = x_val.float().unsqueeze(-1)
    # y_val = y_val.float()

    # Convert to torch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor(y_val, dtype=torch.float32)


    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = LSTMModel(
        input_dim=1,
        hidden_dim=nodes,
        layer_dim=num_layers,
        output_dim=1
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # -----------------------------
    # Training loop
    # -----------------------------
    for _ in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb.squeeze())
            loss.backward()
            optimizer.step()

    # -----------------------------
    # Validation RMSE
    # -----------------------------
    model.eval()
    preds = []

    with torch.no_grad():
        for xb, _ in val_loader:
            p = model(xb)
            preds.append(p)

    y_pred = torch.cat(preds).cpu().numpy().flatten()
    y_true = y_val.cpu().numpy().flatten()

    rmse = calculate_rmse(y_true, y_pred)
    return rmse


# -----------------------------
# Distance Matrix
# -----------------------------
def distance_matrix(X):
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i,j] = np.linalg.norm(X[i] - X[j])
    return D

# -----------------------------
# Attraction-Repulsion
# -----------------------------
def attraction_repulsion(X, f, k=3):
    N = X.shape[0]
    D = distance_matrix(X)
    X_new = X.copy()
    for i in range(N):
        neighbors_idx = np.argsort(D[i])[1:k+1]
        for j in neighbors_idx:
            w = f[j] - f[i]
            step = np.sign(w) * (X[j] - X[i])
            X_new[i] = X_new[i] + np.round(np.random.rand() * step).astype(int)
    return X_new

# -----------------------------
# Global Best Attraction
# -----------------------------
def attract_global_best(X, X_best):
    X_new = X.copy()
    for i in range(X.shape[0]):
        step = X_best - X[i]
        X_new[i] = X_new[i] + np.round(np.random.rand() * step).astype(int)
    return X_new

# -----------------------------
# Local Search Operator
# -----------------------------
def local_search(X, param_ranges):
    X_new = X.copy()
    for i in range(X.shape[0]):
        for d, (min_d, max_d) in enumerate(param_ranges):
            step = np.random.randint(-1, 2)
            X_new[i,d] = np.clip(X_new[i,d] + step, min_d, max_d)
    return X_new

# -----------------------------
# Crossover Operator
# -----------------------------
def crossover_operator(X, param_ranges):
    N = X.shape[0]
    X_new = X.copy()
    for i in range(0, N, 2):
        alpha = np.random.rand()
        X_new[i] = np.round(alpha * X[i] + (1-alpha) * X[(i+1)%N]).astype(int)
        for d, (min_d, max_d) in enumerate(param_ranges):
            X_new[i,d] = np.clip(X_new[i,d], min_d, max_d)
    return X_new

# -----------------------------
# Population-Based Operator
# -----------------------------
def population_based_operators(X, param_ranges):
    X_new = X.copy()
    for i in range(X.shape[0]):
        for d, (min_d, max_d) in enumerate(param_ranges):
            step = np.random.randint(-1, 2)
            X_new[i,d] = np.clip(X_new[i,d] + step, min_d, max_d)
    return X_new

# -----------------------------
# Quantum Update
# -----------------------------
def quantum_update(X_best, param_ranges):
    X_new = X_best.copy()
    for d, (min_d, max_d) in enumerate(param_ranges):
        step = np.random.randint(-2, 3)
        X_new[d] = np.clip(X_new[d] + step, min_d, max_d)
    return X_new

# -----------------------------
# Main CCQAROA Function
# -----------------------------
def CCQAROA_LSTM(N, Dim, param_ranges, X_train_normalized, MaxFEs, k=3):
    start = time.time()
    X = henon_sine_hyperchaotic_init(N, Dim, param_ranges)
    # print(X)
    f = np.array([fitness(ind, X_train_normalized) for ind in X])
    X_best = X[np.argmin(f)].copy()
    f_best = f.min()
    FEs = N
    cnvg = [f_best]
    # print('Hello')
    iter = 0
    while FEs < MaxFEs:
        iter += 1
        print(f"\n=== Iteration {iter} | FEs={FEs} ===", flush=True)
        # print(FEs)
        X = attraction_repulsion(X, f, k)
        X = attract_global_best(X, X_best)
        X = local_search(X, param_ranges)
        X = crossover_operator(X, param_ranges)

        f = np.array([fitness(ind, X_train_normalized) for ind in X])
        FEs += N
        idx_best = np.argmin(f)
        if f[idx_best] < f_best:
            X_best = X[idx_best].copy()
            f_best = f[idx_best]

        X = population_based_operators(X, param_ranges)
        f = np.array([fitness(ind, X_train_normalized) for ind in X])
        FEs += N

        if len(cnvg) > 1 and (cnvg[-1] - cnvg[-2]) > 0:
            X_best = quantum_update(X_best, param_ranges)

        cnvg.append(f_best)
        print(f"FEs={FEs}, Best RMSE={f_best}, Best hyperparameters={X_best}")

    elapsed = time.time() - start
    print(f"Evaluation Time: {elapsed:.6f} seconds")
    print(f"Best Hyperparameters: {X_best}")

    return X_best



def evaluate_on_test(best_individual, Train_Data, Test_Data):

    # -----------------------------
    # Unpack tuned hyperparameters
    # -----------------------------
    ws, nd, ep, nl, bs, lr = best_individual

    window_size  = int(ws)
    nodes        = int(nd)
    num_epochs   = int(ep)
    num_layers   = int(nl)
    batch_size   = int(bs)
    learning_rate = float(lr)

    # -----------------------------
    # Normalize using TRAIN min/max ONLY
    # -----------------------------
    minTr = np.min(Train_Data)
    maxTr = np.max(Train_Data)

    Train_norm = (Train_Data - minTr) / (maxTr - minTr)
    Test_norm  = (Test_Data  - minTr) / (maxTr - minTr)

    # -----------------------------
    # Create sequences
    # -----------------------------
    x_train, y_train = create_sequences(Train_norm, window_size)
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_test, y_test = create_sequences(Test_norm, window_size)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    # -----------------------------
    # Build tuned model
    # -----------------------------
    model = LSTMModel(
        input_dim=1,
        hidden_dim=nodes,
        layer_dim=num_layers,
        output_dim=1
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # -----------------------------
    # Train on the full training set
    # -----------------------------
    for _ in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb.squeeze())
            loss.backward()
            optimizer.step()


    # -----------------------------
    # Predict on TEST set
    # -----------------------------
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb))

    y_pred = torch.cat(preds).cpu().numpy().flatten()
    y_true = y_test.cpu().numpy().flatten()

    # -----------------------------
    # Convert predictions back to ORIGINAL scale
    # -----------------------------
    y_pred_original = y_pred * (maxTr - minTr) + minTr
    y_true_original = y_true * (maxTr - minTr) + minTr

    # -----------------------------
    # Compute RMSE in chunks of 100
    # -----------------------------
    chunk_rmse = []
    N = len(y_true_original)
    
    for start in range(0, N, 100):
        end = min(start + 100, N)
        
        rmse = calculate_rmse(
            y_true_original[start:end],
            y_pred_original[start:end]
        )
        
        chunk_rmse.append((start, end - 1, rmse))

    return y_pred_original, y_true_original, chunk_rmse
