import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim

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

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        x.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)


class Instance:

    def __init__(self, data, chromosome):
        self.data = data
        self.chromosome = chromosome
        self.model = LSTMModel(1, self.chromosome['hidden_dim'],
                                self.chromosome['layer_dim'], 1)
        self.train_data, self.test_data = train_test_split(data, train_size=0.75, shuffle=False)
        self.test_preds = []
        self.test_targets = []
        
        
    def get_dataloader(self, train_stage = True):
        batch = self.chromosome['batch_size']
        window = self.chromosome['window']
        data = self.train_data if train_stage else self.test_data
        X, y = create_sequences(data, window)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset=dataset,
                                           batch_size=batch,
                                           shuffle=train_stage,
                                           drop_last=False)
        return dataloader

    def train_phase(self):
        epochs = self.chromosome['epochs']
        lr = self.chromosome['learning_rate']
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        loader = self.get_dataloader(train_stage=True)
        
        for epoch in tqdm(range(epochs)):
            preds_list = []
            targets_list = []
            train_loss = 0
            for _, (X, y) in enumerate(loader):
                self.model.train()
                x_batch = X.unsqueeze(-1)
                y_batch = y.unsqueeze(1)
                y_pred = self.model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch == epochs - 1:
                    preds_list.append(y_pred.detach().numpy())
                    targets_list.append(y_batch.detach().numpy())
            train_loss /= len(loader)

    def validation_phase(self):
        self.test_preds = []
        self.test_targets = []
        loader = self.get_dataloader(train_stage=False)
        self.model.eval()
        with torch.inference_mode():
            for X, y in loader:
                x_batch = X.unsqueeze(-1)
                y_batch = y.unsqueeze(1)
                preds = self.model(x_batch)
                self.test_preds.append(preds.numpy())
                self.test_targets.append(y_batch.numpy())
        
        self.test_preds = np.concatenate(self.test_preds).flatten()
        self.test_targets = np.concatenate(self.test_targets).flatten()
        rmse = np.sqrt(np.mean((self.test_preds - self.test_targets) ** 2))
        ss_res = np.sum((self.test_targets - self.test_preds) ** 2)
        ss_tot = np.var(self.test_targets) * len(self.test_targets)
        r2 = 1 - (ss_res / ss_tot)
        return rmse, r2, self.test_preds, self.test_targets 


def evaluate_on_new_data(trained_model,
                          new_data,
                            chromosome):
    
    new_instance = Instance(new_data, chromosome)
    min = np.min(new_data)
    max = np.max(new_data)
    new_instance.test_data = (new_data - min) / (max - min)
    new_instance.model = trained_model
    _, _, pred, targ = new_instance.validation_phase()
    preds = pred * (max - min) + min
    targs = targ * (max - min) + min

    rmse = np.sqrt(np.mean((preds - targs) ** 2))
    ss_res = np.sum((targs - preds) ** 2)
    ss_tot = np.var(targs) * len(targs)
    r2 = 1 - (ss_res / ss_tot)

    return rmse, r2, new_instance, preds, targs


from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from joblib import Parallel, delayed




class K_DTW:

    '''Usage example:
    =====================
    clustering = KShapeDTWClustering(K_Clusters=3, max_iter=100, tol=1e-6)
    mem, centroids = clustering.fit(Data)

    '''

    def __init__(self, K_Clusters, max_iter=50, tol=1e-5, random_state=42):
        self.K_Clusters = K_Clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.mem = None

    def initialize_mem(self, N):
        """Initialize the membership array randomly."""
        mem = np.array([i % self.K_Clusters for i in range(N)])
        np.random.shuffle(mem)
        return mem


    def Centroid(self, mem, Data, k):
        """Calculate the centroid for cluster k."""
        iDs = np.where(mem == k)[0].tolist()
        if not iDs:
            random_index = np.random.choice(Data.shape[0])
            cent = Data[random_index]
        elif len(iDs) == 1:
            cent = Data[iDs]
        else:
            members = Data[iDs]
            cent = np.mean(members, axis=0)
        return cent

    def compute_DTW(self, X_t, cent):
        """Compute DTW distance."""
        X_t = X_t.reshape(-1, 1)  # Flatten to 1D
        cent = cent.reshape(-1, 1)  # Flatten to 1D 
        distance, _ = fastdtw(X_t, cent, dist=euclidean)
        return distance

    def fit(self, Data):
        """Fit the k-shape DTW clustering algorithm."""
        N, M = Data.shape
        self.centroids = np.zeros((self.K_Clusters, M))
        self.mem = self.initialize_mem(N)

        for _ in tqdm(range(self.max_iter), desc="Iterations", ncols=100):
            D = np.zeros((N, self.K_Clusters))
            prev_mem = self.mem

            # Update centroids
            for k in range(self.K_Clusters):
                self.centroids[k, :] = self.Centroid(self.mem, Data, k)

            # Compute DTW distances in parallel
            D = Parallel(n_jobs=-1)(
                delayed(self.compute_DTW)(Data[i], self.centroids[k, :]) 
                for i in range(N) for k in range(self.K_Clusters)
            )
            D = np.array(D).reshape(N, self.K_Clusters)

            # Update memberships
            self.mem = np.argmin(D, axis=1)

            # Check for convergence
            if np.linalg.norm(prev_mem - self.mem) < self.tol:
                print("Premature convergence")
                break

        return self.mem, self.centroids

from sklearn.metrics import silhouette_score
def evaluate_clusters(Data, D, cluster_range):
    results = {}
    for k in cluster_range:
        print(f"\nClustering with k={k}")
        model = K_DTW(K_Clusters=k)
        labels, _ = model.fit(Data)

        # # Compute pairwise DTW distance matrix
        # N = Data.shape[0]
        # dist_matrix = np.zeros((N, N))
        # for i in tqdm(range(N), desc=f"DTW matrix for k={k}"):
        #     for j in range(i+1, N):
        #         dist, _ = fastdtw(Data[i].reshape(-1,1), Data[j].reshape(-1,1), dist=euclidean)
        #         dist_matrix[i, j] = dist_matrix[j, i] = dist

        score = silhouette_score(D, labels, metric="precomputed")
        results[k] = score
        print(f"Silhouette Score for k={k}: {score:.4f}")

    return results


def minmax_scale(X):
    """
    Scale each time series (row) in X to [0, 1].
    
    Parameters
    ----------
    X : array, shape (n_samples, n_timestamps)
    
    Returns
    -------
    X_scaled : array, same shape as X
    """
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_scaled = (X - X_min) / (X_max - X_min)  # add eps to avoid /0
    return X_scaled



def compute_DTW_matrix(X):
    """Compute DTW distance."""

    Rows = X.shape[0]
    # Columns = X.shape(1)
    DM = np.zeros((Rows,Rows))
    for r in range(Rows):
        for c in range(r+1, Rows):
            distance, _ = fastdtw(X[r].reshape(-1, 1),
                                   X[c].reshape(-1, 1),
                                     dist=euclidean)
            DM[r,c] = DM[c,r] = distance
            # DM[c,r] = distance
    return DM



from sklearn.manifold import MDS



# =========================================
# Option 1: DTW + MDS + KMeans
# =========================================
def elbow_method_kmeans(D, max_k=15, random_state=42):
    """
    Elbow method using DTW distance matrix + MDS embedding + KMeans.
    
    Parameters:
        D (ndarray): DTW distance matrix (NxN).
        max_k (int): maximum number of clusters to test.
        random_state (int): for reproducibility.

    Returns:
        inertias (list): inertia values for each k.
    """
    # Step 1: Embed DTW distance matrix into Euclidean space
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
    X_embedded = mds.fit_transform(D)

    # Step 2: Run KMeans for different k
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X_embedded)
        inertias.append(kmeans.inertia_)

    total_gain = inertias[0] - inertias[-1]
    threshold = 0.95 * total_gain

    gains = [inertias[0] - val for val in inertias]
    optimal_k = next(i+1 for i, g in enumerate(gains) if g >= threshold)
    print("Optimal k =", optimal_k)

    # Step 3: Plot elbow curve
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method with DTW + KMeans (via MDS)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return optimal_k











