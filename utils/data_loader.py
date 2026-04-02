"""
utils/data_loader.py
Downloads METR-LA, builds the PyG dataset, returns DataLoaders.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import requests
from tqdm import tqdm


# ── METR-LA download ─────────────────────────────────────────────────────────

# Primary: LargeST benchmark repo (direct GitHub raw links, no redirect)
# Fallback: original DCRNN repo on Google Drive (via gdown)
METR_LA_FILES = {
    "metr-la.h5": [
        "https://github.com/liuxu77/LargeST/raw/main/data/metr-la/metr-la.h5",
        "https://github.com/liyaguang/DCRNN/raw/master/data/metr-la.h5",
    ],
    "distances_la_2012.csv": [
        "https://github.com/liuxu77/LargeST/raw/main/data/metr-la/distances_la_2012.csv",
        "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/distances_la_2012.csv",
    ],
    "graph_sensor_locations.csv": [
        "https://github.com/liuxu77/LargeST/raw/main/data/metr-la/graph_sensor_locations.csv",
        "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/graph_sensor_locations.csv",
    ],
}

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Minimum expected file sizes (bytes) — guards against getting an HTML error page
MIN_SIZES = {
    "metr-la.h5":                    1_000_000,   # ~256 MB real; HTML redirect ~13 kB
    "distances_la_2012.csv":             10_000,
    "graph_sensor_locations.csv":         2_000,
}


def _download_file(url: str, dest: str) -> bool:
    """
    Download url → dest with progress bar.
    Returns True on success, False if the server returned something too small
    (i.e. an HTML redirect / error page instead of real data).
    """
    headers = {"User-Agent": "Mozilla/5.0"}  # some hosts block bare Python UA
    try:
        r = requests.get(url, stream=True, allow_redirects=True,
                         headers=headers, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        print(f"    [error] {exc}")
        return False

    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                      desc=f"    {os.path.basename(dest)}") as bar:
        for chunk in r.iter_content(1024 * 64):
            f.write(chunk)
            bar.update(len(chunk))

    # Validate size
    actual = os.path.getsize(dest)
    fname  = os.path.basename(dest)
    min_sz = MIN_SIZES.get(fname, 1_000)
    if actual < min_sz:
        print(f"    [warn] {fname} is only {actual:,} bytes — likely an HTML redirect, not real data.")
        os.remove(dest)
        return False
    return True


def download_metr_la(force: bool = False):
    """
    Download METR-LA raw files.  Tries multiple mirror URLs per file.
    If all mirrors fail, prints manual download instructions.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    all_ok = True

    for fname, urls in METR_LA_FILES.items():
        dest = os.path.join(RAW_DIR, fname)
        if os.path.exists(dest) and not force:
            print(f"  [skip] {fname} already exists ({os.path.getsize(dest):,} bytes)")
            continue

        print(f"\n  Downloading {fname} ...")
        success = False
        for url in urls:
            print(f"    Trying: {url}")
            if _download_file(url, dest):
                print(f"    OK  ({os.path.getsize(dest):,} bytes)")
                success = True
                break

        if not success:
            all_ok = False
            print(f"\n  !! Could not download {fname} automatically.")
            print(f"     Please download it manually and place it in:  data/raw/{fname}")
            if fname == "metr-la.h5":
                print("     Direct link: https://github.com/liyaguang/DCRNN/raw/master/data/metr-la.h5")
                print("     Or run:  pip install gdown && gdown 'https://drive.google.com/uc?id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC' -O data/raw/metr-la.h5")

    if all_ok:
        print("\nDownload complete.")
    else:
        print("\nSome files need manual download (see messages above).")
        print("You can still run training with synthetic data — just omit --speeds_h5 and --dist_csv flags.")


# ── Adjacency matrix ─────────────────────────────────────────────────────────

def build_adjacency(dist_csv: str, sensor_ids: list, sigma2: float = 0.1, epsilon: float = 0.5):
    """
    Gaussian kernel adjacency from pairwise distances.
    W_ij = exp(-d²/σ²) if exp(-d²/σ²) >= ε else 0
    Returns adjacency matrix [N, N] and edge_index [2, E] for PyG.
    """
    df = pd.read_csv(dist_csv)
    num_nodes = len(sensor_ids)
    dist_mx = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist_mx, 0.0)

    # build a sensor_id → index map from column order in speeds file
    # cast items to strings just in case
    sensor_ids = [str(sid).split('.')[0] if isinstance(sid, float) else str(sid) for sid in sensor_ids]
    id2idx = {sid: i for i, sid in enumerate(sensor_ids)}

    for _, row in df.iterrows():
        i = id2idx.get(str(row["from"]).split('.')[0])
        j = id2idx.get(str(row["to"]).split('.')[0])
        if i is not None and j is not None:
            dist_mx[i, j] = row["cost"]

    std = dist_mx[dist_mx != np.inf].std()
    adj = np.exp(-np.square(dist_mx / std))
    adj[adj < epsilon] = 0.0
    np.fill_diagonal(adj, 0.0)

    # convert to edge_index + edge_weight for PyG
    edge_index = np.array(np.nonzero(adj))          # [2, E]
    edge_weight = adj[edge_index[0], edge_index[1]]  # [E]

    return (
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(edge_weight, dtype=torch.float),
        adj,
    )


# ── Sliding-window dataset ────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    """
    Each sample:
      x : [T_in,  N, F]  — input window  (default 12 steps = 1 hour)
      y : [T_out, N]     — target speeds (default 12 steps)
    """

    def __init__(self, speeds_np: np.ndarray, t_in: int = 12, t_out: int = 12):
        # speeds_np : [T, N]
        scaler = StandardScaler()
        T, N = speeds_np.shape
        speeds_scaled = scaler.fit_transform(speeds_np)  # [T, N]

        self.scaler = scaler
        self.x, self.y = [], []

        for i in range(T - t_in - t_out + 1):
            x_win = speeds_scaled[i : i + t_in]       # [T_in, N]
            y_win = speeds_np[i + t_in : i + t_in + t_out]  # [T_out, N] (raw)
            self.x.append(x_win)
            self.y.append(y_win)

        # stack → [samples, T, N] → unsqueeze F dim → [samples, T, N, 1]
        self.x = torch.tensor(np.stack(self.x), dtype=torch.float).unsqueeze(-1)
        self.y = torch.tensor(np.stack(self.y), dtype=torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ── Top-level loader function ────────────────────────────────────────────────

def get_dataloaders(
    speeds_h5: str = None,
    dist_csv: str  = None,
    t_in: int      = 12,
    t_out: int     = 12,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    batch_size: int    = 32,
    num_workers: int   = 2,
):
    """
    Returns train_loader, val_loader, test_loader, edge_index, edge_weight, scaler.
    Falls back to synthetic data if files are not found (for quick testing).
    """
    if speeds_h5 and os.path.exists(speeds_h5):
        import pandas as pd
        df_speeds = pd.read_hdf(speeds_h5)
        sensor_ids = list(df_speeds.columns)
        speeds = df_speeds.values  # [T, N]
    else:
        print("[warn] speeds file not found — using synthetic data (207 nodes, 2016 steps)")
        speeds = np.random.uniform(20, 70, size=(2016, 207)).astype(np.float32)
        sensor_ids = list(range(207))

    T, N = speeds.shape
    n_train = int(T * train_ratio)
    n_val   = int(T * val_ratio)

    train_ds = TrafficDataset(speeds[:n_train],               t_in, t_out)
    val_ds   = TrafficDataset(speeds[n_train:n_train+n_val],  t_in, t_out)
    test_ds  = TrafficDataset(speeds[n_train+n_val:],         t_in, t_out)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    # graph edges
    if dist_csv and os.path.exists(dist_csv):
        edge_index, edge_weight, _ = build_adjacency(dist_csv, sensor_ids)
    else:
        # random sparse graph for synthetic testing
        src = torch.randint(0, N, (N * 4,))
        dst = torch.randint(0, N, (N * 4,))
        edge_index  = torch.stack([src, dst])
        edge_weight = torch.rand(N * 4)

    return train_loader, val_loader, test_loader, edge_index, edge_weight, train_ds.scaler