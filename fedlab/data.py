"""Fashion-MNIST loading and non-IID partitioning across clients."""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "x_train": "train-images-idx3-ubyte.gz", "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz", "y_test": "t10k-labels-idx1-ubyte.gz",
}
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def load(data_dir: str | Path = "data"):
    """Return x_train (N,1,28,28) float32 in [0,1], y_train, x_test, y_test."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = data_dir / name
        if not path.exists():
            urllib.request.urlretrieve(URL + name, path)
        with gzip.open(path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        if key.startswith("x"):
            out[key] = (raw[16:].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0 - 0.286) / 0.353
        else:
            out[key] = raw[8:].astype(np.int64)
    return out["x_train"], out["y_train"], out["x_test"], out["y_test"]


def partition_iid(y: np.ndarray, n_clients: int, rng) -> list[np.ndarray]:
    idx = rng.permutation(len(y))
    return [np.sort(part) for part in np.array_split(idx, n_clients)]


def partition_dirichlet(y: np.ndarray, n_clients: int, alpha: float, rng, min_size: int = 10):
    """For each class, split its samples between clients with proportions ~ Dir(alpha).
    Small alpha -> each client is dominated by a few classes (strong label skew)."""
    n_classes = int(y.max()) + 1
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p)) for p in parts]


def partition_pathological(y: np.ndarray, n_clients: int, classes_per_client: int, rng):
    """Extreme non-IID (McMahan et al., 2017): sort by label, cut into shards,
    give each client `classes_per_client` shards -> each client sees ~2 classes only."""
    n_shards = n_clients * classes_per_client
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, n_shards)
    perm = rng.permutation(n_shards)
    return [np.sort(np.concatenate([shards[s] for s in perm[k::n_clients]])) for k in range(n_clients)]


def make_partition(y, n_clients: int, scheme: str, rng):
    if scheme == "iid":
        return partition_iid(y, n_clients, rng)
    if scheme.startswith("dir"):                    # e.g. "dir0.3"
        return partition_dirichlet(y, n_clients, float(scheme[3:]), rng)
    if scheme.startswith("path"):                   # e.g. "path2"
        return partition_pathological(y, n_clients, int(scheme[4:]), rng)
    raise ValueError(scheme)


def label_histogram(y, parts, n_classes: int = 10) -> np.ndarray:
    return np.stack([np.bincount(y[p], minlength=n_classes) for p in parts])
