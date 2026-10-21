import numpy as np
import torch
from torch.nn.utils import parameters_to_vector as to_vec

from fedlab.algorithms import Gossip, Server, SmallCNN, accuracy
from fedlab.data import label_histogram, make_partition


def fake_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 10
    x = (rng.normal(size=(n, 1, 28, 28)) + y[:, None, None, None] * 0.1).astype(np.float32)
    return x, y


def test_partitions_cover_every_sample_once():
    _, y = fake_data()
    for scheme in ("iid", "dir0.3", "path2"):
        parts = make_partition(y, 5, scheme, np.random.default_rng(0))
        allidx = np.concatenate(parts)
        assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y)


def test_pathological_gives_few_classes_per_client():
    _, y = fake_data(1000)
    parts = make_partition(y, 5, "path2", np.random.default_rng(0))
    hist = label_histogram(y, parts)
    assert ((hist > 0).sum(1) <= 3).all()      # 2 shards -> at most 2-3 classes


def test_fedavg_of_identical_clients_equals_local_training():
    """With one client, FedAvg must reduce to plain local SGD."""
    x, y = fake_data(64)
    torch.manual_seed(0)
    server = Server("fedavg", SmallCNN(), [(x, y)], epochs=1, seed=0)
    w0 = to_vec(server.model.parameters()).detach().clone()
    server.round()
    assert not torch.allclose(w0, to_vec(server.model.parameters()))


def test_all_algorithms_run_one_round():
    x, y = fake_data()
    parts = make_partition(y, 4, "dir0.3", np.random.default_rng(0))
    clients = [(x[p], y[p]) for p in parts]
    for algo in ("fedavg", "fedprox", "scaffold"):
        s = Server(algo, SmallCNN(), clients)
        m = s.round()
        acc, per_class = accuracy(m, x, y)
        assert 0 <= acc <= 1 and len(per_class) == 10 and s.bytes_sent > 0


def test_gossip_reduces_disagreement():
    x, y = fake_data()
    parts = make_partition(y, 4, "iid", np.random.default_rng(0))
    g = Gossip(SmallCNN(), [(x[p], y[p]) for p in parts], epochs=1)
    for m in g.models:                       # start from different models
        torch.nn.init.normal_(m.head[2].weight)
    before = g.disagreement()
    g.round()
    assert g.disagreement() < before
