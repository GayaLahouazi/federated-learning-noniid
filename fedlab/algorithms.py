"""Federated optimisation algorithms, implemented from scratch in PyTorch.

    FedAvg   (McMahan et al., 2017)   local SGD + weighted average on the server
    FedProx  (Li et al., 2020)        + proximal term mu/2 ||w - w_global||^2 limits client drift
    SCAFFOLD (Karimireddy et al., 2020) control variates correct the client drift
    Gossip   (decentralised SGD)      no server: each client averages with its graph neighbours
                                      (a distributed *consensus* protocol)

Every function works on flat parameter vectors to keep the maths readable.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector as to_vec
from torch.nn.utils import vector_to_parameters as from_vec


class SmallCNN(nn.Module):
    """~52k parameters: small enough to simulate 10 clients x 30 rounds on a laptop CPU."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def n_bytes(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters()) * 4  # float32


def local_sgd(model, x, y, epochs, lr, batch, rng, prox_mu=0.0, w_global=None, correction=None,
              extra_loss=None):
    """Plain SGD on one client. Returns the number of SGD steps done.

    prox_mu, w_global : FedProx proximal term
    correction        : SCAFFOLD gradient correction (c - c_i), a flat vector
    extra_loss        : callable(model, xb, logits) -> loss, e.g. distillation
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    steps = 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for s in range(0, len(x), batch):
            mb = order[s:s + batch]
            xb, yb = torch.from_numpy(x[mb]), torch.from_numpy(y[mb])
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if extra_loss is not None:
                loss = loss + extra_loss(model, xb, logits)
            if prox_mu > 0:
                loss = loss + prox_mu / 2 * (to_vec(model.parameters()) - w_global).pow(2).sum()
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                if correction is not None:
                    g = to_vec([p.grad for p in model.parameters()]) + correction
                    from_vec(g, [p.grad for p in model.parameters()])
                for p in model.parameters():
                    p -= lr * p.grad
            steps += 1
    return steps


class Server:
    """Runs FedAvg / FedProx / SCAFFOLD rounds over a list of client datasets."""

    def __init__(self, algo: str, model: nn.Module, clients: list[tuple[np.ndarray, np.ndarray]],
                 lr: float = 0.05, epochs: int = 1, batch: int = 32, mu: float = 0.01, seed: int = 0):
        assert algo in ("fedavg", "fedprox", "scaffold")
        self.algo, self.model, self.clients = algo, model, clients
        self.lr, self.epochs, self.batch, self.mu = lr, epochs, batch, mu
        self.rng = np.random.default_rng(seed)
        self.sizes = np.array([len(c[1]) for c in clients], dtype=float)
        dim = to_vec(model.parameters()).numel()
        self.c_global = torch.zeros(dim)
        self.c_local = [torch.zeros(dim) for _ in clients]
        self.bytes_sent = 0

    def round(self, extra_loss=None):
        w_g = to_vec(self.model.parameters()).detach().clone()
        new_ws, delta_cs = [], []
        for k, (x, y) in enumerate(self.clients):
            local = copy.deepcopy(self.model)
            corr = (self.c_global - self.c_local[k]) if self.algo == "scaffold" else None
            steps = local_sgd(local, x, y, self.epochs, self.lr, self.batch, self.rng,
                              prox_mu=self.mu if self.algo == "fedprox" else 0.0, w_global=w_g,
                              correction=corr, extra_loss=extra_loss)
            w_k = to_vec(local.parameters()).detach()
            new_ws.append(w_k)
            if self.algo == "scaffold":   # option II of the paper
                c_new = self.c_local[k] - self.c_global + (w_g - w_k) / (steps * self.lr)
                delta_cs.append(c_new - self.c_local[k])
                self.c_local[k] = c_new
        weights = torch.tensor(self.sizes / self.sizes.sum(), dtype=torch.float32)
        w_new = (torch.stack(new_ws) * weights[:, None]).sum(0)
        from_vec(w_new, self.model.parameters())
        if self.algo == "scaffold":
            self.c_global = self.c_global + torch.stack(delta_cs).mean(0)
        per_client = n_bytes(self.model) * (2 if self.algo == "scaffold" else 1)  # model (+ control variate)
        self.bytes_sent += 2 * per_client * len(self.clients)                     # download + upload
        return self.model


class Gossip:
    """Decentralised learning: client k keeps its own model, trains locally, then
    replaces it by the average of itself and its neighbours (ring graph).
    Repeated averaging = a consensus protocol: models converge to a common value."""

    def __init__(self, model: nn.Module, clients, lr=0.05, epochs=1, batch=32, degree: int = 2, seed=0):
        self.models = [copy.deepcopy(model) for _ in clients]
        self.clients, self.lr, self.epochs, self.batch = clients, lr, epochs, batch
        self.rng = np.random.default_rng(seed)
        n = len(clients)
        self.neighbours = [[(k + d) % n for d in range(-(degree // 2), degree // 2 + 1)] for k in range(n)]
        self.bytes_sent = 0
        self.template = copy.deepcopy(model)

    def round(self, extra_loss=None):
        for m, (x, y) in zip(self.models, self.clients):
            local_sgd(m, x, y, self.epochs, self.lr, self.batch, self.rng, extra_loss=extra_loss)
        ws = [to_vec(m.parameters()).detach().clone() for m in self.models]
        for k, m in enumerate(self.models):
            from_vec(torch.stack([ws[j] for j in self.neighbours[k]]).mean(0), m.parameters())
        self.bytes_sent += n_bytes(self.template) * sum(len(nb) - 1 for nb in self.neighbours)
        return self.consensus_model()

    def consensus_model(self):
        w = torch.stack([to_vec(m.parameters()).detach() for m in self.models]).mean(0)
        from_vec(w, self.template.parameters())
        return self.template

    def disagreement(self) -> float:
        """Mean distance of the client models to their average (0 = perfect consensus)."""
        ws = torch.stack([to_vec(m.parameters()).detach() for m in self.models])
        return float((ws - ws.mean(0)).norm(dim=1).mean())


@torch.no_grad()
def accuracy(model, x, y, batch: int = 2000):
    model.eval()
    preds = torch.cat([model(torch.from_numpy(x[i:i + batch])).argmax(1) for i in range(0, len(x), batch)])
    y = torch.from_numpy(y)
    acc = float((preds == y).float().mean())
    per_class = [float((preds[y == c] == c).float().mean()) for c in range(int(y.max()) + 1)]
    return acc, per_class
