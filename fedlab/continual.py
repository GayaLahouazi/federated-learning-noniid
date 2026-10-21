"""Federated *continual* learning: new classes arrive over time.

Class-incremental scenario: 5 tasks of 2 classes each (0-1, 2-3, ...). During
task t, clients only hold data of the 2 current classes. A single 10-class
model must still recognise the classes of the previous tasks: without care,
it forgets them (catastrophic forgetting).

Mitigations compared:
    naive    FedAvg on the current data only
    replay   each client keeps a small memory of its own past samples (rehearsal)
    lwf      Learning without Forgetting (Li & Hoiem, 2017): distil the outputs of
             the previous global model on the old classes into the new one
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .algorithms import Server, SmallCNN, accuracy
from .data import partition_dirichlet

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def lwf_loss(old_model, old_classes, lam: float = 1.0, T: float = 2.0):
    def loss(model, xb, logits):
        with torch.no_grad():
            target = F.softmax(old_model(xb)[:, old_classes] / T, dim=1)
        log_p = F.log_softmax(logits[:, old_classes] / T, dim=1)
        return lam * T * T * F.kl_div(log_p, target, reduction="batchmean")
    return loss


def run(method: str, x, y, x_test, y_test, n_clients=10, rounds_per_task=8, per_task=4000,
        memory_per_class=20, alpha=0.5, seed=0, log=print):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = SmallCNN()
    memory = [(np.empty((0, 1, 28, 28), np.float32), np.empty(0, np.int64)) for _ in range(n_clients)]
    acc_matrix = np.zeros((len(TASKS), len(TASKS)))  # acc_matrix[t, j]: accuracy on task j after task t
    seen = []
    for t, classes in enumerate(TASKS):
        idx = np.where(np.isin(y, classes))[0]
        idx = rng.choice(idx, size=min(per_task, len(idx)), replace=False)
        parts = partition_dirichlet(y[idx], n_clients, alpha, rng)
        clients = []
        for k, p in enumerate(parts):
            xk, yk = x[idx[p]], y[idx[p]]
            if method == "replay" and len(memory[k][1]):
                xk, yk = np.concatenate([xk, memory[k][0]]), np.concatenate([yk, memory[k][1]])
            clients.append((xk, yk))
        extra = lwf_loss(copy.deepcopy(model).eval(), seen) if (method == "lwf" and seen) else None
        server = Server("fedavg", model, clients, seed=seed + t)
        for _ in range(rounds_per_task):
            model = server.round(extra_loss=extra)
        seen += list(classes)
        for j, cj in enumerate(TASKS[:t + 1]):
            mask = np.isin(y_test, cj)
            acc_matrix[t, j] = accuracy(model, x_test[mask], y_test[mask])[0]
        mask = np.isin(y_test, seen)
        avg = accuracy(model, x_test[mask], y_test[mask])[0]
        log(f"[{method}] after task {t} {classes}: accuracy on all seen classes {avg:.3f}")
        if method == "replay":   # store a few samples of the new classes in each client's memory
            for k, p in enumerate(parts):
                xk, yk = x[idx[p]], y[idx[p]]
                keep = np.concatenate([rng.permutation(np.where(yk == c)[0])[:memory_per_class] for c in classes])
                memory[k] = (np.concatenate([memory[k][0], xk[keep]]), np.concatenate([memory[k][1], yk[keep]]))

    final = acc_matrix[-1]
    forgetting = [acc_matrix[:, j].max() - final[j] for j in range(len(TASKS) - 1)]
    mask = np.isin(y_test, seen)
    return {
        "final_accuracy_all_classes": accuracy(model, x_test[mask], y_test[mask])[0],
        "mean_forgetting": float(np.mean(forgetting)),
        "acc_matrix": acc_matrix.tolist(),
        "memory_samples_per_client": memory_per_class * 10 if method == "replay" else 0,
    }
