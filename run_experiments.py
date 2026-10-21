"""Reproduce the README results.

    python run_experiments.py                 # part A (heterogeneity) + part B (continual)
    python run_experiments.py --part A --rounds 30

Outputs: results/*.json, figures/*.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fedlab import continual
from fedlab.algorithms import Gossip, Server, SmallCNN, accuracy
from fedlab.data import CLASSES, label_histogram, load, make_partition

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
})
ALGOS = {"fedavg": ("FedAvg", BLUE), "fedprox": ("FedProx", ORANGE), "scaffold": ("SCAFFOLD", AQUA),
         "gossip": ("Gossip (no server)", YELLOW)}
SCHEMES = {"iid": "IID", "dir0.3": "Dirichlet alpha=0.3", "path2": "Pathological (2 classes / client)"}


def part_a(x, y, xt, yt, a, out: Path, figs: Path):
    rng = np.random.default_rng(a.seed)
    sub = rng.choice(len(y), size=a.train_size, replace=False)
    x, y = x[sub], y[sub]
    cache = out / "part_a_heterogeneity.json"   # results are saved after every run (resumable)
    results = json.loads(cache.read_text()) if cache.exists() else {}
    for scheme in a.schemes:
        parts = make_partition(y, a.clients, scheme, np.random.default_rng(a.seed))
        results.setdefault(scheme, {"label_hist": label_histogram(y, parts).tolist(), "algos": {}})
        clients = [(x[p], y[p]) for p in parts]
        for algo in a.algos:
            if algo in results[scheme]["algos"]:
                continue
            name, lr = (algo.split("@")[0], float(algo.split("@")[1])) if "@" in algo else (algo, 0.05)
            torch.manual_seed(a.seed)
            init = SmallCNN()
            runner = Gossip(init, clients, lr=lr, epochs=a.local_epochs, seed=a.seed) if name == "gossip" else \
                Server(name, init, clients, lr=lr, epochs=a.local_epochs, seed=a.seed)
            curve, t0 = [], time.time()
            for r in range(1, a.rounds + 1):
                model = runner.round()
                if r % 2 == 0 or r == a.rounds:
                    curve.append((r, accuracy(model, xt, yt)[0]))
            acc, per_class = accuracy(model, xt, yt)
            res = {"final_accuracy": acc, "per_class": per_class, "worst_class": min(per_class),
                   "curve": curve, "MB_sent": runner.bytes_sent / 1e6, "seconds": time.time() - t0}
            if name == "gossip":
                res["disagreement"] = runner.disagreement()
            results[scheme]["algos"][algo] = res
            print(f"[A] {scheme:7s} {algo:9s} acc {acc:.3f}  worst class {min(per_class):.3f}  "
                  f"{res['MB_sent']:.0f} MB  ({res['seconds']:.0f}s)", flush=True)
            if cache.exists():   # another process may have written other runs meanwhile: merge
                on_disk = json.loads(cache.read_text())
                on_disk.setdefault(scheme, results[scheme])
                on_disk[scheme]["algos"].update(results[scheme]["algos"])
                results = on_disk
            cache.write_text(json.dumps(results, indent=2))
    if all(sc in results and all(al in results[sc]["algos"] for al in ALGOS) for sc in SCHEMES):
        plot_part_a(results, figs)
    return results


def part_b(x, y, xt, yt, a, out: Path, figs: Path):
    results = {m: continual.run(m, x, y, xt, yt, n_clients=a.clients, seed=a.seed)
               for m in ("naive", "lwf", "replay")}
    (out / "part_b_continual.json").write_text(json.dumps(results, indent=2))
    plot_part_b(results, figs)
    return results


def plot_part_a(results, figs):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    for ax, (scheme, title) in zip(axes, SCHEMES.items()):
        for algo, (label, color) in ALGOS.items():
            c = np.array(results[scheme]["algos"][algo]["curve"])
            ax.plot(c[:, 0], c[:, 1], color=color, label=label, linewidth=1.6)
        if "scaffold@0.02" in results[scheme]["algos"]:   # same algorithm, smaller local learning rate
            c = np.array(results[scheme]["algos"]["scaffold@0.02"]["curve"])
            ax.plot(c[:, 0], c[:, 1], color=AQUA, linestyle="--", linewidth=1.6, label="SCAFFOLD, local lr 0.02")
        ax.set_title(title, color=INK, loc="left", fontsize=9)
        ax.set_xlabel("communication round")
    axes[0].set_ylabel("test accuracy (global model)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle("Fashion-MNIST, 10 clients: accuracy vs rounds as client data become more heterogeneous "
                 "(local lr 0.05 unless stated)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(figs / "heterogeneity_curves.png"); plt.close(fig)

    # label distribution of the clients (what "non-IID" looks like)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (scheme, title) in zip(axes, SCHEMES.items()):
        h = np.array(results[scheme]["label_hist"], dtype=float)
        ax.imshow(h / h.sum(1, keepdims=True), cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(title, color=INK, loc="left", fontsize=9)
        ax.set_xlabel("class"); ax.set_ylabel("client"); ax.grid(False)
        ax.set_xticks(range(10)); ax.set_yticks(range(h.shape[0]))
        ax.tick_params(labelsize=6)
    fig.suptitle("Share of each class in each client's data (darker = more)", color=INK, fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(figs / "client_label_distribution.png"); plt.close(fig)


def plot_part_b(results, figs):
    labels = {"naive": ("FedAvg (naive)", INK2, "--", 3.2), "lwf": ("FedAvg + LwF distillation", ORANGE, "-", 1.6),
              "replay": ("FedAvg + replay memory", BLUE, "-", 1.8)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for m, (label, color, ls, lw) in labels.items():
        mat = np.array(results[m]["acc_matrix"])
        seen_avg = [mat[t, :t + 1].mean() for t in range(len(mat))]   # equal test size per task
        axes[0].plot(range(1, 6), seen_avg, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=4,
                     label=label)
        axes[1].plot(range(1, 6), mat[:, 0], color=color, linestyle=ls, linewidth=lw, marker="o", markersize=4,
                     label=label)
    axes[0].plot(range(1, 6), [1 / (t + 1) for t in range(5)], color=INK2, linewidth=0.8, linestyle=":")
    axes[0].text(3.2, 0.08, "dotted: 1 / number of tasks seen", color=INK2, fontsize=7)
    axes[0].set_ylabel("accuracy on all classes seen so far")
    axes[0].set_title("Average accuracy (naive and LwF overlap)", color=INK, loc="left", fontsize=9)
    axes[1].set_ylabel("accuracy on task 1 (T-shirt, Trouser)")
    axes[1].set_title("Forgetting of the first task", color=INK, loc="left", fontsize=9)
    for ax in axes:
        ax.set_xticks(range(1, 6), [f"task {i}" for i in range(1, 6)])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("after learning")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Federated class-incremental learning on Fashion-MNIST (5 tasks of 2 classes)", color=INK,
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(figs / "continual_forgetting.png"); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="AB")
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    p.add_argument("--algos", nargs="+", default=list(ALGOS),
                   help="e.g. scaffold@0.02 to run SCAFFOLD with a local learning rate of 0.02")
    a = p.parse_args()
    torch.set_num_threads(1)
    out, figs = Path("results"), Path("figures")
    out.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    x, y, xt, yt = load("data")
    if "A" in a.part:
        part_a(x, y, xt, yt, a, out, figs)
    if "B" in a.part:
        part_b(x, y, xt, yt, a, out, figs)
