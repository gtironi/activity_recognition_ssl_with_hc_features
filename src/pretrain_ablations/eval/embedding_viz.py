"""t-SNE scatter + save PNG + coords CSV."""

from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    out_png: Path,
    eval_cfg=None,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    max_points = getattr(eval_cfg, "tsne_max_points", 5000) if eval_cfg else 5000
    perplexity  = getattr(eval_cfg, "tsne_perplexity", 30)  if eval_cfg else 30
    seed        = getattr(eval_cfg, "tsne_seed", 42)         if eval_cfg else 42

    N = len(embeddings)
    if N > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, max_points, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    from sklearn.manifold import TSNE
    coords = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=seed,
        max_iter=1000,
    ).fit_transform(embeddings)

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.colormaps.get_cmap("tab20")
    label_to_name = {i: class_names[i] if i < len(class_names) else str(i)
                     for i in unique_labels}

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, ul in enumerate(unique_labels):
        mask = labels == ul
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.5,
                   color=cmap(i / max(1, len(unique_labels) - 1)),
                   label=label_to_name[ul])
    ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    ax.set_title(f"t-SNE (n={len(coords)}, perplexity={min(perplexity, len(embeddings)-1)})")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # save coords CSV
    out_csv = out_png.with_suffix(".csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label_id", "label_name"])
        for i in range(len(coords)):
            w.writerow([coords[i, 0], coords[i, 1], int(labels[i]),
                        label_to_name.get(int(labels[i]), str(labels[i]))])


def extract_embeddings(encoder, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    encoder.eval()
    embs, labs = [], []
    with torch.no_grad():
        for x, y, _ in dataloader:
            z = encoder(x.to(device)).cpu().numpy()
            embs.append(z); labs.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labs)
