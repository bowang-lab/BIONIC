from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LINE_WIDTH = 1.5


def palette_gen(n_colors):
    palette = plt.get_cmap("tab10")
    curr_idx = 0
    while curr_idx < 10:
        yield palette.colors[curr_idx]
        curr_idx += 1


def plot_losses(
    train_losses: List[np.ndarray],
    net_names: List[Path],
    plot_path: Path,
    label_names: Optional[List[Path]] = None,
) -> None:
    """Plots training loss curves."""

    train_losses = np.array(train_losses).T
    n_epochs = len(train_losses[0])
    x_epochs = np.arange(n_epochs)

    total_recon_loss = train_losses[: len(net_names)].sum(axis=0)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    if label_names is not None:
        n_lines = len(train_losses) + 2
    else:
        n_lines = len(train_losses) + 1

    gen = palette_gen(n_lines)

    if n_lines > 10:
        if label_names is not None:
            ax1.plot(
                x_epochs, total_recon_loss, lw=LINE_WIDTH, c=next(gen), label="Reconstruction Total"
            )
            ax2 = ax1.twinx()
            total_cls_loss = train_losses[len(net_names) :].sum(axis=0)
            ax2.plot(
                x_epochs, total_cls_loss, label="Classification Total", lw=LINE_WIDTH, c=next(gen)
            )

            ax2.set_ylabel("Classification Loss")
            ax2.set_yscale("log")

            plt.title("Total Reconstruction + Classification Loss")
        else:
            ax1.plot(x_epochs, total_recon_loss, lw=LINE_WIDTH, c=next(gen))
            plt.title("Total Reconstruction Loss")
    else:
        for loss, name in zip(train_losses[: len(net_names)], net_names):
            ax1.plot(x_epochs, loss, label=name.name, lw=LINE_WIDTH, c=next(gen))
        ax1.plot(
            x_epochs, total_recon_loss, label="Reconstruction Total", lw=LINE_WIDTH, c=next(gen)
        )

        if label_names is not None:
            ax2 = ax1.twinx()
            for loss, name in zip(train_losses[len(net_names) :], label_names):
                ax2.plot(x_epochs, loss, label=name.name, lw=LINE_WIDTH, c=next(gen))
            total_cls_loss = train_losses[len(net_names) :].sum(axis=0)
            ax2.plot(
                x_epochs, total_cls_loss, label="Classification Total", lw=LINE_WIDTH, c=next(gen)
            )

            ax2.set_ylabel("Classification Loss")
            ax2.set_yscale("log")

            plt.title("Reconstruction + Classification Losses")

        else:
            plt.title("Reconstruction Losses")
    fig.legend()

    plt.xlabel("Epochs")
    ax1.set_ylabel("Reconstruction Loss")
    ax1.set_yscale("log")
    plt.grid(which="minor", axis="y")
    plt.tight_layout()

    plt.savefig(plot_path)


def save_losses(
    train_losses: List[np.ndarray],
    net_names: List[Path],
    save_path: Path,
    label_names: Optional[List[Path]] = None,
) -> None:
    """Saves training loss data in a .tsv file."""

    train_losses = np.array(train_losses).T
    n_epochs = len(train_losses[0])
    x_epochs = np.arange(n_epochs)

    index = net_names
    if label_names is not None:
        index += label_names
    data = pd.DataFrame(train_losses, index=net_names, columns=x_epochs).T
    data.to_csv(save_path, sep="\t")
