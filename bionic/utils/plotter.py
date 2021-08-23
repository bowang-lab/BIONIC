from typing import List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

    if len(train_losses) > 10:
        ax1.plot(x_epochs, total_recon_loss)
        plt.title("Total Error")
    else:
        for loss, name in zip(train_losses[: len(net_names)], net_names):
            ax1.plot(x_epochs, loss, label=name.name)
        ax1.plot(x_epochs, total_recon_loss, label="Reconstruction Total")

        if label_names is not None:
            ax2 = ax1.twinx()
            for loss, name in zip(train_losses[len(net_names) :], label_names):
                ax2.plot(x_epochs, loss, label=name.name)
            total_cls_loss = train_losses[len(net_names) :]
            ax2.plot(x_epochs, total_cls_loss, label="Reconstruction Total")

            ax2.set_ylabel("Classification Loss")
            ax2.set_yscale("log")

            plt.title("Reconstruction + Classification Losses")

        else:
            plt.title("Reconstruction Losses")
        plt.legend()

    plt.xlabel("Epochs")
    ax1.set_ylabel("Reconstruction Loss")
    ax1.set_yscale("log")
    plt.grid(which="minor", axis="y")

    plt.savefig(plot_path)
