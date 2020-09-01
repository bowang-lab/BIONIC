from typing import List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_losses: List[float], names: List[Path], plot_path: Path) -> None:
    """Plots training loss curves."""

    train_losses = np.array(train_losses).T
    n_epochs = len(train_losses[0])
    x_epochs = np.arange(n_epochs)

    total_loss = train_losses.sum(axis=0)

    _ = plt.figure(figsize=(8, 5))

    if len(train_losses) > 10:
        plt.plot(x_epochs, total_loss)
        plt.title("Total Reconstruction Error")
    else:
        for loss, name in zip(train_losses, names):
            plt.plot(x_epochs, loss, label=name.name)
        plt.plot(x_epochs, total_loss, label="Total")
        plt.title("Reconstruction Errors")
        plt.legend()

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(which="minor", axis="y")

    plt.savefig(plot_path)
