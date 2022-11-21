import os
import argparse
import random
import numpy as np
from scipy.stats import uniform


def run_bionic():
    # randomly sample hyperparameters
    epochs = int(round(uniform.rvs(3000, 6000 - 3000)))  # 3000 to 6000
    learning_rate = uniform.rvs(0.0001, 0.001 - 0.0001)  # 0.0001 to 0.001
    gat_dim = int(round(uniform.rvs(128, 256 - 128)))  # 128 to 256
    gat_heads = int(round(uniform.rvs(2, 5 - 2)))  # 2 to 5
    gat_layers = random.sample([1, 2], k=1)[0]  # 1 to 2

    print(f"BIONIC {epochs} {learning_rate} {gat_dim} {gat_heads} {gat_layers}")

    for fold in range(5):  # 5-fold CV
        os.system(
            f"sbatch sbatch_bionic.sh {epochs} {learning_rate} {gat_dim} {gat_heads} {gat_layers} {fold}"
        )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--trial", required=True, type=int)
    # args = parser.parse_args()
    trial = 128

    np.random.seed(42 * trial + 24)
    random.seed(42 * trial + 24)

    run_bionic()
