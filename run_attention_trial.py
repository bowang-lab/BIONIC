import os
import argparse
import random
import numpy as np
from scipy.stats import uniform


def run_bionic():
    # randomly sample hyperparameters
    epochs = int(round(uniform.rvs(3000, 6000 - 3000)))  # 3000 to 6000
    learning_rate = uniform.rvs(0.00005, 0.0005 - 0.00005)  # 0.00005 to 0.0005
    gat_dim = int(round(uniform.rvs(128, 256 - 128)))  # 128 to 256
    gat_heads = int(round(uniform.rvs(2, 5 - 2)))  # 2 to 5
    gat_layers = random.sample([1, 2], k=1)[0]  # 1 to 2
    lambda_ = uniform.rvs(0.05, 0.5 - 0.05)  # 0.05 to 0.5

    experimental_head = 0  # 0 for default linear head
    attention = 1

    print(f"BIONIC {epochs} {learning_rate} {gat_dim} {gat_heads} {gat_layers} {lambda_} {experimental_head} {attention}")

    for fold in range(5):  # 5-fold CV
        os.system(
            f"sbatch sbatch_bionic.sh {epochs} {learning_rate} {gat_dim} {gat_heads} {gat_layers} {lambda_} "
            f"{experimental_head} {attention} {fold} "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trial", required=False, type=int)
    args = parser.parse_args()

    if args.trial is None:
        trial = 128
    else:
        trial = args.trial

    np.random.seed(42 * trial + 24)
    random.seed(42 * trial + 24)

    run_bionic()
