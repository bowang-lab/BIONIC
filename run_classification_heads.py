import os
import argparse
import random
import numpy as np


def run_bionic():
    # best params on default BIONIC
    epochs = 3123
    learning_rate = 0.000292
    gat_dim = 131
    gat_heads = 5
    gat_layers = 2
    lambda_ = 0.47365

    for experimental_head in range(1, 9):  # classification heads from 1 to 8

        print(f"BIONIC {epochs} {learning_rate} {gat_dim} {gat_heads} {gat_layers} {lambda_} {experimental_head}")

        if not os.path.exists(f"bionic/outputs/{experimental_head}"):
            os.mkdir(f"bionic/outputs/{experimental_head}")

        for fold in range(5):  # 5-fold CV
            os.system(
                f"sbatch sbatch_bionic.sh {epochs} {learning_rate} {gat_dim} {gat_heads} {gat_layers} {lambda_} "
                f"{experimental_head} {fold} "
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
