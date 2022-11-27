import os
import numpy as np
import argparse
import logging


def main(trials):
    for it in range(trials):
        os.system(
            f"python run_trial.py --trial {it}"
        )

    logging.info(f'The file run_trial.py was run {trials} times. ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", required=True, type=int)
    args = parser.parse_args()

    trials = args.trials

    main(trials)
