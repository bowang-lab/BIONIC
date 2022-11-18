import sys

sys.path.append("path/to/BIONIC")
from train import Trainer
import argparse


def run_bionic(epochs, learning_rate, gat_dim, gat_heads, gat_layers, trial):

    input_train_path = ["bionic/inputs/gene_data_train_0.json", "bionic/inputs/gene_data_train_1.json",
                        "bionic/inputs/gene_data_train_2.json", "bionic/inputs/gene_data_train_3.json",
                        "bionic/inputs/gene_data_train_4.json"]

    config = {
        "in_path": "path/to/input/networks",
        "net_names": ["bionic/inputs/Hein-2015.txt",
                      "bionic/inputs/Huttlin-2015.txt",
                      "bionic/inputs/Huttlin-2017.txt",
                      "bionic/inputs/Rolland-2014.txt"],
        "epochs": epochs,
        "batch_size": 1024,
        "learning_rate": learning_rate,
        "gat_shapes": {
            "dimension": gat_dim,
            "n_heads": gat_heads,
            "n_layers": gat_layers,
        },
        "initialization": "xavier",
        "embedding_size": 1024,
        "save_model": False,
        "plot_loss": False,
        "sample_rate": 0,
        "sample_while_training": False,
        "load_pretrained_model": False,
        "delimiter": " ",
        "save_label_predictions": True,
    }

    out_name = f"path/to/output/directory/BIONIC_e{epochs}_lr{learning_rate}_d{gat_dim}_h{gat_heads}_l{gat_layers}_trial{trial}"

    config["out_name"] = out_name

    config["in_path"] = input_train_path[trial]

    train_job = Trainer(config)

    train_job.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trial", type=int)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-d", "--gat_dim", type=int)
    parser.add_argument("-gh", "--gat_heads", type=int)
    parser.add_argument("-l", "--gat_layers", type=int)
    args = parser.parse_args()

    epochs, learning_rate, gat_dim, gat_heads, gat_layers, trial = (
        args.epochs,
        args.learning_rate,
        args.gat_dim,
        args.gat_heads,
        args.gat_layers,
        args.trial,
    )

    run_bionic(epochs, learning_rate, gat_dim, gat_heads, gat_layers, trial)
