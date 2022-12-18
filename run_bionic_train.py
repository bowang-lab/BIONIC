import sys

sys.path.append("path/to/BIONIC")
from bionic.train import Trainer
import argparse
import json
import os

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score


def get_fold_validation_score(train_labels, valid_labels, train_job):
    # Create multi-hot encoding
    valid_diseases = np.unique(np.concatenate(list(valid_labels.values())))
    mlb = MultiLabelBinarizer(classes=valid_diseases)  # from sklearn
    valid_labels_mh = mlb.fit_transform(valid_labels.values())
    valid_labels_mh = pd.DataFrame(valid_labels_mh, index=valid_labels.keys(), columns=mlb.classes_)

    union = train_job.index
    valid_genes = np.intersect1d(union, list(valid_labels.keys()))

    # Reindex `valid_labels_mh` to include only genes in `valid_genes`
    valid_labels = valid_labels_mh.reindex(valid_genes).fillna(0)

    # make sure validation labels and train labels have the same classes
    common_classes = np.intersect1d(train_labels.columns, valid_labels.columns)

    assert len(common_classes) > 0
    train_labels = train_labels[common_classes]
    valid_labels = valid_labels[common_classes]

    train_labels = train_labels.loc[valid_genes]

    assert train_labels.shape == valid_labels.shape

    avp = average_precision_score(valid_labels, train_labels, average="micro")

    return avp, valid_labels


def run_bionic_train(epochs, learning_rate, gat_dim, gat_heads, gat_layers, lambda_, experimental_head, attention):
    config = {
        "net_names": [
            "bionic/inputs/bioid-cell-map.txt",
            "bionic/inputs/bioplex-293t.txt",
            "bionic/inputs/bioplex-hct116.txt",
            "bionic/inputs/depmap-99-9.txt",
            "bionic/inputs/opencell.txt",
            "bionic/inputs/Hein-2015.txt",
            "bionic/inputs/Rolland-2014.txt"
        ],
        "epochs": epochs,
        "batch_size": 2048,
        "learning_rate": learning_rate,
        "gat_shapes": {
            "dimension": gat_dim,
            "n_heads": gat_heads,
            "n_layers": gat_layers,
        },
        "lambda": lambda_,
        "initialization": "xavier",
        "embedding_size": 1024,
        "save_model": False,
        "plot_loss": False,
        "sample_rate": 0,
        "sample_while_training": False,
        "load_pretrained_model": False,
        "delimiter": " ",
        "save_label_predictions": True,
        "head_type": experimental_head,
        "attention": attention,
    }

    if not os.path.exists(f"bionic/outputs/{experimental_head}"):
        os.mkdir(f"bionic/outputs/{experimental_head}")

    out_name = f"bionic/outputs/{experimental_head}/BIONIC_e{epochs}_lr{learning_rate}_d{gat_dim}_h{gat_heads}_l{gat_layers}_lmb{lambda_}"

    config["out_name"] = out_name
    config["label_names"] = ["bionic/inputs/gene_data_train.json"]

    train_job = Trainer(config)
    train_job.train()

    train_labels = train_job.forward()
    test_labels = json.load(open("bionic/inputs/test_gene_data.json", "r"))

    avp, test_labels = get_fold_validation_score(train_labels, test_labels, train_job)

    with open(f"{out_name}.txt", "w") as out_file:
        out_file.write(str(avp))

    test_labels.to_csv(f"{out_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5373)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.000308855)
    parser.add_argument("-d", "--gat_dim", type=int, default=143)
    parser.add_argument("-gh", "--gat_heads", type=int, default=2)
    parser.add_argument("-l", "--gat_layers", type=int, default=1)
    parser.add_argument("-lmb", "--lambd", type=float, default=0.404147187)
    parser.add_argument("-head", "--head", type=int, default=2)
    parser.add_argument("-att", "--attention", type=int, default=1)
    args = parser.parse_args()

    epochs, learning_rate, gat_dim, gat_heads, gat_layers, lambd, experimental_head, attention = (
        args.epochs,
        args.learning_rate,
        args.gat_dim,
        args.gat_heads,
        args.gat_layers,
        args.lambd,
        args.head,
        args.attention,
    )

    run_bionic_train(epochs, learning_rate, gat_dim, gat_heads, gat_layers, lambd, experimental_head, attention)
