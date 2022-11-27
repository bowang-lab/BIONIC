import sys

sys.path.append("path/to/BIONIC")
from bionic.train import Trainer
import argparse
import json

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
import logging


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

    return avp


def run_bionic(epochs, learning_rate, gat_dim, gat_heads, gat_layers, lambda_, fold):
    input_train_path = ["bionic/inputs/gene_data_train_0.json", "bionic/inputs/gene_data_train_1.json",
                        "bionic/inputs/gene_data_train_2.json", "bionic/inputs/gene_data_train_3.json",
                        "bionic/inputs/gene_data_train_4.json"]

    input_valid_path = ["bionic/inputs/gene_data_valid_0.json", "bionic/inputs/gene_data_valid_1.json",
                        "bionic/inputs/gene_data_valid_2.json", "bionic/inputs/gene_data_valid_3.json",
                        "bionic/inputs/gene_data_valid_4.json"]

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
        "batch_size": 1024,
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
    }

    logging.warning('After config')

    out_name = f"bionic/outputs/BIONIC_e{epochs}_lr{learning_rate}_d{gat_dim}_h{gat_heads}_l{gat_layers}_lmb{lambda_}_fold{fold}"

    config["out_name"] = out_name
    config["label_names"] = [input_train_path[fold]]

    train_job = Trainer(config)
    logging.warning('before train job')
    train_job.train()

    train_labels = train_job.forward()
    valid_labels = json.load(open(input_valid_path[fold], "r"))

    avp = get_fold_validation_score(train_labels, valid_labels, train_job)

    with open(f"{out_name}.txt", "w") as out_file:
        out_file.write(str(avp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-d", "--gat_dim", type=int)
    parser.add_argument("-gh", "--gat_heads", type=int)
    parser.add_argument("-l", "--gat_layers", type=int)
    parser.add_argument("-lmb", "--lambd", type=float)
    parser.add_argument("-f", "--fold", type=int)
    args = parser.parse_args()

    logging.warning('args')

    epochs, learning_rate, gat_dim, gat_heads, gat_layers, lambd, fold = (
        args.epochs,
        args.learning_rate,
        args.gat_dim,
        args.gat_heads,
        args.gat_layers,
        args.lambd,
        args.fold
    )

    logging.warning('after args')

    run_bionic(epochs, learning_rate, gat_dim, gat_heads, gat_layers, lambd, fold)
