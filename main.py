import os
import time
import math
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.multiprocessing

from model import Bionic
from utils.config_parser import ConfigParser
from utils.plotter import plot_losses
from utils.preprocessor import Preprocessor

from torch_geometric.utils import subgraph
from torch_geometric.data import Data, NeighborSampler


def main(config, out_name=None):
    cuda = torch.cuda.is_available()
    print("Cuda available?", cuda)

    # Parse configuration and load into `params` namespace
    cp = ConfigParser(config, out_name)
    params = cp.parse()

    # Create `SummaryWriter` for tensorboard visualization
    if params.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(flush_secs=10)

    # Preprocess input networks.
    preprocessor = Preprocessor(
        params.names,
        params.out_name,
        delimiter=params.delimiter,
        svd_dim=params.svd_dim,
    )
    index, masks, weights, features, adj = preprocessor.process(cuda=cuda)

    # Create pytorch geometric datasets.
    datasets = [
        Data(
            edge_index=ad.indices().cpu(),
            edge_attr=ad.values().reshape((-1, 1)).cuda(),
            num_nodes=len(index),
        )
        for ad in adj
    ]

    # Create dataloaders for each dataset.
    loaders = [
        NeighborSampler(
            data,
            size=0.4,
            num_hops=params.gat_shapes["n_layers"],
            batch_size=params.batch_size,
            shuffle=False,
            add_self_loops=True,
        )
        for data in datasets
    ]

    # Create model.
    model = Bionic(
        len(index),
        params.gat_shapes,
        params.embedding_size,
        len(datasets),
        svd_dim=params.svd_dim,
    )

    # Initialize model weights
    def init_weights(m):
        if hasattr(m, "weight"):
            if params.initialization == "kaiming":
                torch.nn.init.kaiming_uniform_(m.weight, a=0.1)
            elif params.initialization == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                raise Exception("The initialization scheme provided is not supported.")

    model.apply(init_weights)

    # Load pretrained model
    if params.load_pretrained_model:
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(f"models/{params.out_name}_model.pt"))

    # Push model to cuda device, if available.
    if cuda:
        model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=0.0
    )

    def masked_weighted_mse(output, target, weight, node_ids, mask):
        """Custom loss.
        """

        sub_indices, sub_values = subgraph(
            node_ids, target.indices(), edge_attr=target.values(), relabel_nodes=True
        )
        target = (
            torch.sparse.FloatTensor(sub_indices.cuda(), sub_values)
            .coalesce()
            .to_dense()
        )
        loss = weight * torch.mean(
            mask.reshape((-1, 1)) * torch.mean((output - target) ** 2, dim=-1) * mask
        )
        return loss

    def train(rand_net_idx=None):
        """Defines training behaviour.
        """

        # Get random integers for batch.
        rand_int = torch.randperm(len(index))
        int_splits = torch.split(rand_int, params.batch_size)
        batch_features = features

        # Initialize loaders to current batch.
        if bool(params.sample_size):
            rand_loaders = [loaders[i] for i in rand_net_idx]
            batch_loaders = [l(rand_int) for l in rand_loaders]
            if isinstance(features, list):
                batch_features = [features[i] for i in rand_net_idx]

            # Subset `masks` tensor.
            mask_splits = torch.split(
                masks[:, rand_net_idx][rand_int], params.batch_size
            )

        else:
            batch_loaders = [l(rand_int) for l in loaders]
            mask_splits = torch.split(masks[rand_int], params.batch_size)
            if isinstance(features, list):
                batch_features = features

        # List of losses.
        losses = [0.0 for _ in range(len(batch_loaders))]

        # Get the data flow for each input, stored in a tuple.
        for batch_masks, node_ids in zip(mask_splits, int_splits):
            data_flows = [next(batch_loader) for batch_loader in batch_loaders]

            optimizer.zero_grad()
            if bool(params.sample_size):
                training_datasets = [datasets[i] for i in rand_net_idx]
                output, _, _, _ = model(
                    training_datasets,
                    data_flows,
                    batch_features,
                    batch_masks,
                    rand_net_idxs=rand_net_idx,
                )
                curr_losses = [
                    masked_weighted_mse(
                        output, adj[i], weights[i], node_ids, batch_masks[:, j]
                    )
                    for j, i in enumerate(rand_net_idx)
                ]
            else:
                training_datasets = datasets
                output, _, _, _ = model(
                    training_datasets, data_flows, batch_features, batch_masks
                )
                curr_losses = [
                    masked_weighted_mse(
                        output, adj[i], weights[i], node_ids, batch_masks[:, i]
                    )
                    for i in range(len(adj))
                ]

            losses = [l + cl for l, cl in zip(losses, curr_losses)]
            loss_sum = sum(curr_losses)
            loss_sum.backward()

            optimizer.step()

        return output, losses

    # Track losses per epoch.
    train_loss = []

    best_loss = None
    best_state = None

    # Train model.
    for epoch in range(params.epochs):

        t = time.time()

        # Track average loss across batches.
        epoch_losses = np.zeros(len(adj))

        if bool(params.sample_size):
            rand_net_idxs = np.random.permutation(len(adj))
            idx_split = np.array_split(
                rand_net_idxs, math.floor(len(adj) / params.sample_size)
            )
            for rand_idxs in idx_split:
                _, losses = train(rand_idxs)
                for idx, loss in zip(rand_idxs, losses):
                    epoch_losses[idx] += loss

        else:
            _, losses = train()

            epoch_losses = [
                ep_loss + b_loss.item() / (len(index) / params.batch_size)
                for ep_loss, b_loss in zip(epoch_losses, losses)
            ]

        # Print training progress.
        print(
            f"Epoch: {epoch + 1} |",
            "Loss Total: {:.6f} |".format(sum(epoch_losses)),
            end=" ",
            flush=True,
        )
        if len(adj) <= 10:
            for i, loss in enumerate(epoch_losses):
                print("Loss {}: {:.6f} |".format(i + 1, loss), end=" ", flush=True)
        print("Time: {:.4f}s".format(time.time() - t), flush=True)

        # Add loss data to tensorboard visualization
        if params.use_tensorboard:
            if len(adj) <= 10:
                writer_dct = {name: loss for name, loss in zip(names, epoch_losses)}
                writer_dct["Total"] = sum(epoch_losses)
                writer.add_scalars("Reconstruction Errors", writer_dct, epoch)

            else:
                writer.add_scalar(
                    "Total Reconstruction Error", sum(epoch_losses), epoch
                )

        train_loss.append(epoch_losses)

        # Store best parameter set
        if not best_loss or sum(epoch_losses) < best_loss:
            best_loss = sum(epoch_losses)
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
            }
            best_state = state
            # torch.save(state, f'checkpoints/{params.out_name}_model.pt')

    if params.use_tensorboard:
        writer.close()

    # Output loss plot
    if params.plot_loss:
        plot_losses(
            train_loss, params.names, f"./outputs/plots/{params.out_name}_loss.png"
        )

    # Begin inference
    print("Forward pass...")
    model.load_state_dict(
        best_state["state_dict"]
    )  # Recover model with lowest reconstruction loss
    print(
        f'Loaded best model from epoch {best_state["epoch"]} '
        + f'with loss {best_state["best_loss"]}.'
    )

    # Save model
    if params.save_model:
        print("Saving model...")
        torch.save(model.state_dict(), f"./outputs/models/{params.out_name}_model.pt")

    model.eval()
    emb_list = []

    # Redefine dataloaders for each dataset for evaluation.
    loaders = [
        NeighborSampler(
            data,
            size=1.0,
            num_hops=params.gat_shapes["n_layers"],
            batch_size=1,
            shuffle=False,
            add_self_loops=True,
        )
        for data in datasets
    ]
    loaders = [loader(torch.arange(len(index))) for loader in loaders]

    # Build embedding one node at a time
    for batch_masks, idx in tqdm(zip(masks, index), desc="Forward pass"):
        batch_masks = batch_masks.reshape((1, -1))
        data_flows = [next(loader) for loader in loaders]
        dot, emb, _, learned_weights = model(
            datasets, data_flows, features, batch_masks, evaluate=True
        )
        emb_list.append(emb.detach().cpu().numpy())
    emb = np.concatenate(emb_list)
    emb_df = pd.DataFrame(emb, index=index)
    emb_df.to_csv(f"./outputs/features/{params.out_name}_features.csv")

    # Create visualization of integrated features using tensorboard projector
    if params.use_tensorboard:
        writer.add_embedding(emb, metadata=index)

    # Save internal learned network weights
    if params.save_network_weights:
        learned_weights = pd.DataFrame(
            learned_weights.detach().cpu().numpy(), columns=params.names
        ).T
        learned_weights.to_csv(
            f"./outputs/features/{params.out_name}_network_weights.csv", header=False
        )

    # Free memory (necessary for sequential runs)
    torch.cuda.empty_cache()

    print("Complete!")


if __name__ == "__main__":
    description = """Trains the BIONIC model and outputs integrated gene and protein features.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c", "--config", required=True, help="Name of config file", type=str
    )
    parser.add_argument(
        "-o", "--out_name", help="Name of BIONIC output files", type=str
    )
    args = parser.parse_args()

    main(args.config, args.out_name)
