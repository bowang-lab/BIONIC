import time
import math
from pathlib import Path
from typing import Union

import typer
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.multiprocessing

from .utils.config_parser import ConfigParser
from .utils.plotter import plot_losses
from .utils.preprocessor import Preprocessor
from .utils.sampler import StatefulSampler, NeighborSamplerWithWeights
from .utils.common import extend_path
from .model.model import Bionic
from .model.loss import masked_scaled_mse


class Trainer:

    cuda = torch.cuda.is_available()
    typer.echo("Using CUDA") if cuda else typer.echo("Using CPU")

    def __init__(self, config: Union[Path, dict]):
        self.params = self._parse_config(
            config
        )  # parse configuration and load into `params` namespace
        self.writer = (
            self._init_tensorboard()
        )  # create `SummaryWriter` for tensorboard visualization
        self.index, self.masks, self.weights, self.features, self.adj = self._preprocess_inputs()
        self.train_loaders = self._make_train_loaders()
        self.inference_loaders = self._make_inference_loaders()
        self.model, self.optimizer = self._init_model()

    def _parse_config(self, config):
        cp = ConfigParser(config)
        return cp.parse()

    def _init_tensorboard(self):
        if self.params.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(flush_secs=10)
        return None

    def _preprocess_inputs(self):
        preprocessor = Preprocessor(
            self.params.names, delimiter=self.params.delimiter, svd_dim=self.params.svd_dim,
        )
        return preprocessor.process(cuda=Trainer.cuda)

    def _make_train_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[10] * self.params.gat_shapes["n_layers"],
                batch_size=self.params.batch_size,
                shuffle=False,
                sampler=StatefulSampler(torch.arange(len(self.index))),
            )
            for ad in self.adj
        ]

    def _make_inference_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[-1] * self.params.gat_shapes["n_layers"],  # all neighbors
                batch_size=1,
                shuffle=False,
                sampler=StatefulSampler(torch.arange(len(self.index))),
            )
            for ad in self.adj
        ]

    def _init_model(self):
        model = Bionic(
            len(self.index),
            self.params.gat_shapes,
            self.params.embedding_size,
            len(self.adj),
            svd_dim=self.params.svd_dim,
        )
        model.apply(self._init_model_weights)

        # Load pretrained model
        # TODO: refactor this
        if self.params.load_pretrained_model:
            typer.echo("Loading pretrained model...")
            model.load_state_dict(torch.load(f"models/{self.params.out_name}_model.pt"))

        # Push model to cuda device, if available
        if Trainer.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=self.params.learning_rate, weight_decay=0.0)

        return model, optimizer

    def _init_model_weights(self, model):
        if hasattr(model, "weight"):
            if self.params.initialization == "kaiming":
                torch.nn.init.kaiming_uniform_(model.weight, a=0.1)
            elif self.params.initialization == "xavier":
                torch.nn.init.xavier_uniform_(model.weight)
            else:
                raise ValueError(
                    f"The initialization scheme {self.params.initialization} \
                    provided is not supported"
                )

    def train(self, **kwargs):
        """Trains BIONIC model.

        Keyword arguments matching those found in `self.params` (i.e. the config arguments)
        can be passed here to overwrite the corresponding `self.params` arguments.
        """
        self.params.__dict__ = {**vars(self.params), **kwargs}  # TODO: check this works
        # TODO: overwrite and save config (maybe?), add this functionality to CLI

        # Track losses per epoch.
        train_loss = []

        best_loss = None
        best_state = None

        # Train model.
        for epoch in range(self.params.epochs):

            t = time.time()

            # Track average loss across batches.
            epoch_losses = np.zeros(len(self.adj))

            if bool(self.params.sample_size):
                rand_net_idxs = np.random.permutation(len(self.adj))
                idx_split = np.array_split(
                    rand_net_idxs, math.floor(len(self.adj) / self.params.sample_size)
                )
                for rand_idxs in idx_split:
                    _, losses = self._train_step(rand_idxs)
                    for idx, loss in zip(rand_idxs, losses):
                        epoch_losses[idx] += loss

            else:
                _, losses = self._train_step()

                epoch_losses = [
                    ep_loss + b_loss.item() / (len(self.index) / self.params.batch_size)
                    for ep_loss, b_loss in zip(epoch_losses, losses)
                ]

            # Print training progress.
            separator_string = typer.style("|", fg=typer.colors.MAGENTA, bold=True)

            def color_header_string(string: str, **kwargs):
                return typer.style(string, fg=typer.colors.CYAN, bold=True, **kwargs)

            progress_string = f"{color_header_string('Epoch')}: {epoch + 1} {separator_string} {color_header_string('Loss Total')}: {sum(epoch_losses):.6f} {separator_string}"
            if len(self.adj) <= 10:
                for i, loss in enumerate(epoch_losses):
                    progress_string += (
                        f" {color_header_string(f'Loss {i + 1}')}: {loss:.6f} {separator_string}"
                    )
            progress_string += f" {color_header_string('Time (s)')}: {time.time() - t:.4f}"
            typer.echo(progress_string)

            # Add loss data to tensorboard visualization
            if self.params.use_tensorboard:
                if len(self.adj) <= 10:
                    writer_dct = {name: loss for name, loss in zip(self.names, epoch_losses)}
                    writer_dct["Total"] = sum(epoch_losses)
                    self.writer.add_scalars("Reconstruction Errors", writer_dct, epoch)

                else:
                    self.writer.add_scalar("Total Reconstruction Error", sum(epoch_losses), epoch)

            train_loss.append(epoch_losses)

            # Store best parameter set
            if not best_loss or sum(epoch_losses) < best_loss:
                best_loss = sum(epoch_losses)
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_loss": best_loss,
                }
                best_state = state
                # torch.save(state, f'checkpoints/{self.params.out_name}_model.pt')

        if self.params.use_tensorboard:
            self.writer.close()

        self.train_loss, self.best_state = train_loss, best_state

    def _train_step(self, rand_net_idx=None):
        """Defines training behaviour.
        """

        # Get random integers for batch.
        rand_int = StatefulSampler.step(len(self.index))
        int_splits = torch.split(rand_int, self.params.batch_size)
        batch_features = self.features

        # Initialize loaders to current batch.
        if bool(self.params.sample_size):
            batch_loaders = [self.train_loaders[i] for i in rand_net_idx]
            if isinstance(self.features, list):
                batch_features = [self.features[i] for i in rand_net_idx]

            # Subset `masks` tensor.
            mask_splits = torch.split(self.masks[:, rand_net_idx][rand_int], self.params.batch_size)

        else:
            batch_loaders = self.train_loaders
            mask_splits = torch.split(self.masks[rand_int], self.params.batch_size)
            if isinstance(self.features, list):
                batch_features = self.features

        # List of losses.
        losses = [0.0 for _ in range(len(batch_loaders))]

        # Get the data flow for each input, stored in a tuple.
        for batch_masks, node_ids, *data_flows in zip(mask_splits, int_splits, *batch_loaders):

            self.optimizer.zero_grad()
            if bool(self.params.sample_size):
                training_datasets = [self.adj[i] for i in rand_net_idx]
                output, _, _, _ = self.model(
                    training_datasets,
                    data_flows,
                    batch_features,
                    batch_masks,
                    rand_net_idxs=rand_net_idx,
                )
                curr_losses = [
                    masked_scaled_mse(
                        output,
                        self.adj[i],
                        self.weights[i],
                        node_ids,
                        batch_masks[:, j],
                        cuda=Trainer.cuda,
                    )
                    for j, i in enumerate(rand_net_idx)
                ]
            else:
                training_datasets = self.adj
                output, _, _, _ = self.model(
                    training_datasets, data_flows, batch_features, batch_masks
                )
                curr_losses = [
                    masked_scaled_mse(
                        output,
                        self.adj[i],
                        self.weights[i],
                        node_ids,
                        batch_masks[:, i],
                        cuda=Trainer.cuda,
                    )
                    for i in range(len(self.adj))
                ]

            losses = [loss + curr_loss for loss, curr_loss in zip(losses, curr_losses)]
            loss_sum = sum(curr_losses)
            loss_sum.backward()

            self.optimizer.step()

        return output, losses

    def forward(self):
        # Begin inference
        self.model.load_state_dict(
            self.best_state["state_dict"]
        )  # Recover model with lowest reconstruction loss
        typer.echo(
            (
                f"Loaded best model from epoch {self.best_state['epoch']} "
                f"with loss {self.best_state['best_loss']:.6f}"
            )
        )

        self.model.eval()
        StatefulSampler.step(len(self.index), random=False)
        emb_list = []

        # Build embedding one node at a time
        for mask, idx, *data_flows in tqdm(
            zip(self.masks, self.index, *self.inference_loaders), desc="Forward pass"
        ):
            mask = mask.reshape((1, -1))
            dot, emb, _, learned_scales = self.model(
                self.adj, data_flows, self.features, mask, evaluate=True
            )
            emb_list.append(emb.detach().cpu().numpy())
        emb = np.concatenate(emb_list)
        emb_df = pd.DataFrame(emb, index=self.index)
        emb_df.to_csv(extend_path(self.params.out_name, "_features.tsv"), sep="\t")
        # emb_df.to_csv(extend_path(self.params.out_name, "_features.csv"))

        # Free memory (necessary for sequential runs)
        torch.cuda.empty_cache()

        # Create visualization of integrated features using tensorboard projector
        if self.params.use_tensorboard:
            writer.add_embedding(emb, metadata=self.index)

        # Output loss plot
        if self.params.plot_loss:
            typer.echo("Plotting loss...")
            plot_losses(
                self.train_loss, self.params.names, extend_path(self.params.out_name, "_loss.png")
            )

        # Save model
        if self.params.save_model:
            typer.echo("Saving model...")
            torch.save(self.model.state_dict(), extend_path(self.params.out_name, "_model.pt"))

        # Save internal learned network scales
        if self.params.save_network_scales:
            typer.echo("Saving network scales...")
            learned_scales = pd.DataFrame(
                learned_scales.detach().cpu().numpy(), columns=self.params.names
            ).T
            learned_scales.to_csv(
                extend_path(self.params.out_name, "_network_weights.tsv"), header=False, sep="\t"
            )

        typer.echo("Complete!")
