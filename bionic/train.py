import json
import time
import math
import warnings
from pathlib import Path
from typing import Union, List, Optional

import typer
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.multiprocessing

from .utils.config_parser import ConfigParser
from .utils.plotter import plot_losses, save_losses
from .utils.preprocessor import Preprocessor
from .utils.sampler import StatefulSampler, NeighborSamplerWithWeights
from .utils.common import extend_path, cyan, magenta, Device
from .model.model import Bionic, BionicParallel
from .model.loss import masked_scaled_mse, classification_loss


class Trainer:
    def __init__(self, config: Union[Path, dict]):
        """Defines the relevant training and forward pass logic for BIONIC.

        A model is trained by calling `train()` and the resulting gene embeddings are
        obtained by calling `forward()`.

        Args:
            config (Union[Path, dict]): Path to config file or dictionary containing config
                parameters.
        """

        typer.secho("Using CUDA", fg=typer.colors.GREEN) if Device() == "cuda" else typer.secho(
            "Using CPU", fg=typer.colors.RED
        )

        self.params = self._parse_config(
            config
        )  # parse configuration and load into `params` namespace
        self.writer = (
            self._init_tensorboard()
        )  # create `SummaryWriter` for tensorboard visualization
        (
            self.index,
            self.masks,
            self.weights,
            self.adj,
            self.labels,
            self.label_masks,
            self.class_names,
        ) = self._preprocess_inputs()

        self.train_loaders = self._make_train_loaders()
        self.inference_loaders = self._make_inference_loaders()
        self.model, self.optimizer = self._init_model()

    def _parse_config(self, config):
        cp = ConfigParser(config)
        return cp.parse()

    def _init_tensorboard(self):
        if self.params.tensorboard and (
            self.params.tensorboard["training"] or self.params.tensorboard["embedding"]
        ):
            from torch.utils.tensorboard import SummaryWriter

            typer.secho("Using TensorBoard logging", fg=typer.colors.GREEN)
            log_dir = (
                None
                if "log_dir" not in self.params.tensorboard
                else self.params.tensorboard["log_dir"]
            )
            comment = (
                ""
                if "comment" not in self.params.tensorboard
                else self.params.tensorboard["comment"]
            )
            return SummaryWriter(log_dir=log_dir, comment=comment, flush_secs=10,)
        return None

    def _preprocess_inputs(self):
        preprocessor = Preprocessor(
            self.params.net_names,
            label_names=self.params.label_names,
            delimiter=self.params.delimiter,
            svd_dim=self.params.svd_dim,
        )
        return preprocessor.process()

    def _make_train_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[self.params.neighbor_sample_size] * self.params.gat_shapes["n_layers"],
                batch_size=self.params.batch_size,
                sampler=StatefulSampler(torch.arange(len(self.index))),
                shuffle=False,
            )
            for ad in self.adj
        ]

    def _make_inference_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[-1] * self.params.gat_shapes["n_layers"],  # all neighbors
                batch_size=1,
                sampler=StatefulSampler(torch.arange(len(self.index))),
                shuffle=False,
            )
            for ad in self.adj
        ]

    def _init_model(self):

        if self.labels:
            n_classes = [label.shape[1] for label in self.labels]
        else:
            n_classes = None

        Model = BionicParallel if self.params.model_parallel else Bionic

        model = Model(
            len(self.index),
            self.params.gat_shapes,
            self.params.embedding_size,
            len(self.adj),
            svd_dim=self.params.svd_dim,
            shared_encoder=self.params.shared_encoder,
            n_classes=n_classes,
        )
        model.apply(self._init_model_weights)

        # Load pretrained model
        if self.params.pretrained_model_path:
            typer.echo("Loading pretrained model...")
            missing_keys, unexpected_keys = model.load_state_dict(
                torch.load(self.params.pretrained_model_path), strict=False
            )
            if missing_keys:
                warnings.warn(
                    "The following parameters were missing from the provided pretrained model:"
                    f"{missing_keys}"
                )
            if unexpected_keys:
                warnings.warn(
                    "The following unexpected parameters were provided in the pretrained model:"
                    f"{unexpected_keys}"
                )

        # Push model to device
        if not self.params.model_parallel:
            model.to(Device())

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

    def train(self, verbosity: Optional[int] = 1):
        """Trains BIONIC model.

        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """

        # Track losses per epoch.
        train_loss = []

        best_loss = None
        best_state = None

        # Train model.
        for epoch in range(self.params.epochs):

            time_start = time.time()

            # Track average loss across batches
            if self.labels is not None:
                epoch_losses = np.zeros(len(self.adj) + len(self.labels))
            else:
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

                    # Add classification losses if applicable
                    for idx, loss in enumerate(losses[len(rand_idxs) :]):
                        epoch_losses[len(rand_idxs) + idx] = loss

            else:
                _, losses = self._train_step()

                epoch_losses = [
                    ep_loss + b_loss.item() / (len(self.index) / self.params.batch_size)
                    for ep_loss, b_loss in zip(epoch_losses, losses)
                ]

            if verbosity:
                progress_string = self._create_progress_string(epoch, epoch_losses, time_start)
                typer.echo(progress_string)

            # Add loss data to tensorboard visualization
            if self.writer and self.params.tensorboard["training"]:

                recon_loss_dct = {}
                cls_loss_dct = {}
                for i, loss in enumerate(epoch_losses):
                    if self.labels is not None and i >= len(self.adj):
                        name = self.params.label_names[i - len(self.adj)].stem
                        cls_loss_dct[name] = loss
                    else:
                        name = self.params.net_names[i].stem
                        recon_loss_dct[name] = loss

                recon_loss_dct["Total"] = sum(recon_loss_dct.values())
                cls_loss_dct["Total"] = sum(cls_loss_dct.values())

                self.writer.add_scalars("Reconstruction Loss", recon_loss_dct, epoch)
                if cls_loss_dct:
                    self.writer.add_scalars("Classification Loss", cls_loss_dct, epoch)

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

        if self.writer:
            self.writer.close()

        self.train_loss, self.best_state = train_loss, best_state

    def _train_step(self, rand_net_idx=None):
        """Defines training behaviour."""

        # Get random integers for batch.
        rand_int = StatefulSampler.step(len(self.index))
        int_splits = torch.split(rand_int, self.params.batch_size)

        # Initialize loaders to current batch.
        if bool(self.params.sample_size):
            batch_loaders = [self.train_loaders[i] for i in rand_net_idx]

            # Subset `masks` tensor.
            mask_splits = torch.split(self.masks[:, rand_net_idx][rand_int], self.params.batch_size)

        else:
            batch_loaders = self.train_loaders
            mask_splits = torch.split(self.masks[rand_int], self.params.batch_size)

        # List of losses.
        if self.labels is not None:
            losses = [0.0 for _ in range(len(batch_loaders) + len(self.labels))]
        else:
            losses = [0.0 for _ in range(len(batch_loaders))]

        # Get the data flow for each input, stored in a tuple.
        for batch_masks, node_ids, *data_flows in zip(mask_splits, int_splits, *batch_loaders):

            self.optimizer.zero_grad()

            # Subset supervised labels and masks if provided
            if self.labels is not None:
                batch_labels = [labels[node_ids, :] for labels in self.labels]
                batch_labels_masks = [label_masks[node_ids] for label_masks in self.label_masks]

            if bool(self.params.sample_size):
                output, _, _, _, label_preds = self.model(
                    data_flows, batch_masks, rand_net_idxs=rand_net_idx,
                )
                recon_losses = [
                    masked_scaled_mse(
                        output,
                        self.adj[i],
                        self.weights[i],
                        node_ids,
                        batch_masks[:, j],
                        self.params.lambda_,
                        device="cuda:0" if self.params.model_parallel else None,
                    )
                    for j, i in enumerate(rand_net_idx)
                ]
            else:
                output, _, _, _, label_preds = self.model(data_flows, batch_masks)
                recon_losses = [
                    masked_scaled_mse(
                        output,
                        self.adj[i],
                        self.weights[i],
                        node_ids,
                        batch_masks[:, i],
                        self.params.lambda_,
                        device="cuda:0" if self.params.model_parallel else None,
                    )
                    for i in range(len(self.adj))
                ]

            if label_preds is not None:
                cls_losses = [
                    classification_loss(pred, label, label_mask, self.params.lambda_)
                    for pred, label, label_mask in zip(
                        label_preds, batch_labels, batch_labels_masks
                    )
                ]
                curr_losses = recon_losses + cls_losses
                losses = [loss + curr_loss for loss, curr_loss in zip(losses, curr_losses)]
                loss_sum = sum(curr_losses)
            else:
                losses = [loss + curr_loss for loss, curr_loss in zip(losses, recon_losses)]
                loss_sum = sum(recon_losses)

            loss_sum.backward()
            self.optimizer.step()

        return output, losses

    def _create_progress_string(
        self, epoch: int, epoch_losses: List[float], time_start: float
    ) -> str:
        """Creates a training progress string to display."""
        sep = magenta("|")

        progress_string = (
            f"{cyan('Epoch')}: {epoch + 1} {sep} "
            f"{cyan('Loss Total')}: {sum(epoch_losses):.6f} {sep} "
        )
        if len(self.adj) <= 10:
            for i, loss in enumerate(epoch_losses):
                if self.labels is not None and i >= len(self.adj):
                    progress_string += (
                        f"{cyan(f'ClsLoss {i + 1 - len(self.adj)}')}: {loss:.6f} {sep} "
                    )
                else:
                    progress_string += f"{cyan(f'Loss {i + 1}')}: {loss:.6f} {sep} "
        progress_string += f"{cyan('Time (s)')}: {time.time() - time_start:.4f}"
        return progress_string

    def forward(self, verbosity: int = 1):
        """Runs the forward pass on the trained BIONIC model.

        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """

        # Begin inference
        if self.labels is None:
            self.model.load_state_dict(
                self.best_state["state_dict"]
            )  # Recover model with lowest reconstruction loss if no classification objective
            if verbosity:
                typer.echo(
                    (
                        f"""Loaded best model from epoch {magenta(f"{self.best_state['epoch']}")} """
                        f"""with loss {magenta(f"{self.best_state['best_loss']:.6f}")}"""
                    )
                )

        self.model.eval()
        StatefulSampler.step(len(self.index), random=False)
        emb_list = []
        prediction_lists = [[] for _ in self.labels] if self.labels is not None else None

        # Build embedding one node at a time
        with typer.progressbar(
            zip(self.masks, self.index, *self.inference_loaders),
            label=f"{cyan('Forward Pass')}:",
            length=len(self.index),
        ) as progress:
            for mask, _, *data_flows in progress:
                mask = mask.reshape((1, -1))
                _, emb, _, learned_scales, label_preds = self.model(data_flows, mask, evaluate=True)
                emb_list.append(emb.detach().cpu().numpy())

                if label_preds is not None:
                    for i, pred in enumerate(label_preds):
                        prediction_lists[i].append(torch.sigmoid(pred).detach().cpu().numpy())

        emb = np.concatenate(emb_list)
        # emb_df = pd.DataFrame(emb, index=self.index)
        # emb_df.to_csv(extend_path(self.params.out_name, "_features.tsv"), sep="\t")

        # Free memory (necessary for sequential runs)
        if Device() == "cuda":
            torch.cuda.empty_cache()

        # Create visualization of integrated features using tensorboard projector
        if self.writer and self.params.tensorboard["embedding"]:
            if verbosity:
                typer.echo("Saving TensorBoard projector embedding...")
            self.writer.add_embedding(emb, metadata=self.index, tag=self.params.out_name.stem)

        # Output loss plot
        if self.params.plot_loss:
            if verbosity:
                typer.echo("Plotting loss...")
            plot_losses(
                self.train_loss,
                self.params.net_names,
                extend_path(self.params.out_name, "_loss.png"),
                self.params.label_names,
            )

        # Save losses per epoch
        if self.params.save_loss_data:
            if verbosity:
                typer.echo("Saving loss data...")
            save_losses(
                self.train_loss,
                self.params.net_names,
                extend_path(self.params.out_name, "_loss.tsv"),
                self.params.label_names,
            )

        # Save model
        if self.params.save_model:
            if verbosity:
                typer.echo("Saving model...")
            torch.save(self.model.state_dict(), extend_path(self.params.out_name, "_model.pt"))

        # Save internal learned network scales
        if self.params.save_network_scales:
            if verbosity:
                typer.echo("Saving network scales...")
            learned_scales = pd.DataFrame(
                learned_scales.detach().cpu().numpy(), columns=self.params.net_names
            ).T
            learned_scales.to_csv(
                extend_path(self.params.out_name, "_network_weights.tsv"), header=False, sep="\t"
            )

        # Save label predictions
        # pred = None
        result_labels = None
        if self.params.save_label_predictions:

            if verbosity:
                typer.echo("Saving predicted labels...")
            if self.params.label_names is None:
                warnings.warn(
                    "The `label_names` parameter was not provided so there are "
                    "no predicted labels to save."
                )
            else:
                for i, (pred, class_names) in enumerate(zip(prediction_lists, self.class_names)):
                    pred = np.concatenate(pred)
                    pred = pd.DataFrame(pred, index=self.index, columns=class_names)
                    result_labels = pred.copy()
                    # pred.to_csv(
                    #     extend_path(self.params.out_name, f"_label_set_{i+1}_predictions.tsv"),
                    #     sep="\t",
                    # )

        typer.echo(magenta("Complete!"))

        return result_labels
