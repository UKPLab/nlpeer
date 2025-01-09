import argparse
import logging
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, \
    get_linear_schedule_with_warmup

# model
from nlpeer import DATASETS, DATASET_REVIEW_OVERALL_SCALES
from nlpeer.tasks import get_optimizer, get_loss_function, get_paragraph_text, get_class_map_skimming


class LitBaselineSkimmingModule(pl.LightningModule):
    def __init__(self,
                 **kwargs):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, inputs):
        return torch.Tensor([(len(i)) for i in inputs["txt"]])

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        positives = batch["positives"]
        negatives = batch["negatives"]

        batch_size = len(positives)

        total_vloss = None
        for i in range(batch_size):
            out_p = self(positives[i])
            targets_p = positives[i]["labels"].to(torch.int32)

            out_n = self(negatives[i])
            targets_n = negatives[i]["labels"].to(torch.int32)

            vloss = self.dev_loss(torch.cat((out_p, out_n), 0).flatten(), torch.cat((targets_p, targets_n), 0).flatten())
            total_vloss = total_vloss + vloss if total_vloss is not None else vloss

        self.log("val_loss", total_vloss / float(batch_size))

        return total_vloss / float(batch_size)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        positives = batch["positives"]
        negatives = batch["negatives"]

        batch_size = len(positives)

        targets, predictions, logits = [],[], []
        for i in range(batch_size):
            out_p = self(positives[i])
            targets_p = positives[i]["labels"].to(torch.int32)

            out_n = self(negatives[i])
            targets_n = negatives[i]["labels"].to(torch.int32)

            targets += [torch.concat((targets_p.flatten(), targets_n.flatten()), 0)]
            predictions += [torch.concat((torch.Tensor([(1 if p > self.thresh else 0) for p in out_p]).flatten(), torch.Tensor([(1 if p > self.thresh else 0) for p in out_n]).flatten()), 0)]
            logits += [torch.concat((torch.Tensor([[0, x] for x in out_p.flatten()]),
                                     torch.Tensor([[0, x] for x in out_n.flatten()])), 0)]

        return {"predictions": predictions, "labels": targets, "logits": logits}

    def on_predict_epoch_end(self, results):
        preds = [[y.detach().cpu().numpy() for y in x["predictions"]] for x in results[0]]
        labels = [[y.detach().cpu().numpy() for y in x["labels"]] for x in results[0]]
        logits = [[y.detach().cpu().numpy() for y in x["logits"]] for x in results[0]]

        return {"predictions": preds, "labels": labels, "logits": logits}

    def get_tokenizer(self) -> Callable:
        def tokenize(dataset):
            return {"txt": dataset}

        return tokenize

    def get_prepare_input(self) -> tuple[Callable, Callable or None]:
        input_transform = get_paragraph_text()
        target_transform = lambda x: get_class_map_skimming()[x]

        return input_transform, target_transform


def from_config(config):
    return LitBaselineSkimmingModule()