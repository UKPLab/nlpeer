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
from nlpeer.data import DATASETS
from nlpeer import DATASET_REVIEW_OVERALL_SCALES
from nlpeer.tasks import get_optimizer, get_loss_function, get_paragraph_text, get_class_map_skimming


class LitTransformerSkimmingModule(pl.LightningModule):
    def __init__(self,
                 model_type,
                 train_loss,
                 dev_loss,
                 optimizer,
                 num_labels,
                 learning_mode="pointwise",
                 structure_labels=False,
                 weight_decay: float = 0.1,
                 warmup_ratio: int = 0.06,
                 **kwargs):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

        # optimizer args
        self.optimizer = optimizer

        # define model
        self.config = AutoConfig.from_pretrained(model_type, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_type, config=self.config)
        self.model_type = model_type

        self.structure_labels = structure_labels

        # use provided losses
        self.train_loss = train_loss
        self.dev_loss = dev_loss

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        positives = batch["positives"]
        negatives = batch["negatives"]

        batch_size = len(positives)

        total_tloss = None
        for i in range(batch_size):
            out_p = self(attention_mask=positives[i]["attention_mask"], input_ids=positives[i]["input_ids"],
                         labels=positives[i]["labels"])
            loss_p, logits_p = out_p[0], out_p[1]
            targets_p = positives[i]["labels"]

            out_n = self(attention_mask=negatives[i]["attention_mask"], input_ids=negatives[i]["input_ids"],
                         labels=negatives[i]["labels"])
            loss_n, logits_n = out_n[0], out_n[1]
            targets_n = negatives[i]["labels"]

            tloss = self.train_loss(torch.cat((logits_p, logits_n), 0), torch.cat((targets_p, targets_n), 0))
            total_tloss = total_tloss + tloss if total_tloss is not None else tloss

        self.log("train_loss", total_tloss)

        return total_tloss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        positives = batch["positives"]
        negatives = batch["negatives"]

        batch_size = len(positives)

        total_vloss = None
        for i in range(batch_size):
            out_p = self(attention_mask=positives[i]["attention_mask"], input_ids=positives[i]["input_ids"], labels=positives[i]["labels"])
            loss_p, logits_p = out_p[0], out_p[1]

            predictions_p = torch.argmax(logits_p, axis=1).to(torch.int32)
            targets_p = positives[i]["labels"].to(torch.int32)

            out_n = self(attention_mask=negatives[i]["attention_mask"], input_ids=negatives[i]["input_ids"], labels=negatives[i]["labels"])
            loss_n, logits_n = out_n[0], out_n[1]

            predictions_n = torch.argmax(logits_n, axis=1).to(torch.int32)
            targets_n = negatives[i]["labels"].to(torch.int32)

            vloss = self.dev_loss(torch.cat((predictions_p, predictions_n), 0).flatten(), torch.cat((targets_p, targets_n), 0).flatten())
            total_vloss = total_vloss + vloss if total_vloss is not None else vloss

        self.log("val_loss", total_vloss / float(batch_size))

        return total_vloss / float(batch_size)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        positives = batch["positives"]
        negatives = batch["negatives"]

        batch_size = len(positives)

        softmax = torch.nn.Softmax(dim=1)

        targets = []
        predictions = []
        logits = []
        for i in range(batch_size):
            out_p = self(attention_mask=positives[i]["attention_mask"], input_ids=positives[i]["input_ids"],
                         labels=positives[i]["labels"])
            loss_p, logits_p = out_p[0], out_p[1]

            predictions_p = torch.argmax(logits_p, axis=1).to(torch.int32)
            targets_p = positives[i]["labels"].to(torch.int32)

            out_n = self(attention_mask=negatives[i]["attention_mask"], input_ids=negatives[i]["input_ids"],
                         labels=negatives[i]["labels"])
            loss_n, logits_n = out_n[0], out_n[1]

            predictions_n = torch.argmax(logits_n, axis=1).to(torch.int32)
            targets_n = negatives[i]["labels"].to(torch.int32)

            targets += [torch.concat((targets_p.flatten(), targets_n.flatten()), 0)]
            predictions += [torch.concat((predictions_p.flatten(), predictions_n.flatten()), 0)]
            logits += [torch.concat((softmax(logits_p), softmax(logits_n)), 0)]

        return {"predictions": predictions, "logits": logits, "labels": targets}

    def on_predict_epoch_end(self, results):
        preds = [[y.detach().cpu().numpy() for y in x["predictions"]] for x in results[0]]
        labels = [[y.detach().cpu().numpy() for y in x["labels"]] for x in results[0]]
        logits = [[y.detach().cpu().numpy() for y in x["logits"]]for x in results[0]]

        return {"predictions": preds, "labels": labels, "logits": logits}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self.optimizer(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.eps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]


    def get_tokenizer(self) -> Callable:
        tokenizer = AutoTokenizer.from_pretrained(self.model_type, use_fast=True)

        def tokenize(dataset):
            return tokenizer(dataset, padding="max_length", truncation=True, max_length=512)

        return tokenize

    def get_prepare_input(self) -> tuple[Callable, Callable or None]:
        input_transform = get_paragraph_text(self.structure_labels)
        target_transform = lambda x: get_class_map_skimming()[x]

        return input_transform, target_transform


def from_config(config):
    # load from config
    if config and "model" in config and "train" in config and "load_path" not in config["model"]:
        model_params = config["model"]

        if model_params["type"] in ["biobert", "roberta", "scibert"]:  # aliases
            if model_params["type"] == "roberta":
                mtype = "roberta-base"
            elif model_params["type"] == "biobert":
                mtype = "dmis-lab/biobert-v1.1"
            elif model_params["type"] == "scibert":
                mtype = "allenai/scibert_scivocab_uncased"
        else:
            mtype = model_params["type"]

        return LitTransformerSkimmingModule(model_type=mtype,
                                                 train_loss=get_loss_function(config["train"]["train_loss"]),
                                                 dev_loss=get_loss_function(config["train"]["dev_loss"],
                                                                            **config["train"][
                                                                                "dev_loss_kwargs"] if "dev_loss_kwargs" in
                                                                                                      config[
                                                                                                          "train"] else {}),
                                                 optimizer=get_optimizer(config["train"]["optimizer"]),
                                                 lr=config["train"]["learning_rate"],
                                                 eps=config["train"]["epsilon"],
                                                 structure_labels=config["train"]["structure_labels"],
                                                 num_labels=len(get_class_map_skimming()))
    # from disk
    elif config and "model" in config and "load_path" in config["model"]:
        return LitTransformerSkimmingModule.load_from_checkpoint(config["model"]["load_path"])
    else:
        raise ValueError("Malformed config. Requires training regime and/or a model path to load from")
