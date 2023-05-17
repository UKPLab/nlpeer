from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, \
    get_linear_schedule_with_warmup

# model
from nlpeer.tasks import get_optimizer, get_loss_function, review_sentence_no_context, get_class_map_pragmatics


class LitTransformerPragmaticLabelingModule(pl.LightningModule):
    def __init__(self,
                 model_type,
                 train_loss,
                 dev_loss,
                 optimizer,
                 num_labels,
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

        # use provided losses
        self.train_loss = train_loss
        self.dev_loss = dev_loss

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss, logits = out[0], out[1]

        targets = batch["labels"]

        tloss = self.train_loss(logits, targets)
        self.log("train_loss", tloss)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        out = self(**batch)
        loss, logits = out[0], out[1]

        targets = batch["labels"].to(torch.int32)
        predictions = torch.argmax(logits, axis=1).to(torch.int32)

        vloss = self.dev_loss(predictions, targets)

        self.log("val_loss", vloss)

        return vloss

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

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        out = self(**batch)
        loss, logits = out[0], out[1]

        targets = batch["labels"].to(torch.int32)
        predictions = torch.argmax(logits, axis=1).to(torch.int32)

        return {"loss": loss, "predictions": predictions, "logits": logits, "labels": targets}

    def on_predict_epoch_end(self, results):
        preds = torch.cat([x['predictions'] for x in results[0]]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in results[0]]).detach().cpu().numpy()
        logits = torch.cat([x['logits'] for x in results[0]]).detach().cpu().numpy()

        return {"predictions": preds, "labels": labels, "logits": logits}

    def get_tokenizer(self) -> Callable:
        tokenizer = AutoTokenizer.from_pretrained(self.model_type, use_fast=True)

        def tokenize(dataset):
            return tokenizer(dataset, padding="max_length", truncation=True, max_length=512)

        return tokenize

    def get_prepare_input(self) -> tuple[Callable, Callable or None]:
        input_transform = review_sentence_no_context()
        target_transform = lambda x: get_class_map_pragmatics()[x]

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

        return LitTransformerPragmaticLabelingModule(model_type=mtype,
                                                     train_loss=get_loss_function(config["train"]["train_loss"]),
                                                     dev_loss=get_loss_function(config["train"]["dev_loss"], **config["train"][
                                              "dev_loss_kwargs"] if "dev_loss_kwargs" in config["train"] else {}),
                                                     optimizer=get_optimizer(config["train"]["optimizer"]),
                                                     lr=config["train"]["learning_rate"],
                                                     eps=config["train"]["epsilon"],
                                                     num_labels=len(get_class_map_pragmatics()))
    # from disk
    elif config and "model" in config and "load_path" in config["model"]:
        return LitTransformerPragmaticLabelingModule.load_from_checkpoint(config["model"]["load_path"])
    else:
        raise ValueError("Malformed config. Requires training regime and/or a model path to load from")