import argparse
import os
import time
import uuid
from random import random

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
import wandb

from nlpeer.data import DATASETS
from nlpeer.tasks import merge_configs
from nlpeer.tasks.review_score.data import ReviewScorePredictionDataModule

OUT_PATH = None
BENCHMARK_PATH = None
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


def get_default_config():
    global BENCHMARK_PATH, DEVICE

    assert BENCHMARK_PATH is not None and os.path.exists(BENCHMARK_PATH),\
        f"Provided benchmark path is None or does not exist: {BENCHMARK_PATH}"

    return {
        "data_loader": {
            "num_workers": 8,
            "shuffle": True
        },
        "dataset": {
            "benchmark_path": BENCHMARK_PATH
        },
        "machine": {
            "device": DEVICE
        }
    }


def setup(config):
    # get module and transforms
    if config["model"]["type"] in ["roberta", "biobert", "scibert"]:
        from nlpeer.tasks.review_score.models.TransformerBased import from_config
        module = from_config(config)
    elif config["model"]["type"].startswith("baseline"):
        from nlpeer.tasks.review_score.models.baseline import from_config
        module = from_config(config)
    else:
        raise ValueError(f"The provided model type {config['model']['type']} is not supported")

    input_transform, target_transform = module.get_prepare_input(DATASETS[config["dataset"]["type"]])
    tokenizer = module.get_tokenizer()

    # load data module
    data_module = ReviewScorePredictionDataModule(benchmark_path=config["dataset"]["benchmark_path"],
                                                  dataset_type=DATASETS[config["dataset"]["type"]],
                                                  in_transform=input_transform,
                                                  target_transform=target_transform,
                                                  tokenizer=tokenizer,
                                                  filter=config["dataset"]["filter"] if "filter" in config["dataset"] else None,
                                                  data_loader_config=config["data_loader"],
                                                  paper_version=config["dataset"]["paper_version"])

    return module, data_module


def train(model, data_module, params, logger=None, debug=False):
    global OUT_PATH

    print(f"RUN = {wandb.run.name}")
    wandb.log({"config": params})

    chkp_dir = os.path.join(OUT_PATH, f"checkpoints/{params['dataset']['type']}/{params['model']['type']}")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          mode="min",
                                          dirpath=chkp_dir,
                                          filename="{epoch}-{val_loss}-" + str(uuid.uuid4()),
                                          save_top_k=1,
                                          #every_n_train_steps=10
                                          )
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=8,
                                        min_delta=0.001)

    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      limit_train_batches=0.1 if debug else 1.0,
                      #devices=1,
                      max_epochs=params["train"]["epochs"],
                      accelerator=params["machine"]["device"],
                      callbacks=[checkpoint_callback, early_stop_callback])

    # fit the model
    trainer.fit(model, data_module)

    # run final validation step
    res = trainer.validate(model.load_from_checkpoint(checkpoint_callback.best_model_path), data_module)
    print(res)

    # output best model path
    wandb.log({"best_model": checkpoint_callback.best_model_path})
    print(f"best_model = {checkpoint_callback.best_model_path}")


def run(config, debug=False, project=None):
    global OUT_PATH

    dconfig = get_default_config()
    merge_configs(config, dconfig)

    # set seed and log
    seed = int(time.time()) % 100000
    config["random_seed"] = seed
    seed_everything(seed)

    # actual training
    model, data = setup(config)
    train(model, data, config, WandbLogger(dir=os.path.join(OUT_PATH, "logs"), project=project), debug)


def main(args):
    global BENCHMARK_PATH, OUT_PATH
    BENCHMARK_PATH = args.benchmark_path
    OUT_PATH = args.store_results

    assert os.path.exists(BENCHMARK_PATH) and os.path.exists(OUT_PATH),\
        f"Benchmark or out path do not exist. Check {BENCHMARK_PATH} and {OUT_PATH} again."

    # use default config (basically for debugging)
    dconf = {
        "dataset": {
            "type": args.dataset,
            "paper_version": 1
        },
        "model": {
            "type": args.model
        },
        "train": {
            "epochs": 20,
            "train_loss": "mse",
            "dev_loss": "mse",
            "optimizer": "adam",
            "learning_rate": 2e-5 if args.lr is None else args.lr,
            "epsilon": 1e-8
        },
        "data_loader": {
            "batch_size": 16 if args.batch_size is None else args.batch_size
        }
    }
    if args.downsample:
        def drop_random(pid):
            dec = random()
            return dec <= args.downsample

        dconf["dataset"]["filter"] = drop_random

    print("Using default config..." + str(dconf))
    runs = args.repeat if args.repeat else 1

    for i in range(runs):
        run(dconf, debug= args.debug, project=args.project if args.project is not None else "RSP_train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", required=True, type=str, help="Path to the benchmark dir")
    parser.add_argument("--store_results", required=True, type=str, help="Path for logs + results")
    parser.add_argument("--dataset", required=False, choices=[d.name for d in DATASETS], help="Dataset name if no sweep provided")
    parser.add_argument("--model", required=False, type=str, help="model name if no sweep provided")
    parser.add_argument("--lr", required=False, type=float, help="learning rate (opt)")
    parser.add_argument("--downsample", required=False, type=float, help="downsample ratio, if present")
    parser.add_argument("--batch_size", required=False, type=int, help="batch size (opt)")
    parser.add_argument("--repeat", required=False, type=int, default=1, help="Number of repetitions")
    parser.add_argument("--debug", required=False, type=bool, default=False, help="Number of repetitions")

    parser.add_argument("--project", required=False, type=str, help="Project name in WANDB")

    args = parser.parse_args()

    assert args.project is not None and args.repeat is not None, "Project name required for running a wandb sweep"

    assert args.dataset is not None and args.model is not None,\
        "Dataset type required if not loading from sweep config"

    main(args)
