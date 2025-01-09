import argparse
import os
from collections import Counter
from typing import Any

import numpy as np
import torch
import wandb
import pytorch_lightning as pl

from nlpeer import DATASETS, DATASET_REVIEW_OVERALL_SCALES
from nlpeer.tasks.review_score.data import load_dataset_splits


class LitBaselineModule(pl.LightningModule):
    def __init__(self, score):
        super().__init__()

        self.default_score = score

    def forward(self, inputs) -> Any:
        return {"predictions": torch.tensor([self.default_score for i in range(inputs["labels"].shape[0])]),
                "labels": inputs["labels"]}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def get_tokenizer(self):
        return lambda lot: {"nocontent": [[l] for l in lot]}

    def get_prepare_input(self, dataset_type: DATASETS):
        input_transform = lambda x: {"txt": ""}
        target_transform = None

        return input_transform, target_transform


def from_config(config):
    assert config["model"]["type"].startswith("baseline"),\
        "Loading config for a baseline model, but passed a non-baseline config!"

    # module
    model_params = config["model"]
    if "load_path" in model_params:
        with open(model_params["load_path"], "r") as file:
            score = float(file.read().strip())
    else:
        score = model_params["score"]

    module = LitBaselineModule(score)

    # use tokenizer and transforms that do nothing, but bring the data into the right format
    tokenizer = lambda lot: {"nocontent": [[l] for l in lot]}
    input_transform = lambda x: [1]
    target_transform = None

    return module


def compute_baselines(config):
    train, dev, _ = load_dataset_splits(config["dataset"]["benchmark_path"],
                                               config["dataset"]["paper_version"],
                                               config["dataset"]["type"],
                                               None,
                                               None,
                                               None)

    rev_scores = [train[i][1] for i in range(len(train))]
    rev_scores += [dev[i][1] for i in range(len(dev))]

    # avg. review score
    avg_score = np.mean(rev_scores)

    # avg. rounded review score
    scale = DATASET_REVIEW_OVERALL_SCALES[config["dataset"]["type"]][1]
    avg_rounded_score = scale[np.argmin(np.abs(scale - avg_score))]

    # majority
    maj_score = Counter(rev_scores).most_common(1)[0][0]

    return {
        "average_score": avg_score,
        "average_rounded_score": avg_rounded_score,
        "majority_score": maj_score
    }


def main(args):
    if args.dataset == "ALL":
        datasets = list(DATASETS)
    else:
        datasets = [DATASETS[args.dataset]]

    with wandb.init(project=args.project):
        for d in datasets:
            config = {
                "dataset":{
                    "benchmark_path": args.benchmark_dir,
                    "paper_version": args.paper_version,
                    "type": d
                }
            }

            baselines = compute_baselines(config)
            tbl = wandb.Table(data=[[k, v] for k,v in baselines.items()], columns=["type", "value"])
            wandb.log({f"baseline_{d.name}": tbl})

            for k, v in baselines.items():
                if not os.path.exists(os.path.join(args.out_dir, d.name)):
                    os.mkdir(os.path.join(args.out_dir, d.name))

                if not os.path.exists(os.path.join(args.out_dir, d.name, f"baseline_{k}")):
                    os.mkdir(os.path.join(args.out_dir, d.name, f"baseline_{k}"))

                with open(os.path.join(args.out_dir, d.name, f"baseline_{k}", "score.txt"), "w+") as f:
                    f.write(str(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", required=True, type=str, help="Path to the benchmark dir")
    parser.add_argument("--out_dir", required=True, type=str, help="Path to store output")
    parser.add_argument("--project", required=True, type=str, help="Project associated with the sweep")
    parser.add_argument("--dataset", required=True, choices=[d.name for d in DATASETS] + ["ALL"], help="Name of the dataset")
    parser.add_argument("--paper_version", required=False, default=1, type=int, help="Version of the paper")

    args = parser.parse_args()
    main(args)