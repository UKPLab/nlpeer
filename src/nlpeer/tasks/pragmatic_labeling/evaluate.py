import argparse
from collections import Counter

import simplejson
import json
import os
from copy import copy
from typing import Callable
import plotly.graph_objects as go

from os.path import join as pjoin

import numpy as np
import scipy.stats
import sklearn
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import tasks
from nlpeer import DATASETS
from nlpeer.tasks import get_class_map_pragmatics
from nlpeer.data.utils import list_files
from nlpeer.tasks.pragmatic_labeling.data import PragmaticLabelingDataModule

OUT_PATH = None
BENCHMARK_PATH = None
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


def get_default_config():
    global BENCHMARK_PATH, DEVICE

    assert BENCHMARK_PATH is not None and os.path.exists(BENCHMARK_PATH), \
        f"Provided benchmark path is None or does not exist: {BENCHMARK_PATH}"

    return {
        "data_loader": {
            "num_workers": 0,
            "shuffle": False
        },
        "dataset": {
            "benchmark_path": BENCHMARK_PATH,
            "splits_from_file": True
        },
        "machine": {
            "device": DEVICE
        }
    }


# 1. F1-score + accuracy + confusion matrix
def error(true_scores, predicted_scores, log=False):
    f1_micro = tasks.get_loss_function("F1-micro")(predicted_scores, true_scores)
    f1_macro = tasks.get_loss_function("F1-macro")(predicted_scores, true_scores)
    accuracy = sklearn.metrics.accuracy_score(predicted_scores, true_scores)
    confusion_matrix = sklearn.metrics.confusion_matrix(true_scores, predicted_scores, labels=list(get_class_map_pragmatics().values()))

    majority_score = Counter(true_scores).most_common(1)[0][0]
    majority_baseline = [majority_score for s in true_scores]

    f1_micro_baseline = sklearn.metrics.f1_score(true_scores, majority_baseline, average="micro")
    f1_macro_baseline = sklearn.metrics.f1_score(true_scores, majority_baseline, average="macro")
    accuracy_baseline = sklearn.metrics.accuracy_score(true_scores, majority_baseline)

    if log:
        wandb.log({"f1_micro": f1_micro})
        wandb.log({"f1_macro": f1_macro})
        wandb.log({"accuracy": accuracy})
        wandb.log({
            "confusion_matrix":
                       wandb.Table(data=confusion_matrix, columns=list(get_class_map_pragmatics().keys()))
            })
        wandb.log({"labels": get_class_map_pragmatics()})
        wandb.log({"baseline": {"f1_micro": f1_micro_baseline, "f1_macro": f1_macro_baseline, "accuracy": accuracy_baseline}})

    return {
        "accuracy": float(accuracy),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": str(confusion_matrix),
        "labels": get_class_map_pragmatics(),
        "baseline_f1_micro": float(f1_micro_baseline),
        "baseline_f1_macro": float(f1_macro_baseline),
        "baseline_accuracy": float(accuracy_baseline)
    }


def setup(config):
    assert "load_path" in config["model"], "Provide a 'load_path' attribute in the config.model to load a checkpoint"

    if config["model"]["type"] in ["roberta", "biobert", "scibert"]:
        from nlpeer.tasks.pragmatic_labeling.models.TransformerBased import from_config
        module = from_config(config)
    else:
        raise ValueError("The provided model type is not supported")

    input_transform, target_transform = module.get_prepare_input()
    tokenizer = module.get_tokenizer()

    # prepare data (loading splits from disk)
    data_module_main = PragmaticLabelingDataModule(benchmark_path=config["dataset"]["benchmark_path"],
                                                 dataset_type=DATASETS[config["dataset"]["type"]],
                                                 in_transform=input_transform,
                                                 target_transform=target_transform,
                                                 tokenizer=tokenizer,
                                                 data_loader_config=config["data_loader"],
                                                 paper_version=config["dataset"]["paper_version"])

    return module, data_module_main, (input_transform, target_transform, tokenizer)


def resetup(config, transforms):
    input_transform, target_transform, tokenizer = transforms

    other = DATASETS.F1000 if config["dataset"]["type"] == DATASETS.ARR22.name else DATASETS.ARR22
    data_module_secondary = PragmaticLabelingDataModule(benchmark_path=config["dataset"]["benchmark_path"],
                                                      dataset_type=other,
                                                      in_transform=input_transform,
                                                      target_transform=target_transform,
                                                      tokenizer=tokenizer,
                                                      data_loader_config=config["data_loader"],
                                                      paper_version=config["dataset"]["paper_version"])

    return data_module_secondary


def get_predictions(model, data_module, params, logger, debug=False):
    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      devices=1,
                      limit_predict_batches= 0.1 if debug else 1.0,
                      accelerator=params["machine"]["device"])

    # accuracy as a loss/performance metric is stored with device information -- discard for predictions.
    model.dev_loss = None
    model.train_loss = None
    model.to("cuda" if params["machine"]["device"] == "gpu" else params["machine"]["device"])

    return trainer.predict(model, data_module)


def run(config, debug=False):
    global OUT_PATH

    cname = f"{config['dataset']['type']}_{config['model']['type']}"

    with wandb.init(dir=os.path.join(OUT_PATH, "logs"), config=config, project=config["project"], name=cname):
        dconfig = get_default_config()
        tasks.merge_configs(config, dconfig)

        wandb_logger = WandbLogger(dir=os.path.join(OUT_PATH, "logs"), config=config)

        print(f"Starting evaluation with config {config}")

        # get model checkpoints and config paths
        model_load_path = pjoin(config["model"]["load_path"], config["model"]["type"])
        mconf_path = pjoin(model_load_path, "config.json") if os.path.exists(
            pjoin(model_load_path, "config.json")) else None
        checkpoints = [c for c in list_files(model_load_path) if os.path.basename(c) != "config.json"]

        assert len(checkpoints) > 0, f"Model load path directory lacks checkpoints: {checkpoints}"

        # load the model config
        mconf = copy(config)

        print("Evaluation...")

        accu_res = evaluation(checkpoints, config, mconf, wandb_logger, debug)
        mean_res = compute_mean_stats(accu_res)

        wandb.log({"mean_res": mean_res})

        mean_res["accu"] = accu_res

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()
            else:
                return None

        print("Storing results")

        with open(pjoin(OUT_PATH, f"stance_eval_{config['dataset']['type']}_{config['model']['type']}.json"),
                  "w+") as f:
            simplejson.dump(mean_res, f, indent=4, ignore_nan=True, default=np_encoder)


def compute_mean_stats(accu_res):
    mean_res = {}
    for s in accu_res:
        for t in s:
            if type(s[t]) != dict:
                continue

            if t not in mean_res:
                mean_res[t] = {}

            for k, v in s[t].items():
                try:
                    vf = float(v)
                except (ValueError, TypeError):
                    continue

                if k not in mean_res[t]:
                    mean_res[t][k] = []

                mean_res[t][k] += [vf]
    for t in mean_res:
        for k, v in mean_res[t].items():
            mean_res[t][k] = {"mean": float(np.mean(v)), "median": float(np.median(v)), "std": float(np.std(v))}

    return mean_res


def evaluation(checkpoints, config, model_config, wandb_logger, debug=False):
    accu_res = []
    for cpath in checkpoints:
        # define setup config
        model_config["model"]["load_path"] = cpath
        model_config["dataset"] = config["dataset"]
        model_config["data_loader"] = config["data_loader"]
        model_config["machine"] = config["machine"]

        # prep evaluation
        model, data_module_main, transforms = setup(model_config)

        predictions = get_predictions(model, data_module_main, config, wandb_logger, debug)
        labels = torch.cat([x["labels"] for x in predictions]).detach().cpu().tolist()
        predictions = torch.cat([x["predictions"] for x in predictions]).detach().cpu().tolist()

        print(predictions)
        print(labels)

        # evaluate
        eval_conf = config["evaluation"]

        res = {}
        if "error" in eval_conf and eval_conf["error"]:
            res["error"] = error(labels, predictions, True)

        if "domain_ransfer" in eval_conf and eval_conf["domain_ransfer"]:
            data_module_sec = resetup(config, transforms)

            predictions2 = get_predictions(model, data_module_sec, config, wandb_logger, debug)
            labels2 = torch.cat([x["labels"] for x in predictions2]).detach().cpu().tolist()
            predictions2 = torch.cat([x["predictions"] for x in predictions2]).detach().cpu().tolist()

            print(predictions2)
            print(labels2)

            res[f"transfer_from_{config['dataset']['type']}_to_{data_module_sec.dataset_type.name}"] = error(labels2, predictions2, True)

        accu_res += [res]
    return accu_res


def main(args):
    global BENCHMARK_PATH, OUT_PATH
    BENCHMARK_PATH = args.benchmark_dir
    OUT_PATH = args.store_results

    assert os.path.exists(BENCHMARK_PATH) and os.path.exists(OUT_PATH), \
        f"Benchmark or out path do not exist. Check {BENCHMARK_PATH} and {OUT_PATH} again."

    config = {
        "name": f"Evaluation",
        "project": args.project,
        "random_seed": {
            "value": 29491
        },
        "model": {
            "type": args.model_type,
            "load_path": pjoin(args.chkp_dir, f"{DATASETS[args.dataset].name}")
        },
        "data_loader": {
            "batch_size": 15
        },
        "dataset": {
            "type": DATASETS[args.dataset].name,
            "paper_version": args.paper_version
        },
        "evaluation": {
            "error": True,
            "domain_ransfer": True
        }
    }

    run(config, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", required=True, type=str, help="Path to the benchmark dir")
    parser.add_argument("--project", required=True, type=str, help="Project name in WANDB")
    parser.add_argument("--store_results", required=True, type=str, help="Path for logs + results")
    parser.add_argument("--debug", required=False, default=False, type=bool, help="Set true for debugging")
    parser.add_argument("--chkp_dir", required=True, type=str, help="Path to load the checkpoints from")
    parser.add_argument("--dataset", required=True, choices=[d.name for d in DATASETS], help="Name of the dataset")
    parser.add_argument("--model_type", required=True, type=str, help="Model type e.g. biobert")
    parser.add_argument("--paper_version", required=False, default=1, type=int, help="Version of the paper")

    args = parser.parse_args()
    main(args)
