import argparse
import simplejson
import json
import os
from copy import copy
from typing import Callable
from collections import Counter
import plotly.graph_objects as go

from os.path import join as pjoin

import numpy as np
import scipy.stats
import sklearn
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import nlpeer.tasks
from nlpeer import DATASETS
from nlpeer.tasks import get_class_map_skimming
from nlpeer.data.utils import list_files
from nlpeer.tasks.skimming.data import SkimmingDataModule
from nlpeer.tasks.stance_detection.data import StanceDetectionDataModule

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
def classification_error(true_scores, predicted_scores, log=False):
    majority_score = get_class_map_skimming()["negative"]

    f1_accu, prec_accu, rec_accu, acc_accu = [], [], [], []
    base_f1_accu, base_prec_accu, base_rec_accu, base_acc_accu = [], [], [], []
    for sample_t, sample_p in zip(true_scores, predicted_scores):
        f1_accu += [sklearn.metrics.f1_score(sample_t, sample_p, pos_label=get_class_map_skimming()["positive"])]
        prec_accu += [sklearn.metrics.precision_score(sample_t, sample_p, pos_label=get_class_map_skimming()["positive"])]
        rec_accu += [sklearn.metrics.recall_score(sample_t, sample_p, pos_label=get_class_map_skimming()["positive"])]
        acc_accu += [sklearn.metrics.accuracy_score(sample_t, sample_p)]

        majority_baseline = [majority_score for s in sample_t]
        base_f1_accu += [sklearn.metrics.f1_score(sample_t, sample_p, pos_label=get_class_map_skimming()["positive"])]
        base_prec_accu += [sklearn.metrics.precision_score(sample_t, sample_p, pos_label=get_class_map_skimming()["positive"])]
        base_rec_accu += [sklearn.metrics.recall_score(sample_t, sample_p, pos_label=get_class_map_skimming()["positive"])]
        base_acc_accu += [sklearn.metrics.accuracy_score(sample_t, majority_baseline)]

    if log:
        wandb.log({"f1": {"mean": np.mean(f1_accu), "std": np.std(f1_accu)}})
        wandb.log({"prec": {"mean": np.mean(prec_accu), "std": np.std(prec_accu)}})
        wandb.log({"rec": {"mean": np.mean(rec_accu), "std": np.std(rec_accu)}})
        wandb.log({"accuracy": {"mean": np.mean(acc_accu), "std": np.std(acc_accu)}})
        wandb.log({"labels": get_class_map_skimming()})
        wandb.log(
            {"baseline": {"f1": {"mean": np.mean(base_f1_accu), "std": np.std(base_f1_accu)},
                          "prec": {"mean": np.mean(base_prec_accu), "std": np.std(base_prec_accu)},
                          "rec": {"mean": np.mean(base_rec_accu), "std": np.std(base_rec_accu)},
                          "accuracy": {"mean": np.mean(base_acc_accu), "std": np.std(base_acc_accu)}}})

    return {
        "f1": {"mean": float(np.mean(f1_accu)), "std": float(np.std(f1_accu))},
        "prec": {"mean": float(np.mean(prec_accu)), "std": float(np.std(prec_accu))},
        "rec": {"mean": float(np.mean(rec_accu)), "std": float(np.std(rec_accu))},
        "accuracy": {"mean": float(np.mean(acc_accu)), "std": float(np.std(acc_accu))},
        "base_f1": {"mean": float(np.mean(base_f1_accu)), "std": float(np.std(base_f1_accu))},
        "base_prec": {"mean": float(np.mean(base_prec_accu)), "std": float(np.std(base_prec_accu))},
        "base_rec": {"mean": float(np.mean(base_rec_accu)), "std": float(np.std(base_rec_accu))},
        "base_accuracy": {"mean": float(np.mean(base_acc_accu)), "std": float(np.std(base_acc_accu))},
    }


def rank_measures(true_scores, estimated_scores, ks:list):
    accu = []
    auroc_accu = []
    aupr_accu = []
    mrr = []
    for true_s, estimated_s in zip(true_scores, estimated_scores):
        ranking = list(sorted(zip(true_s, estimated_s), key=lambda x: x[1], reverse=True))

        measures_at_k = {}
        for k in ks:
            if k > len(ranking):
                measures_at_k[f"precision@{k}"] = np.nan
                measures_at_k[f"recall@{k}"] = np.nan

            measures_at_k[f"precision@{k}"] = sklearn.metrics.precision_score([r[0] for r in ranking[:k]], [get_class_map_skimming()["positive"] for r in ranking[:k]], pos_label=get_class_map_skimming()["positive"])
            measures_at_k[f"recall@{k}"] = sklearn.metrics.recall_score([r[0] for r in ranking[:k]], [get_class_map_skimming()["positive"] for r in ranking[:k]], pos_label=get_class_map_skimming()["positive"])

        auroc = sklearn.metrics.roc_auc_score(true_s, estimated_s)
        auroc_accu += [auroc]

        pr_curve = sklearn.metrics.precision_recall_curve(true_s, estimated_s)
        au_pr_curve = sklearn.metrics.auc(pr_curve[1], pr_curve[0])
        aupr_accu += [au_pr_curve]

        accu += [measures_at_k]

        mrr += [1 / (1+float(next(i for i, r in enumerate(ranking) if r[0] == get_class_map_skimming()["positive"])))]

    result_at_k = {}
    mean_prec = {f"mean_precision@{k}": float(np.nanmean([m[f"precision@{k}"] for m in accu])) for k in ks}
    std_prec = {f"std_precision@{k}": float(np.nanstd([m[f"precision@{k}"] for m in accu])) for k in ks}
    result_at_k.update(mean_prec)
    result_at_k.update(std_prec)

    mean_rec = {f"mean_recall@{k}": float(np.nanmean([m[f"recall@{k}"] for m in accu])) for k in ks}
    std_rec = {f"std_recall@{k}": float(np.nanstd([m[f"recall@{k}"] for m in accu])) for k in ks}
    result_at_k.update(mean_rec)
    result_at_k.update(std_rec)

    result_auroc = {}
    result_auroc["mean_auroc"] = float(np.nanmean(auroc_accu))
    result_auroc["std_auroc"] = float(np.nanstd(auroc_accu))

    result_aupr = {}
    result_aupr["mean_aupr"] = float(np.nanmean(aupr_accu))
    result_aupr["std_aupr"] = float(np.nanstd(aupr_accu))

    result_mrr = {"mean_reciprocal_rank": float(np.mean(mrr)), "std_reciprocal_rank": float(np.std(mrr))}

    return result_at_k, result_auroc, result_aupr, result_mrr


# 2. Ranking error
def ranking_error(true_relevance, estimated_relevance, log=False):
    precision_recall, auroc, aupr, mrr = rank_measures(true_relevance, estimated_relevance, ks=[2, 3, 4, 5, 6, 8])

    random_baseline = [[float(x) for x in np.random.random(len(x)).tolist()] for x in true_relevance]
    base_pr, base_auroc, base_aupr, base_mrr = rank_measures(true_relevance, random_baseline, ks=[2, 3, 4, 5, 6, 8])

    if log:
        wandb.log({"precision_recall_at_k": precision_recall})
        wandb.log({"AUROC": auroc})
        wandb.log({"AU_PR_Curve": aupr})
        wandb.log({"MRR": mrr})

    return {
        "precision_recall_at_k": precision_recall,
        "auroc": auroc,
        "au_pr_curve": aupr,
        "mrr": mrr,
        "baseline": {
            "precision_recall_at_k": base_pr,
            "auroc": base_auroc,
            "au_pr_curve": base_aupr,
            "mrr": base_mrr,
        }
    }

def setup(config):
    assert "load_path" in config["model"], "Provide a 'load_path' attribute in the config.model to load a checkpoint"

    if config["model"]["type"] in ["roberta", "biobert", "scibert"]:
        from nlpeer.tasks.skimming.models.TransformerBased import from_config
        module = from_config(config)
    elif config["model"]["type"] in ["baseline"]:
        from nlpeer.tasks.skimming.models.LengthBaseline import from_config
        module = from_config(config)
    else:
        raise ValueError("The provided model type is not supported")

    input_transform, target_transform = module.get_prepare_input()
    tokenizer = module.get_tokenizer()

    # prepare data (loading splits from disk)
    data_module_main = SkimmingDataModule(benchmark_path=config["dataset"]["benchmark_path"],
                                          dataset_type=DATASETS[config["dataset"]["type"]],
                                          in_transform=input_transform,
                                          target_transform=target_transform,
                                          tokenizer=tokenizer,
                                          sampling_strategy="random", #todo load from params
                                          sampling_size=5, # tood load from params
                                          data_loader_config=config["data_loader"],
                                          paper_version=config["dataset"]["paper_version"])

    return module, data_module_main, (input_transform, target_transform, tokenizer)


def get_predictions(model, data_module, params, logger, debug=False):
    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      devices=1,
                      limit_predict_batches= 0.4 if debug else 1.0,
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
        if debug:
            dconfig["data_loader"]["num_workers"] = 0

        tasks.merge_configs(config, dconfig)

        wandb_logger = WandbLogger(dir=os.path.join(OUT_PATH, "logs"), config=config)

        print(f"Starting evaluation with config {config}")

        # get model checkpoints and config paths
        model_load_path = pjoin(config["model"]["load_path"], config["model"]["type"])
        mconf_path = pjoin(model_load_path, "config.json") if os.path.exists(
            pjoin(model_load_path, "config.json")) else None
        checkpoints = [c for c in list_files(model_load_path) if os.path.basename(c) != "config.json"]

        if len(checkpoints) == 0:
            print(f"WARNING: Model load path directory lacks checkpoints: {checkpoints}")

        if config["model"]["type"] in ["baseline"]:
            checkpoints = [f"baseline-{config['model']['thresh']}"]

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

        with open(pjoin(OUT_PATH, f"skimming_eval_{config['dataset']['type']}_{config['model']['type']}.json"),
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


def compute_nested_mean_stats(accu_res):
    mean_res = {}
    for s in accu_res:
        measures = [([], list(s.items()))]
        while len(measures) > 0:
            prefix, items = measures.pop()

            for key, val in items:
                if type(val) == dict:
                    measures += [(prefix + [key], list(val.items()))]
                elif type(val) not in [str, list]:
                    cur_position = mean_res
                    for p in prefix:
                        if p not in cur_position:
                            cur_position[p] = {}

                        cur_position = cur_position[p]

                    if key not in cur_position:
                        cur_position[key] = [val]
                    else:
                        cur_position[key] += [val]
                else:
                    raise ValueError(f"Not supported type {type(val)} found. Cannot aggregate.")

    pointers = [([], list(mean_res.items()))]
    while len(pointers) > 0:
        prefix, items = pointers.pop()

        for key, val in items:
            if type(val) == dict:
                pointers += [(prefix + [key], list(val.items()))]
            else:
                cur_position = mean_res
                for p in prefix:
                    if p not in cur_position:
                        cur_position[p] = {}

                    cur_position = cur_position[p]

                measurements = val[:]
                cur_position[key] = np.mean(measurements), np.median(measurements), np.std(measurements)

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

        labels, estimates, pred_labels, rankings = [], [] , [], []
        for batch in predictions:
            es = [sample.detach().cpu().tolist() for sample in batch["logits"]]

            labels += [sample.detach().cpu().tolist() for sample in batch["labels"]]
            estimates += es
            pred_labels += [sample.detach().cpu().tolist() for sample in batch["predictions"]]

            rankings += [[s[get_class_map_skimming()["positive"]] for s in e] for e in es]

        print("labels", labels)
        print("predicted", pred_labels)
        print("rankings", rankings)
        print("estimates", estimates)

        # evaluate
        eval_conf = config["evaluation"]

        res = {}
        if "error" in eval_conf and eval_conf["error"]:
            res["error"] = classification_error(labels, pred_labels, True)

        if "ranking_error" in eval_conf and eval_conf["ranking_error"]:
            res["ranking_error"] = ranking_error(labels, rankings, True)

        res["predicted_ranking"] = rankings
        res["true_labels"] = labels

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
            "value": 6295629
        },
        "model": {
            "type": args.model_type,
            "load_path": pjoin(args.chkp_dir, f"{DATASETS[args.dataset].name}")
        },
        "data_loader": {
            "batch_size": 2
        },
        "dataset": {
            "type": DATASETS[args.dataset].name,
            "paper_version": args.paper_version
        },
        "evaluation": {
            "error": True,
            "ranking_error": True,
            "domain_ransfer": False
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
