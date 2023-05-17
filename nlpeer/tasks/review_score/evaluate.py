import argparse
import simplejson
import json
import os
from copy import copy
from typing import Callable
import plotly.graph_objects as go

from os.path import join as pjoin

import numpy as np
import scipy.stats
from scipy.stats import entropy
import sklearn
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import nlpeer.tasks
from nlpeer import DATASETS, PAPERFORMATS, DATASET_REVIEW_OVERALL_SCALES, ReviewPaperDataset
from nlpeer.data.utils import list_files
from nlpeer.tasks import histogram
from nlpeer.tasks.review_score import train
from nlpeer.tasks.review_score.data import ReviewScorePredictionDataModule

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


# 1. Diversity
def relative_diversity(score_range, true_scores, predicted_scores, log=False):
    true_histogram = histogram(score_range, true_scores)
    pred_histogram = histogram(score_range, predicted_scores)

    kldiv = entropy(pred_histogram[1], qk=true_histogram[1], base=2)

    if log:
        t1 = wandb.Table(data=[[bucket, cnt] for bucket, cnt in zip(true_histogram[0], true_histogram[1])],
                         columns=["score", "cnt"])
        t2 = wandb.Table(data=[[bucket, cnt] for bucket, cnt in zip(pred_histogram[0], pred_histogram[1])],
                         columns=["score", "cnt"])

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=list(true_histogram[0]), y=true_histogram[1], name="True Scores"))
        fig.add_trace(
            go.Bar(x=list(pred_histogram[0]), y=pred_histogram[1], name="Pred Scores"))

        wandb.log({f"diversity_hist": fig})

        t3 = wandb.Table(data=[[p] for p in predicted_scores],
                         columns=["scores"])
        wandb.log({f"diversity_fine_hist": wandb.plot.histogram(t3, "scores", title="Predicted score dist")})

        wandb.log({"diversity_true": t1})
        wandb.log({"diversity_model": t2})
        wandb.log({"kl_divergence": kldiv})

    return {
        "kldiv": float(kldiv),
        "true_histogram": true_histogram,
        "pred_histogram": pred_histogram,
        "true_mean": float(np.mean(true_histogram[2])),
        "true_std": float(np.std(true_histogram[2])),
        "pred_mean_rounded": float(np.mean(pred_histogram[2])),
        "pred_std_rounded": float(np.std(pred_histogram[2])),
        "pred_mean": float(np.mean(predicted_scores)),
        "pred_std": float(np.std(predicted_scores)),
    }


# 2. MRSE plus rounded classification loss
def error(score_range, true_scores, predicted_scores, log=False):
    # mrse
    mrse = sklearn.metrics.mean_squared_error(true_scores, predicted_scores)

    # r2
    r2 = sklearn.metrics.r2_score(true_scores, predicted_scores)

    # rounded classification loss
    tbucketed = [int(x) for x in histogram(score_range, true_scores)[-1]]
    pbucketed = [int(x) for x in histogram(score_range, predicted_scores)[-1]]

    print("true", tbucketed)
    print("predicted", pbucketed)

    f1_micro = sklearn.metrics.f1_score(tbucketed, pbucketed, average="micro")
    f1_macro = sklearn.metrics.f1_score(tbucketed, pbucketed, average="macro")
    # cll = train.get_loss_function("CLL")(torch.Tensor(pbucketed).reshape(), torch.Tensor(tbucketed).flatten())

    if log:
        wandb.log({"mrse": mrse})
        wandb.log({"r^2": r2})
        wandb.log({"f1_micro_rounded": f1_micro})
        wandb.log({"f1_macro_rounded": f1_macro})
        # wandb.log({"cumulative_link_loss": cll})

    return {
        "mrse": float(mrse),
        "r^2": float(r2),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro)
    }


# 3. Fairness by paper-type (long vs. short; dataset vs. method; F1000 categories)
def relative_fairness(true_scores, predicted_scores, original_dataset, index_map, paper_criterion: Callable,
                      criterion_name: str, log=False):
    crit_per_sample = np.array([paper_criterion(original_dataset[index_map[i]]) for i in range(len(true_scores))])
    crits = set(crit_per_sample)

    samples_per_crit = {c: crit_per_sample == c for c in crits}
    true_scores_per_crit = {c: np.array(true_scores)[samples_per_crit[c]] for c in crits}
    predicted_scores_per_crit = {c: np.array(predicted_scores)[samples_per_crit[c]] for c in crits}

    if len(crits) <= 1:
        print("Fairness not applicable -- only one class present!")
        wandb.log({f"fairness_{criterion_name}": "not applicable, only one class present"})
        return {}, {}

    # "human fairness"
    h, m, = {}, {}
    # h["pearson"] = scipy.stats.pearsonr(true_scores, crit_per_sample)con
    # h["spearman"] = scipy.stats.spearmanr(true_scores, crit_per_sample)
    # h["kendalltau"] = scipy.stats.kendalltau(true_scores, crit_per_sample)
    # h["point_biserial"] = scipy.stats.pointbiserialr(np.array([predicted_scores_per_crit[s] for s in crit_per_sample]), true_scores)
    annova = scipy.stats.f_oneway(*[true_scores_per_crit[c] for c in crits])
    h["human_annova"] = {"stat": float(annova.statistic), "pvalue": float(annova.pvalue)}

    # "model fairness"
    # m["pearson"] = scipy.stats.pearsonr(predicted_scores, crit_per_sample)
    # m["spearman"] = scipy.stats.spearmanr(predicted_scores, crit_per_sample)
    # m["kendalltau"] = scipy.stats.kendalltau(predicted_scores, crit_per_sample)
    # m["point_biserial"] = scipy.stats.pointbiserialr(predicted_scores, true_scores)
    annova = scipy.stats.f_oneway(*[predicted_scores_per_crit[c] for c in crits])
    m["model_annova"] = {"stat": float(annova.statistic), "pvalue": float(annova.pvalue)}

    if log:
        scores_per_crit = [("true", true_scores_per_crit), ("pred", predicted_scores_per_crit)]
        means = {}
        stds = {}
        for t, spc in scores_per_crit:
            means[t] = []
            stds[t] = []

            for crit in crits:
                means[t] += [np.mean(spc[crit])]
                stds[t] += [np.std(spc[crit])]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=list(crits), y=means["true"], error_y=dict(type="data", array=stds["true"]), name="True Scores"))
        fig.add_trace(
            go.Bar(x=list(crits), y=means["pred"], error_y=dict(type="data", array=stds["pred"]), name="Pred Scores"))

        wandb.log({f"fairness_{criterion_name}": fig})

        wandb.log({"human_fairness": h, "model_fairness": m})

    return h, m


def get_criterion(criterion_name, dataset_type):
    if criterion_name == "paper_type":
        if dataset_type == DATASETS.F1000:
            def f1000_paper_type(p):
                return p[1]["atype"]

            return f1000_paper_type
        elif dataset_type in [DATASETS.ARR22, DATASETS.ACL17, DATASETS.CONLL16]:
            def staracl_paper_type(p):
                if "bib_page_index" not in p[1]:
                    return "short"

                bindex = p[1]["bib_page_index"]
                if bindex is None or bindex < 8:
                    return "short"
                else:
                    return "long"

            return staracl_paper_type
        elif dataset_type == DATASETS.COLING20:
            def coling_paper_type(p):
                t = p[1]["type"]
                return "long" if t.lower() == "long paper" else "short"

            return coling_paper_type
    else:
        raise ValueError(f"No such criterion exists: {criterion_name}")


def setup(config):
    assert "load_path" in config["model"], "Provide a 'load_path' attribute in the config.model to load a checkpoint"

    if config["model"]["type"] in ["roberta", "biobert", "scibert"]:
        from nlpeer.tasks.review_score.models.TransformerBased import from_config
        module = from_config(config)
    elif config["model"]["type"].startswith("baseline"):
        from nlpeer.tasks.review_score.models.baseline import from_config
        module = from_config(config)
    else:
        raise ValueError("The provided model type is not supported")

    input_transform, target_transform = module.get_prepare_input(DATASETS[config["dataset"]["type"]])
    tokenizer = module.get_tokenizer()

    # prepare data (loading splits from disk)
    data_module = ReviewScorePredictionDataModule(benchmark_path=config["dataset"]["benchmark_path"],
                                                  dataset_type=DATASETS[config["dataset"]["type"]],
                                                  in_transform=input_transform,
                                                  target_transform=target_transform,
                                                  tokenizer=tokenizer,
                                                  data_loader_config=config["data_loader"],
                                                  paper_version=config["dataset"]["paper_version"])
    original_data = ReviewPaperDataset(config["dataset"]["benchmark_path"],
                                       DATASETS[config["dataset"]["type"]],
                                       config["dataset"]["paper_version"],
                                       PAPERFORMATS.ITG)

    # mapping between split indexes and original dataset (trivial in this case)
    imap = {i: i for i in range(len(original_data))}

    return module, data_module, original_data, imap


def get_predictions(model, data_module, params, logger, debug=False):
    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      limit_predict_batches=0.1 if debug else 1.0,
                      accelerator=params["machine"]["device"])

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
        print(model_load_path)
        mconf_path = pjoin(model_load_path, "config.json") if os.path.exists(
            pjoin(model_load_path, "config.json")) else None
        checkpoints = [c for c in list_files(model_load_path) if os.path.basename(c) != "config.json"]

        assert len(checkpoints) > 0, f"Model load path {model_load_path} directory lacks checkpoints: {checkpoints}"

        # load the model config
        mconf = copy(config)

        accu_res = evaluation(checkpoints, config, mconf, wandb_logger, debug)
        mean_res = compute_mean_stats(accu_res)

        wandb.log({"mean_res": mean_res})

        mean_res["accu"] = accu_res

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()
            else:
                return None

        with open(pjoin(OUT_PATH, f"rsp_eval_{config['dataset']['type']}_{config['model']['type']}.json"), "w+") as f:
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
        model, data_module, original_dataset, imap = setup(model_config)
        score_range = DATASET_REVIEW_OVERALL_SCALES[original_dataset.dataset_type][1]

        predictions = get_predictions(model, data_module, config, wandb_logger, debug)
        labels = torch.cat([x["labels"] for x in predictions]).detach().cpu().tolist()
        predictions = torch.cat([x["predictions"] for x in predictions]).detach().cpu().tolist()

        # assuming normalization of outputs
        predictions = [(i * (np.max(score_range) - np.min(score_range))) + np.min(score_range) for i in
                       predictions]

        labels = [(i * (np.max(score_range) - np.min(score_range))) + np.min(score_range) for i in
                       labels]

        # evaluate
        eval_conf = config["evaluation"]

        res = {}
        if "diversity" in eval_conf and eval_conf["diversity"]:
            res["diversity"] = relative_diversity(score_range, labels, predictions, True)
        if "error" in eval_conf and eval_conf["error"]:
            res["error"] = error(score_range, labels, predictions, True)
        if "fairness" in eval_conf and eval_conf["fairness"] is not None:
            criterion_name = eval_conf["fairness"]["criterion"]
            paper_criterion = get_criterion(criterion_name, original_dataset.dataset_type)
            res["fairness"] = relative_fairness(labels,
                                                predictions,
                                                original_dataset,
                                                imap,
                                                paper_criterion,
                                                criterion_name,
                                                True)
        res["predictions"] = predictions

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
            "load_path": pjoin(args.chkp_dir, f"{DATASETS[args.dataset].name}"),
            "normalized_output": False   #not pretty
        },
        "data_loader": {
            "batch_size": 8
        },
        "dataset": {
            "type": DATASETS[args.dataset].name,
            "paper_version": args.paper_version
        },
        "evaluation": {
            "diversity": True,
            "error": True,
            "fairness": {
                "criterion": "paper_type"
            }
        }
    }

    run(config, debug= args.debug)
    # wandb.agent(args.sweep_id.strip(), function=run, project=args.project, count=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", required=True, type=str, help="Path to the benchmark dir")
    parser.add_argument("--project", required=True, type=str, help="Project name in WANDB")
    parser.add_argument("--store_results", required=True, type=str, help="Path for logs + results")
    parser.add_argument("--chkp_dir", required=True, type=str, help="Path to load the checkpoints from")
    parser.add_argument("--dataset", required=True, choices=[d.name for d in DATASETS], help="Name of the dataset")
    parser.add_argument("--model_type", required=True, type=str, help="Model type e.g. biobert")
    parser.add_argument("--paper_version", required=False, default=1, type=int, help="Version of the paper")
    parser.add_argument("--debug", required=False, default=False, type=bool, help="Turn on debugging")

    args = parser.parse_args()
    main(args)
