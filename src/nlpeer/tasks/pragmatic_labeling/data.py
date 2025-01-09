import argparse
import json
import os
from copy import copy
from os.path import join as pjoin

from datasets import DatasetDict, Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from nlpeer import DATASETS, PAPERFORMATS, PaperReviewDataset
from nlpeer.tasks import PragmaticLabelingDataset, random_split_pragmatic_labeling_dataset


class PragmaticLabelingDataModule(LightningDataModule):
    def __init__(self, benchmark_path: str,
                 dataset_type: DATASETS,
                 in_transform,
                 target_transform,
                 tokenizer,
                 data_loader_config,
                 paper_version=1,
                 paper_format=PAPERFORMATS.ITG):
        super().__init__()

        self.base_path = benchmark_path
        self.dataset_type = dataset_type
        self.paper_version = paper_version
        self.paper_format = paper_format

        self.in_transform = in_transform
        self.target_transform = target_transform
        self.tokenizer = tokenizer

        self.splits = self._load_splits_idx()
        self.full_data = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.data_loader_config = data_loader_config

    def _load_splits_idx(self):
        fp = pjoin(self.base_path, self.dataset_type.value, "splits", "prag_split.json")

        assert os.path.exists(fp) and os.path.isfile(
            fp), f"Cannot setup Pragmatic Labeling splits, as {fp} does not exist."

        with open(fp, "r") as file:
            split_data = json.load(file)

        return split_data["splits"]

    def setup(self, stage: str | None) -> None:
        '''called one each GPU separately - stage defines if we are at fit or test step'''
        full_dataset = PaperReviewDataset(self.base_path,
                                          self.dataset_type,
                                          self.paper_version,
                                          self.paper_format,
                                          hold_in_memory=True)

        sd_dataset = PragmaticLabelingDataset(full_dataset, self.in_transform, self.target_transform)

        split_ixs = []
        for s in self.splits:
            split_ixs += [[sd_dataset.ids().index(rid) for i, rid in s]]

        self.dataset = DatasetDict({
            "train": Dataset.from_dict(sd_dataset.to_dict(split_ixs[0])),
            "dev": Dataset.from_dict(sd_dataset.to_dict(split_ixs[1])),
            "test": Dataset.from_dict(sd_dataset.to_dict(split_ixs[2])),
        })

        for split in self.dataset.keys():
            print(f"Setting up {split}")
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["txt", "label"]
            )
            self.dataset[split].set_format(type="torch")

        print("Dataset initialized")

    def train_dataloader(self):
        '''returns training dataloader'''
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = True

        return DataLoader(self.dataset["train"], **dl_config)

    def val_dataloader(self):
        '''returns validation dataloader'''
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["dev"], **dl_config)

    def test_dataloader(self):
        '''returns test dataloader'''
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["test"], **dl_config)

    def predict_dataloader(self):
        '''returns test dataloader'''
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["test"], **dl_config)

    def convert_to_features(self, sample_batch, indices=None):
        labels = sample_batch["label"]
        input_raw = sample_batch["txt"]

        features = self.tokenizer(input_raw)
        features["labels"] = labels

        return features


def store_splits(dataset, splits, random_gen, out_path):
    jsplits = {
        "dataset_type": dataset.data.dataset_type.name,
        "dataset": dataset.__class__.__name__,
        "splits": [
            [[int(s), dataset.ids()[s]] for s in split] for split in splits
        ],
        "random": str(random_gen)
    }

    with open(out_path, "w+") as f:
        json.dump(jsplits, f)


def create_and_store_splits(full_data: PragmaticLabelingDataset,
                            out_dir,
                            splits: list[float],
                            random_gen: int = None):
    split_ix, split_ids = random_split_pragmatic_labeling_dataset(full_data, splits, random_gen)

    out_path = os.path.join(out_dir, "prag_split.json")
    store_splits(full_data, split_ix, random_gen, out_path)

    return out_path


def prepare_dataset_splits(benchmark_path, paper_version, splits, random_gen, datasets=None):
    if datasets is None:
        datasets = [d for d in DATASETS]

    out_files = []
    for d in datasets:
        full_dataset = PaperReviewDataset(benchmark_path,
                                          d,
                                          paper_version,
                                          PAPERFORMATS.ITG)

        sd_dataset = PragmaticLabelingDataset(full_dataset)

        out_path = os.path.join(benchmark_path, d.value, "splits")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        out_files += [create_and_store_splits(sd_dataset, out_path, splits, random_gen)]

    return out_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", required=True, help="Path to the benchmark directory")
    parser.add_argument("--paper_version", required=True, help="Which paper version", type=int)
    parser.add_argument("--random_seed", required=True, help="Random seed to generate random splits", type=int)
    parser.add_argument("--datasets", nargs="*", required=False, help="list of datasets, if applicable", type=str)

    args = parser.parse_args()

    assert set(DATASETS[d] for d in args.datasets).issubset({DATASETS.ARR22, DATASETS.F1000}), \
        "Only ARR22 and F1000 supported for stance classification and hence splitting"

    prepare_dataset_splits(args.benchmark_dir,
                           args.paper_version,
                           [0.7, 0.1, 0.2],
                           args.random_seed,
                           [DATASETS[d] for d in args.datasets])
