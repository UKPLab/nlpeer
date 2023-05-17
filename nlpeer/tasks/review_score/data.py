import argparse
import os
from copy import copy
from os.path import join as pjoin

import sklearn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

from nlpeer.data import filter_gte_x_reviews, load_splits_from_file, \
    store_splits_to_file, paperwise_random_split
from nlpeer import DATASETS, PAPERFORMATS, ReviewPaperDataset
from nlpeer.tasks import ReviewScorePredictionDataset


class ReviewScorePredictionDataModule(LightningDataModule):
    def __init__(self, benchmark_path: str,
                 dataset_type: DATASETS,
                 in_transform,
                 target_transform,
                 tokenizer,
                 data_loader_config,
                 paper_version=1,
                 filter=None,
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
        self.dataset = None
        self.filter = filter

        self.data_loader_config = data_loader_config

    def _load_splits_idx(self):
        fp = pjoin(self.base_path, self.dataset_type.value, "splits", "rsp_split.json")

        assert os.path.exists(fp) and os.path.isfile(fp), \
            f"Cannot setup ReviewScorePrediction splits, as {fp} does not exist."

        return load_splits_from_file(fp, self.dataset_type)

    def setup(self, stage: str | None) -> None:
        '''called one each GPU separately - stage defines if we are at fit or test step'''

        # load all data
        full_dataset = ReviewPaperDataset(self.base_path, self.dataset_type, self.paper_version, self.paper_format)

        rsp_data = ReviewScorePredictionDataset(full_dataset,
                                                transform=self.in_transform,
                                                target_transform=self.target_transform)


        # assign each review a split by its rid loaded from disk
        split_ixs = []
        for s in self.splits:
            split_ixs += [[rsp_data.ids().index(rid) for i, rid in s]]

        # discard train samples
        if self.filter:
            split_ixs[0] = [s for s in split_ixs[0] if self.filter(s)]

        print(f"{len(split_ixs[0])} Training samples loaded")

        self.dataset = DatasetDict({
            "train": Dataset.from_dict(rsp_data.to_dict(split_ixs[0])),
            "dev": Dataset.from_dict(rsp_data.to_dict(split_ixs[1])),
            "test": Dataset.from_dict(rsp_data.to_dict(split_ixs[2])),
        })

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["txt", "oscore"]
            )
            self.dataset[split].set_format(type="torch")

    def train_dataloader(self):
        """returns training dataloader"""
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = True

        return DataLoader(self.dataset["train"], **dl_config)

    def val_dataloader(self):
        """returns validation dataloader"""
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["dev"], **dl_config)

    def test_dataloader(self):
        """returns test dataloader"""
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["test"], **dl_config)

    def predict_dataloader(self):
        """returns test dataloader"""
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["test"], **dl_config)

    def convert_to_features(self, sample_batch, indices=None):
        labels = sample_batch["oscore"]
        input_raw = sample_batch["txt"]

        features = self.tokenizer(input_raw)
        features["labels"] = labels

        return features


def create_and_store_splits(full_data: ReviewPaperDataset,
                            out_dir,
                            splits: list[float],
                            random_gen: int = None):
    splits = paperwise_random_split(full_data, splits, random_gen)

    out_path = os.path.join(out_dir, "rsp_split.json")
    store_splits_to_file(full_data, splits, out_path, random_gen)

    return out_path


def prepare_dataset_splits(benchmark_path, paper_version, splits, random_gen, datasets=None):
    if datasets is None:
        datasets = [d for d in DATASETS]

    out_files = []
    for d in datasets:
        full_dataset = ReviewPaperDataset(benchmark_path,
                                          d,
                                          paper_version,
                                          PAPERFORMATS.ITG)

        # filter by >= 1 review per paper
        filter_gte_x_reviews(full_dataset, 1)

        out_path = os.path.join(benchmark_path, d.value, "splits")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        out_files += [create_and_store_splits(full_dataset, out_path, splits, random_gen)]

    return out_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", required=True, help="Path to the benchmark directory")
    parser.add_argument("--paper_version", required=True, help="Which paper version", type=int)
    parser.add_argument("--random_seed", required=True, help="Random seed to generate random splits", type=int)
    parser.add_argument("--datasets", nargs="*", required=False, help="list of datasets, if applicable", type=str)

    args = parser.parse_args()

    prepare_dataset_splits(args.benchmark_dir,
                           args.paper_version,
                           [0.7, 0.1, 0.2],
                           args.random_seed,
                           [DATASETS[d] for d in args.datasets])