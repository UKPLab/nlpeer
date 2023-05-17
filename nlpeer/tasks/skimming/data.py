import argparse
import json
import os
from collections import Counter
from copy import copy
from os.path import join as pjoin

from datasets import DatasetDict, Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from nlpeer.data import paperwise_stratified_split
from nlpeer import DATASETS, PAPERFORMATS, PaperReviewDataset
from nlpeer.tasks import SkimmingDataset


class SkimmingDataModule(LightningDataModule):
    def __init__(self, benchmark_path: str,
                 dataset_type: DATASETS,
                 in_transform,
                 target_transform,
                 tokenizer,
                 data_loader_config,
                 sampling_strategy: str = "random",
                 sampling_size: int = 5,
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

        self.sampling_strategy = sampling_strategy
        self.sampling_size = sampling_size

        self.splits = self._load_splits_idx()
        self.full_data = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.data_loader_config = data_loader_config

    def _load_splits_idx(self):
        fp = pjoin(self.base_path, self.dataset_type.value, "splits", "skimming_split.json")

        assert os.path.exists(fp) and os.path.isfile(
            fp), f"Cannot setup Actionability splits, as {fp} does not exist."

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

        sk_dataset = SkimmingDataset(full_dataset,
                                     self.in_transform,
                                     self.target_transform,
                                     selected_types=["quote", "line", "sec-ix", "sec-name"],
                                     sampling=self.sampling_strategy,
                                     sample_size=self.sampling_size)

        # "none" sampling strategy for testing
        sk_dataset_test = SkimmingDataset(full_dataset,
                                          self.in_transform,
                                          self.target_transform,
                                          selected_types=["quote", "line", "sec-ix", "sec-name"],
                                          sampling="none")

        # load splits
        split_ixs = []
        for s in self.splits[:2]:
            split_ixs += [[sk_dataset.ids().index(rid) for i, rid in s]]

        # load test split by pids (instead of pid%node_id (of positive nodes))
        test_pids = set([rid.split("%")[0] for i, rid in self.splits[2]])
        split_ixs += [[sk_dataset_test.ids().index(pid+"%all") for pid in test_pids]]

        self.dataset = DatasetDict({
            "train": Dataset.from_dict(sk_dataset.to_dict(split_ixs[0])),
            "dev": Dataset.from_dict(sk_dataset.to_dict(split_ixs[1])),
            "test": Dataset.from_dict(sk_dataset_test.to_dict(split_ixs[2])),
        })

        for split in self.dataset.keys():
            print(f"Setting up {split}")
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=False
            )
            self.dataset[split].set_format(type="torch")

        print("Dataset initialized")

    @staticmethod
    def _collate_batches(batch):
        result = {"positives": [], "negatives": []}
        for t in result:
            for s in batch:
                result[t] += [s[t]]

        return result

    def train_dataloader(self):
        '''returns training dataloader'''
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = True

        return DataLoader(self.dataset["train"], **dl_config, collate_fn=self._collate_batches)

    def val_dataloader(self):
        '''returns validation dataloader'''
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["dev"], **dl_config, collate_fn=self._collate_batches)

    def test_dataloader(self):
        '''returns test dataloader'''
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["test"], **dl_config, collate_fn=self._collate_batches)

    def predict_dataloader(self):
        '''returns test dataloader'''
        dl_config = copy(self.data_loader_config)
        dl_config["shuffle"] = False

        return DataLoader(self.dataset["test"], **dl_config, collate_fn=self._collate_batches)

    def convert_to_features(self, sample, indices=None):
        # here we get: sample_batch["positives"] -> list of dicts (likewise for negatives)
        # iterate over positives and negatives
        # now we get the relevant features (txt, label) from each of the entries and apply the tokenizer
        # we output a dict of {"positives": [mapped feats], "negatives": [...]}

        features = {"positives": None, "negatives": None}
        for t in features:
            items = sample[t]

            input_raw = items["txt"]
            labels = items["label"]

            features[t] = self.tokenizer(input_raw)
            features[t]["labels"] = labels

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


def create_and_store_splits(full_data: SkimmingDataset,
                            out_dir,
                            splits: list[float],
                            random_gen: int = None):
    # covered papers
    pid_samples = [i.split("%")[0] for i in full_data.ids()]
    pids = list(set(pid_samples))

    hist = Counter(pid_samples)
    max = hist.most_common()[0][1]
    min = hist.most_common()[-1][1]
    bucket_num = 5
    bucket_step = (max - min) // (bucket_num + 1)
    bucket_step = bucket_step if bucket_step > 0 else 1

    def bucketed_samples_per_pid(pid):
        s = hist[pid] - min
        return s // bucket_step

    splitted_pids_ix = paperwise_stratified_split(pids, splits, bucketed_samples_per_pid, random_gen)
    splitted_pids = [[pids[i] for i in split] for split in splitted_pids_ix]
    split_ix, split_ids = [], []
    for spids in splitted_pids:
        ix_of_split = []
        id_of_split = []
        for pid in spids:
            matching_samples = [(i, iid) for i, iid in enumerate(full_data.ids()) if iid.split("%")[0] == pid]
            ix_of_split += [ix for ix, iid in matching_samples]
            id_of_split += [iid for ix, iid in matching_samples]

        split_ix += [ix_of_split]
        split_ids += [id_of_split]

    out_path = os.path.join(out_dir, "skimming_split.json")
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

        sk_dataset = SkimmingDataset(full_dataset,
                                     selected_types=["quote", "line", "sec-ix", "sec-name"],
                                     sampling="full")

        out_path = os.path.join(benchmark_path, d.value, "splits")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        out_files += [create_and_store_splits(sk_dataset, out_path, splits, random_gen)]

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
                           [DATASETS[d] for d in
                            (args.datasets if args.datasets is not None else [d.name for d in DATASETS])])
