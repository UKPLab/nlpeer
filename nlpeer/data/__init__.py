import json
from collections import Counter
from typing import Callable, Collection, List

import numpy as np

import sklearn.utils
from sklearn.model_selection import train_test_split

from nlpeer import DATASETS, PaperReviewDataset, ReviewPaperDataset
from nlpeer.utils import list_dirs


def filter_gte_x_reviews(dataset: PaperReviewDataset, x: int):
    """
    Discards papers with less than x reviews.

    :param dataset: dataset to be filtered
    :param x: the threshold of min number of reviews
    :return: void
    """
    def gte0_reviews(paper_data):
        paper_id, paper_meta, paper, reviews = paper_data

        return len(reviews) >= x

    dataset.filter_paperwise(gte0_reviews)


def load_splits_from_file(file_path: str, dataset_type: DATASETS = None):
    """
    For a given dataset type and file path, loads the splits from file as lists of indexes.

    :param file_path: path to the split file
    :param dataset_type: type of the dataset to verify the split matches
    :return: the list of splits
    """
    with open(file_path, "r") as file:
        split_data = json.load(file)

    if dataset_type is not None:
        assert split_data["dataset_type"] == dataset_type.name, \
            f"Mismatch of dataset types during loading. Expected {dataset_type}"

    return split_data["splits"]


def store_splits_to_file(dataset: PaperReviewDataset, splits: Collection, out_path: str, random_gen):
    """
    Stores the given splits for the given dataset to disk.

    :param dataset: the dataset split by the given splits
    :param splits: the list of split indexes
    :param out_path: the output path to store the split
    :param random_gen: random seed to store along
    :return:
    """
    jsplits = {
        "dataset_type": dataset.dataset_type.name,
        "dataset": dataset.__class__.__name__,
        "splits": [
            [[s, dataset.ids()[s]] for s in split] for split in splits
        ],
        "random": str(random_gen)
    }

    with open(out_path, "w+") as file:
        json.dump(jsplits, file)


def paperwise_random_split(dataset: PaperReviewDataset, splits: List[float], random_seed: int):
    """
    Splits the given dataset by papers with the given split proportions.
    Automatically splits considering the distribution of reviews per paper for stratification.

    :param dataset: dataset to be split
    :param splits: the split proportions as a list of floats
    :param random_seed: random seed for shuffling
    :return: the splits
    """
    # handle review paper datasets
    if type(dataset) == ReviewPaperDataset:
        reviews_per_paper = list(dataset.paperwise_review_ids.items())

        def strat_criterion(rpp):
            return len(rpp[1])

        split_indices = paperwise_stratified_split(reviews_per_paper,
                                                   splits,
                                                   stratification_criterion=strat_criterion,
                                                   random_seed=random_seed)

        rid_splits = []
        for split_i in split_indices:
            covered_papers = [reviews_per_paper[i] for i in split_i]
            rid_splits += [[ridx for pid, ridxs in covered_papers for ridx in ridxs]]
        return rid_splits

    # handle paper review datasets (split by paper proportions, that's it; ignore review proportions)
    else:
        # for consistency of randomness, we use the same method but without stratification
        split_indices = paperwise_stratified_split(dataset, splits, stratification_criterion=None, random_seed=random_seed)

        return [s.tolist() for s in split_indices]


def paperwise_stratified_split(dataset: Collection, splits: List[float], stratification_criterion: Callable, random_seed: int):
    """
    Splits the given dataset by a stratification criterion given as function on papers.

    :param dataset: the dataset to be split
    :param splits: the split proportions
    :param stratification_criterion: the callable stratification criterion mapping to discrete values
    :param random_seed: random seed for shuffling
    :return: the splits
    """
    assert np.round(sum(splits), 1) == 1.0, f"Split sizes need to add evenly to 1.0. Given sum: {sum(splits)}"

    # get "class label" for stratification
    indices = np.arange(0, len(dataset))
    if stratification_criterion is None:
        strat_labels = np.array(["nostrat" for i in indices])
    else:
        strat_labels = np.array([stratification_criterion(s) for s in dataset])

    # prep iterative splitting
    out = []
    to_split = (indices, strat_labels, 0)
    split_further = None

    # perform sequence of splits (until the last one, which is implied)
    for original_split_size in splits[:-1]:
        ixs, lbls, split_sofar = to_split

        split_size = original_split_size / (1-split_sofar) if split_sofar > 0 else original_split_size

        # lbl occurs only once: prohibited by scipy -- remove beforehand
        single_label_instances = []
        for l, c in Counter(lbls).items():
            if c > 1:
                continue
            instances = lbls == l

            single_label_instances += [ixs[instances][0]]
            ixs = ixs[np.logical_not(instances)]
            lbls = lbls[np.logical_not(instances)]

        bi_split = train_test_split(ixs, test_size=split_size, stratify=lbls, random_state=random_seed)
        split_further, split_done = bi_split[0], bi_split[1]

        # add at random to one of the splits
        if len(single_label_instances) > 0:
            split_i = round(split_size*len(single_label_instances))

            single_label_instances = sklearn.utils.shuffle(single_label_instances, random_state=random_seed)

            split_further = np.append(split_further, single_label_instances[:split_i]).astype(int)
            split_done = np.append(split_done, single_label_instances[split_i:]).astype(int)

        out += [split_done]
        to_split = (split_further, strat_labels[split_further], split_sofar + original_split_size)

    # add left-over split
    out += [split_further]

    rout = []
    for s in out:
        rout += [sklearn.utils.shuffle(s)]

    return rout
