import json
import logging
import os
import re
from enum import Enum
from os.path import join as pjoin
from typing import Callable

import numpy as np
from intertext_graph import IntertextDocument

from nlpeer.utils import list_dirs

NTYPE_TITLE = "title"
NTYPE_HEADING = "heading"
NTYPE_PARAGRAPH = "paragraph"
NTYPE_ABSTRACT = "abstract"
NTYPE_LIST = "list"
NTYPE_LIST_ITEM = "list_item"
NYTPE_ELEMENT_REFERENCE = "elem_reference"
NTYPE_BIB_REFERENCE = "bib_reference"
NTYPE_HEADNOTE = "headnote"
NTYPE_FOOTNOTE = "footnote"
NTYPE_FIGURE = "figure"
NTYPE_TABLE = "table"
NTYPE_FORMULA = "formula"
NTYPE_MEDIA = "media"
NTYPE_BIB_ITEM = "bib_item"

NTYPES = []
NTYPES += [NTYPE_TITLE]
NTYPES += [NTYPE_HEADING]
NTYPES += [NTYPE_PARAGRAPH]
NTYPES += [NTYPE_ABSTRACT]
NTYPES += [NTYPE_LIST]
NTYPES += [NTYPE_LIST_ITEM]
NTYPES += [NYTPE_ELEMENT_REFERENCE]
NTYPES += [NTYPE_BIB_REFERENCE]
NTYPES += [NTYPE_HEADNOTE]
NTYPES += [NTYPE_FOOTNOTE]
NTYPES += [NTYPE_FIGURE]
NTYPES += [NTYPE_TABLE]
NTYPES += [NTYPE_FORMULA]
NTYPES += [NTYPE_MEDIA]
NTYPES += [NTYPE_BIB_ITEM]


class DATASETS(Enum):
    ARR22 = "ARR-22"
    ARR23 = "ARR-23"
    ARR24 = "ARR-24"
    F1000 = "F1000"
    ACL17 = "PeerRead-ACL2017"
    CONLL16 = "PeerRead-CONLL2016"
    COLING20 = "COLING2020"
    ELIFE = "ELIFE"
    EMNLP23 = "EMNLP23"
    PLOS = "PLOS"


class PAPERFORMATS(Enum):
    PDF = ".pdf"
    ITG = ".itg.json"
    GROBID = ".tei"
    XML = ".xml"
    TEX = ".tex"


class ANNOTATION_TYPES(Enum):
    ELINKS = "elinks", lambda pdir, ver: pjoin(pdir, f"v{ver}", "elinks.json")
    ILINKS_RD = "rd_ilinks", lambda pdir, ver: pjoin(pdir, f"v{ver}", "rd_ilinks.json")
    ELINKS_RD = "rd_elinks", lambda pdir, ver: pjoin(pdir, f"v{ver}", "rd_elinks.json")
    DIFFS = "diff", lambda pdir, ver: pjoin(pdir, f"diff_{ver}_{ver+1}.json")
    PRAG = "rd_review_pragmatics", lambda pdir, ver: pjoin(pdir, f"v{ver}", "rd_review_pragmatics.json")


DATASET_PAPERFORMATS = {
    DATASETS.ARR22: [PAPERFORMATS.PDF, PAPERFORMATS.ITG, PAPERFORMATS.GROBID],
    DATASETS.ARR23: [PAPERFORMATS.PDF, PAPERFORMATS.ITG],
    DATASETS.ARR24: [PAPERFORMATS.PDF, PAPERFORMATS.ITG],
    DATASETS.F1000: [PAPERFORMATS.PDF, PAPERFORMATS.ITG, PAPERFORMATS.XML],
    DATASETS.ACL17: [PAPERFORMATS.PDF, PAPERFORMATS.ITG, PAPERFORMATS.GROBID],
    DATASETS.CONLL16: [PAPERFORMATS.PDF, PAPERFORMATS.ITG, PAPERFORMATS.GROBID],
    DATASETS.COLING20: [PAPERFORMATS.PDF, PAPERFORMATS.ITG, PAPERFORMATS.GROBID, PAPERFORMATS.TEX],
    DATASETS.ELIFE: [PAPERFORMATS.PDF, PAPERFORMATS.ITG, PAPERFORMATS.GROBID],
    DATASETS.PLOS: [PAPERFORMATS.XML, PAPERFORMATS.ITG],
}


def arr_overall_to_score(oscore):
    match = re.match("^\d+\.?\d* ?=?", oscore)
    return float(oscore[0:match.span()[1] - 1].strip())


def f1000_overall_to_score(oscore):
    if oscore == "approve":
        return 2
    elif oscore == "approve-with-reservations":
        return 1
    elif oscore == "reject":
        return 0
    else:
        raise ValueError("Passed overall score does not exist in the F1000 dataset")


DATASET_REVIEW_OVERALL_SCALES = {
    DATASETS.ARR22: (arr_overall_to_score, np.arange(1, 5 + 0.5, 0.5)),
    DATASETS.ARR23: (arr_overall_to_score, np.arange(1, 5 + 0.5, 0.5)),
    DATASETS.ARR24: (arr_overall_to_score, np.arange(1, 5 + 0.5, 0.5)),
    DATASETS.F1000: (f1000_overall_to_score, np.arange(0, 3, 1)),
    DATASETS.ACL17: (lambda x: int(x), np.arange(1, 5+1, 1)),
    DATASETS.CONLL16: (lambda x: int(x), np.arange(1, 5+1, 1)),
    DATASETS.COLING20: (lambda x: int(x), np.arange(1, 5+1, 1))
}


class PaperReviewDataset:
    """
    Dataset of papers and reviews indexed by paper IDs.

    """
    datapath = None
    _dataset_type = None
    paper_ids = []
    paper_dirs = {}
    index = 0
    _version = None

    cache = None

    def __init__(self, base_path: str,
                 dataset: DATASETS|str,
                 version: int,
                 paper_format:PAPERFORMATS=PAPERFORMATS.ITG,
                 hold_in_memory:bool=True,
                 preload:bool=False,
                 strict_loading=False):
        self.strict_loading = strict_loading

        # get path
        if type(dataset) == str:
            sp = dataset
        else:
            sp = dataset.value

        datapath = pjoin(base_path, sp, "data")
        assert os.path.exists(datapath), f"The passed dataset does not exist in the given directory {datapath}"

        self.datapath = datapath
        self._dataset_type = dataset
        self._version = version

        # warn on memory
        assert not preload or hold_in_memory, "Invalid configuration. Can only preload if holding in memory"

        if preload:
            logging.info(f"You are loading {dataset} completely into memory. For large datasets this is "
                         f"discouraged.")

        if not strict_loading:
            logging.info("You are loading dataset in non-strict mode. I.e. you have no guarantees that each paper "+
                         "text and review can be loaded during iteration. Use only if your dataset contains reviews "+
                         "without papers and vice versa.")

        # setup loader
        self._setup(hold_in_memory, preload, version, paper_format)

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, dataset_type):
        self._dataset_type = dataset_type

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, version):
        self._version = version

    def ids(self):
        return [pid for pid in self.paper_ids]

    def file_path(self):
        return os.path.sep.join(self.datapath.split(os.path.sep)[:-2])

    @staticmethod
    def _get_version_dir(pdir, version):
        vdir = pjoin(pdir, f"v{version}")
        return vdir if os.path.exists(vdir) else None

    def load(self, paper_dir):
        paper_id = paper_dir.split(os.path.sep)[-2]

        # get metadata
        meta_path = pjoin(paper_dir, "meta.json")
        assert os.path.exists(meta_path), f"Path to paper metadata does not exist: {meta_path}"

        with open(meta_path, "r") as fp:
            paper_meta = json.load(fp)

        # get doc
        if self.paper_format == PAPERFORMATS.ITG:
            itg_path = pjoin(paper_dir, f"paper{self.paper_format.value}")
            itg_exists = os.path.exists(itg_path)

            assert not self.strict_loading or itg_exists, f"Path to the ITG version of the paper does not exist: {itg_path}"

            if itg_exists:
                with open(itg_path, "r") as fp:
                    paper = IntertextDocument.load_json(fp)
            else:
                paper = None
        else:
            raise ValueError(f"Paper format {self.paper_format} is currently not supported for loading!")

        # get reviews
        review_path = pjoin(paper_dir, "reviews.json")
        reviews_exist = os.path.exists(review_path)

        assert not self.strict_loading or reviews_exist, f"Path to the reviews of the paper do not exist: {review_path}"

        if reviews_exist:
            with open(review_path, "r") as fp:
                reviews = json.load(fp)
        else:
            reviews = {}

        return paper_id, paper_meta, paper, reviews

    def _setup(self, hold_in_memory, preload, version, paper_format):
        paper_dirs = map(lambda x: PaperReviewDataset._get_version_dir(x, version), list_dirs(self.datapath))
        paper_dirs = filter(lambda x: x is not None, paper_dirs)

        # fill management vars
        self.paper_dirs = dict([(pd.split(os.path.sep)[-2], pd) for pd in paper_dirs])
        self.paper_ids = list(self.paper_dirs.keys())
        self.paper_format = paper_format
        self.cache = None
        self.index = 0

        # if all data should be loaded at once and held in memory, do so
        if hold_in_memory:
            if preload:
                self.cache = {pid: self.load(pd) for pid, pd in paper_dirs}
            else:
                self.cache = {}

    def __len__(self):
        return len(self.paper_ids)

    def __getitem__(self, idx):
        if type(idx) == str:
            assert idx in self.paper_ids, "Passed paper ID string is not part of this dataset"
            pid = idx
        elif type(idx) == int:
            assert 0 <= idx < len(self.paper_ids), f"Passed paper index {idx} is out of range"
            pid = self.paper_ids[idx]
        elif type(idx) == list:
            return [self[lx] for lx in idx]
        else:
            raise TypeError(f"The passed index type {type(idx)} is not supported for this dataset")

        # with cached data
        if self.cache is not None:
            if pid in self.cache:
                return self.cache[pid]
            else:
                self.cache[pid] = self.load(self.paper_dirs[pid])
                return self.cache[pid]

        # regular loading
        return self.load(self.paper_dirs[pid])

    def __next__(self):
        self.index += 1

        if self.index-1 >= len(self):
            raise StopIteration()

        return self[self.index-1]

    def __iter__(self):
        while True:
            try:
                yield next(self)
            except StopIteration:
                break

        self.index = 0

    def filter_paperwise(self, paperwise_filter: Callable):
        matched_paper_ids = [paper[0] for paper in self if paperwise_filter(paper)]
        non_matched_paper_ids = [pid for pid in self.paper_ids if pid not in matched_paper_ids]

        self.paper_ids = matched_paper_ids
        for nm in non_matched_paper_ids:
            del self.paper_dirs[nm]

            if self.cache is not None and nm in self.cache:
                del self.cache[nm]

        # just to be sure, reset as the index is invalid as of now
        self.index = 0


class ReviewPaperDataset(PaperReviewDataset):
    """
    Dataset of papers and reviews indexed by review IDs.
    """
    review_ids = []
    paperwise_review_ids = {}
    force_hold_in_memory = False

    def __init__(self, base_path: str, dataset: DATASETS, version: int, paper_format:PAPERFORMATS=PAPERFORMATS.ITG, hold_in_memory:bool=False, preload:bool=False):
        # hold data in memory by default, but add an own cleaning strategy while loading by index
        self.force_hold_in_memory = hold_in_memory

        super().__init__(base_path, dataset, version, paper_format, True, preload)

    def _setup(self, hold_in_memory, preload, version, paper_format):
        # load file pointers as in the paper review dataset
        super()._setup(hold_in_memory, preload, version, paper_format)

        assert all("%" not in pid for pid in self.paper_ids), "There exist malformed paper ids. % is not permitted"

        # index the reviews
        self.review_ids = []
        self.paperwise_review_ids = {}
        for pid in self.paper_ids:
            pdir = self.paper_dirs[pid]
            rfile = pjoin(pdir, "reviews.json")

            with open(rfile, "r") as rfs:
                new_rids = [f"{pid}%{str(i)}" for i in range(len(json.load(rfs)))]

                self.paperwise_review_ids[pid] = self.paperwise_review_ids.get(pid, []) \
                                                 + [len(self.review_ids) + rid for rid in range(len(new_rids))]
                self.review_ids += new_rids

    def ids(self):
        return [pid for pid in self.review_ids]

    @staticmethod
    def _review_wise_sample(data, review_num):
        paper_id, paper_meta, paper, reviews = data

        return paper_id, paper_meta, paper, review_num, reviews[review_num]

    def __len__(self):
        # override by review ids' length
        return len(self.review_ids)

    def __getitem__(self, idx):
        # override to fetch by index on reviews
        if type(idx) == str:
            assert idx in self.review_ids, "Passed review ID string is not part of this dataset"
            assert idx.count("%") == 1, "Passed review ID string is malformed. % is only allowed as a separator."
            rid = idx
        elif type(idx) == int:
            assert 0 <= idx < len(self.review_ids), f"Passed review index {idx} is out of range"
            rid = self.review_ids[idx]
        elif type(idx) == list:
            return [self[lx] for lx in idx]
        else:
            raise TypeError(f"The passed index type {type(idx)} is not supported for this dataset")

        # determine pid and review number
        pid, rnum = rid.split("%")[0], int(rid.split("%")[1])

        # with cached data (always the case)
        if pid in self.cache:
            res = self._review_wise_sample(self.cache[pid], rnum)
        else:
            self.cache[pid] = self.load(self.paper_dirs[pid])
            res = self._review_wise_sample(self.cache[pid], rnum)

        # cache cleaning strategy: discard loaded papers if the rnum is the last one
        if not self.force_hold_in_memory and rnum == len(self.cache[pid][3]):
            del self.cache[pid]

        # return result
        return res

    def filter_paperwise(self, paperwise_filter: Callable):
        # filter paper ids, such that paper_ids only old the still permitted ones
        # cache is cleared, too
        matched_paper_ids, non_matched_paper_ids = [], []
        for pid in self.paper_ids:
            paper = super().__getitem__(pid)
            if paperwise_filter(paper):
                matched_paper_ids += [pid]
            else:
                non_matched_paper_ids += [pid]

        self.paper_ids = matched_paper_ids
        for nm in non_matched_paper_ids:
            del self.paper_dirs[nm]

            if self.cache is not None and nm in self.cache:
                del self.cache[nm]

        # get review ids to remove
        reviewids_to_remove = [r for pid, revs in self.paperwise_review_ids.items() if pid in non_matched_paper_ids
                                 for r in revs ]
        for pid in non_matched_paper_ids:
            del self.paperwise_review_ids[pid]

        for rid in reviewids_to_remove:
            self.review_ids.remove(rid)

        # just to make sure, in case someone is iterating over this (index would be wrong now)
        self.index = 0


class PaperReviewAnnotations:
    """
    Extends a given paper review dataset by annotations according to the specified annotaiton type.

    """
    cache = None

    def __init__(self, annotation_type: ANNOTATION_TYPES, dataset: PaperReviewDataset):
        base_path = dataset.file_path()

        annopath = pjoin(base_path, dataset.dataset_type.value, "annotations")
        assert os.path.exists(annopath), f"The passed dataset does not exist in the given directory {annopath}"

        self.annopath = annopath
        self._annotation_type = annotation_type

        assert type(dataset) != ReviewPaperDataset, "ReviewPaperDatasets not supported."
        self.dataset = dataset

        # setup loader
        self._setup()

        self.index = 0

    def _setup(self):
        version = self.dataset.version

        anno_paper_dirs = list_dirs(self.annopath)
        anno_paper_dirs = filter(lambda x: os.path.exists(self._annotation_type.value[1](x, version)), anno_paper_dirs)

        # fill management vars
        self.anno_paper_dirs = dict([(os.path.basename(pd), pd) for pd in anno_paper_dirs])
        self.anno_pids = list(self.anno_paper_dirs.keys())
        self._extract_annotations = lambda x: self._annotation_type.value[1](x, version)

        self.cache = {}

    def ids(self):
        return self.anno_pids

    def __len__(self):
        return len(self.anno_paper_dirs)

    def __getitem__(self, idx):
        if type(idx) == str:
            assert idx in self.anno_pids, "Passed paper ID string is not part of this dataset"
            pid = idx
        elif type(idx) == int:
            assert 0 <= idx < len(self.anno_pids), f"Passed paper index {idx} is out of range"
            pid = self.anno_pids[idx]
        elif type(idx) == list:
            return [self[lx] for lx in idx]
        else:
            raise TypeError(f"The passed index type {type(idx)} is not supported for this dataset")

        d = self.dataset[pid]

        # get annotation
        if pid not in self.cache:
            anno_path = self._extract_annotations(self.anno_paper_dirs[pid])
            with open(anno_path, "r") as jf:
                self.cache[pid] = json.load(jf)

        return d, self.cache[pid]

    def __next__(self):
        self.index += 1

        if self.index-1 >= len(self):
            raise StopIteration()

        return self[self.index-1]

    def __iter__(self):
        while True:
            try:
                yield next(self)
            except StopIteration:
                break

        self.index = 0

