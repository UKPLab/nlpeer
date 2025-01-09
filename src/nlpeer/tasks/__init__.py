import random
from typing import Tuple, List

import numpy as np
import sklearn
import sklearn.metrics
import spacecutter.losses
import torch
import torchmetrics
from intertext_graph import Node, Etype, IntertextDocument, SpanNode
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import AdamW
from torch.utils.data import Dataset

from nlpeer.data import paperwise_stratified_split
from nlpeer import NTYPE_TITLE, NTYPE_HEADING, NTYPE_PARAGRAPH, NTYPE_ABSTRACT, DATASETS, ANNOTATION_TYPES, \
    DATASET_REVIEW_OVERALL_SCALES, PaperReviewDataset, ReviewPaperDataset, PaperReviewAnnotations


def get_optimizer(name):
    if name == "adam":
        return AdamW
    else:
        raise ValueError("Unknown optimizer")


def get_loss_function(name, **kwargs):
    name = name.lower()

    if name == "mse":
        return torch.nn.MSELoss()
    elif name == "mrse":
        return lambda x, y: torch.sqrt(torch.nn.MSELoss()(x, y)) #fixme
    elif name == "cll": # cumulative link loss
        ## https://people.csail.mit.edu/jrennie/papers/ijcai05-preference.pdf
        ## https://fa.bianp.net/blog/2013/loss-functions-for-ordinal-regression/
        ## https://www.ethanrosenthal.com/2018/12/06/spacecutter-ordinal-regression/
        return spacecutter.losses.CumulativeLinkLoss()
    elif name == "ce":
        return CrossEntropyLoss()
    elif name == "nll":
        return NLLLoss()
    elif name == "nll-weighted":
        return NLLLoss(weight=torch.tensor([0.4, 0.6]))
    elif name == "acc_reg_3":
        def acc_at_reg_intervals_3(x, y):
            acc = torchmetrics.Accuracy().cuda()

            x2 = torch.add(torch.round(torch.multiply(x, 2)), 1).to(torch.int)
            y2 = torch.add(torch.round(torch.multiply(y, 2)), 1).to(torch.int)
            return acc(x2, y2)

        return acc_at_reg_intervals_3
    elif name == "acc":
        return torchmetrics.Accuracy()
    elif name == "f1-micro":
        return lambda x, y: sklearn.metrics.f1_score(x, y, average="micro")
    elif name == "f1-macro":
        return lambda x, y: sklearn.metrics.f1_score(x, y, average="macro")
    elif name == "f1-macro-rounded":
        sc = kwargs["score_range"]

        def metric(x, y):
            print("x......", x.tolist())
            print("y.......", y.tolist())

            xh = histogram([(s-min(sc)) / (max(sc) - min(sc)) for s in sc], x.tolist())
            yh = histogram([(s-min(sc)) / (max(sc) - min(sc)) for s in sc], y.tolist())
            print("xh", xh)
            print("yh", yh)

            tbucketed = [int(k) for k in xh[-1]]
            pbucketed = [int(k) for k in yh[-1]]

            return sklearn.metrics.f1_score(tbucketed, pbucketed, average="macro")

        return metric
    else:
        raise ValueError(f"Unknown loss function {name}")


def histogram(score_range, scores):
    buckets = score_range

    # create buckets
    bucket_names, counts, full = [], [], []
    for i, left in enumerate(buckets):
        right = buckets[i + 1] if i < len(buckets) - 1 else np.infty

        bucket_names += [f"{left}-{right}"]
        counts += [0]

    # fill in  scores
    for s in scores:
        # round to left most
        if s < buckets[0]:
            s = buckets[0]

        for i, left in enumerate(buckets):
            right = buckets[i + 1] if i < len(buckets) - 1 else np.infty

            if left <= s < right:
                counts[i] += 1
                full += [left]

    return bucket_names, counts, full


def merge_configs(config, default_config):
    to_expand = [(config, default_config)]
    while len(to_expand) > 0:
        c, oc = to_expand.pop(0)

        for k, v in oc.items():
            if k not in c:
                c[k] = v
            elif type(c[k]) == dict:
                to_expand += [(c[k], v)]
            # else ignore oc config, use the conf you already have


def join_review_fields(review_report):
    with_main = "main" in review_report and review_report["main"] is not None

    return "\n".join(f"{fname.title()}\n"+ ftext.replace('\n', '') + f"\n" for (fname, ftext) in review_report.items() if fname != "main") \
           + (review_report["main"].replace("\n", "") + "\n" if with_main else "")


class ReviewScorePredictionDataset(Dataset):
    """
    Task:
        Review x paper (abs) -> review score
    """
    def __init__(self, dataset: ReviewPaperDataset, transform=None, target_transform=None):
        self.data = dataset

        self.transform = transform
        self.target_transform = target_transform

    def ids(self):
        return self.data.ids()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()

        paper_id, paper_meta, paper, review_num, review = self.data[idx]

        extract_oscore = DATASET_REVIEW_OVERALL_SCALES[self.data.dataset_type][0]
        oscore = extract_oscore(review["scores"]["overall"])

        if self.target_transform:
            oscore = self.target_transform(oscore)

        sample = {
            "pid": paper_id,
            "abstract": paper_meta["abstract"],
            "paper": paper,
            "rid": review_num,
            "review": join_review_fields(review["report"]),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample, np.float32(oscore)

    def to_dict(self, idxs:list=None):
        entries = list(range(len(self))) if idxs is None else idxs

        sam0, _ = self[entries[0]]
        fields = list(sam0.keys())
        df = {f: [] for f in fields + ["oscore"]}

        for i in entries:
            sample, score = self[i]
            for f in fields:
                df[f] += [sample[f]]
            df["oscore"] += [score]

        return df


def abstract_with_review_only(sep_token="<s>", truncate_paper=None, truncate_review=None):
    assert truncate_paper is None or 0 <= truncate_paper, "None or int >= 0 expected for paper truncation"
    assert truncate_review is None or 0 <= truncate_review, "None or int >= 0 expected for review truncation"

    def get_abs(sample):
        if truncate_paper is None:
            abs = sample["abstract"]
        else:
            abs = sample["abstract"][:min(len(sample["abstract"]), truncate_paper)]

        if truncate_review is None:
            rev = sample["review"]
        else:
            rev = sample["review"][:min(len(sample["review"]), truncate_review)]

        return {"txt": rev + sep_token + abs}

    return get_abs


class PragmaticLabelingDataset(Dataset):
    """
    Task:
        Review sentece -> review score
    """

    def __init__(self, dataset: PaperReviewDataset, transform=None, target_transform=None):
        self.data = dataset

        assert self.data.dataset_type in [DATASETS.ARR22, DATASETS.F1000], \
            "Only ARR3Y and F1000 are supported for loading a pragmatic labeling dataset"

        self.transform = transform
        self.target_transform = target_transform

        self._setup()

    def _setup(self):
        if self.data.dataset_type == DATASETS.ARR22:
            def get_sentences_with_pragmatics(sample):
                paper_id, paper_meta, paper, reviews = sample

                out = {}
                for review in reviews:
                    rout = []
                    for f, l in [("paper_summary", "neutral"), ("summary_of_strengths", "strength"),
                                 ("summary_of_weaknesses", "weakness"), ("comments,_suggestions_and_typos", "request")]:
                        txt = review["report"][f]
                        if txt is None:
                            continue

                        sent_spans = review["meta"]["sentences"][f]
                        sentences = [txt[s[0]:s[1]].strip() for s in sent_spans]

                        # discard too short sentences (likely erroneous splitting and without content)
                        sentences = [s for s in sentences if len(s) > 5]

                        rout += [(l, s) for s in sentences]
                    out[review["rid"]] = rout

                return out

            paper_wise_sents = {sid: get_sentences_with_pragmatics(self.data[sid]) for sid in self.data.ids()}
            # discard too short sentences (bad splitting, noisy examples)
            paper_wise_sents = {sid: {rid: [(i[0], i[1].strip().replace("\n", " ")) for i in s if len(i[1]) > 6]
                                      for rid, s in sents.items()}
                                for sid, sents in paper_wise_sents.items()}
        elif self.data.dataset_type == DATASETS.F1000:
            self.annos = PaperReviewAnnotations(annotation_type=ANNOTATION_TYPES.PRAG,
                                                dataset=self.data)

            def f1000rd_to_labelset(lbl):
                if lbl == "Strength":
                    return "strength"
                elif lbl == "Weakness":
                    return "weakness"
                elif lbl == "Todo":
                    return "request"
                elif lbl in ["Structure", "Recap", "Other"]:
                    return "neutral"
                else:
                    raise ValueError(f"Passed label {lbl} is not part of the F1000RD label set!")

            def get_sentences_with_pragmatics(sample):
                anno = sample[1]
                paper_id, paper_meta, paper, reviews = sample[0]

                out = {}
                for review in reviews:
                    if review["rid"] not in anno:
                        continue

                    txt = review["report"]["main"]
                    sent_spans = review["meta"]["sentences"]["main"]
                    sentences = [txt[s[0]:s[1]].strip() for s in sent_spans]

                    out[review["rid"]] = [(f1000rd_to_labelset(a), sentences[int(i)]) for i, a in
                                          anno[review["rid"]].items()]

                return out

            paper_wise_sents = {p[0][0]: get_sentences_with_pragmatics(p) for p in self.annos}

        review_wise_sents = {f"{sid}%{k}": v for sid, sents in paper_wise_sents.items() for k, v in sents.items()}
        covered_reviews = list(review_wise_sents.keys())

        self.sentences = {f"{rid}_{i}": s for rid in covered_reviews for i, s in enumerate(review_wise_sents[rid])}
        self.sentence_ids = list(self.sentences.keys())

    def ids(self):
        return self.sentence_ids

    def __len__(self):
        return len(self.sentence_ids)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for this dataset")

        sid = self.sentence_ids[idx]
        label, sentence = self.sentences[sid]

        if self.target_transform:
            label = self.target_transform(label)

        pid, ridi = tuple(sid.split("%"))
        rid = tuple(ridi.split("_"))[0]

        paper_id, paper_meta, paper, reviews = self.data[pid]
        review = next(r for r in reviews if r["rid"] == rid)

        sample = {
            "pid": paper_id,
            "rid": rid,
            "review": join_review_fields(review["report"]),
            "sentence": sentence
        }

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def to_dict(self, idxs:list=None):
        entries = list(range(len(self))) if idxs is None else idxs

        sam0, _ = self[entries[0]]
        fields = list(sam0.keys())
        df = {f: [] for f in fields + ["label"]}

        for i in entries:
            sample, lbl = self[i]
            for f in fields:
                df[f] += [sample[f]]
            df["label"] += [lbl]

        return df


def random_split_pragmatic_labeling_dataset(dataset: PragmaticLabelingDataset, splits: list, random_seed: int = None):
    ids = dataset.ids()

    shuffled_ids = sklearn.utils.shuffle(ids, random_state=random_seed)
    split_idx = [list(idx) for idx in paperwise_stratified_split(shuffled_ids, splits, None, random_seed)]
    split_ids = [list(np.array(ids)[idx]) for idx in split_idx]

    return split_idx, split_ids


def review_sentence_no_context():
    def transform(sample):
        return {"txt": sample["sentence"].strip()}

    return transform


def get_class_map_pragmatics():
    labels = ["strength", "weakness", "request", "neutral"]

    return {
        l: i for i, l in enumerate(labels)
    }


class SkimmingDataset(Dataset):
    """
    Task:
        For each paper paragraph, determine whether it was referenced in a review
    """

    def __init__(self, dataset: PaperReviewDataset,
                 transform=None,
                 target_transform=None,
                 selected_types=None,
                 sampling:str="random",
                 sample_size:int=5):
        self.data = dataset

        self.transform = transform
        self.target_transform = target_transform

        self.samples = None
        self.sample_ids = None

        self.selected_types = selected_types
        self.sampling = sampling
        self.sample_size = sample_size

        self._setup()

    @classmethod
    def _get_linked_paper_nodes(cls, el_sample):
        paper_data, elinks = el_sample
        paper_id, paper_meta, paper, reviews, = paper_data

        res = []
        for rid in elinks:
            for els in elinks[rid].values():
                for el in els:
                    anchor = el["paper_target"]
                    ltype = el["type"]

                    if anchor is not None:
                        res += [(anchor, ltype)]

        return res, paper

    def _sample(self, positive, negative, doc) -> Tuple[List[List[Node]], List[List[Node]], List[str]]:
        # we cannot deal with no negatives or positives -- we always need both in the mix
        if len(positive) == 0 or len(negative) == 0:
            return [], [], []

        if self.sampling == "random":
            negative = [n for n in negative if len(n.content) > 10] # quality assure paragraphs
            paras = positive + negative

            positive_sampled, negative_sampled, sampled_ids = [], [], []
            for n in positive:
                # make random choices until you include at least one negative
                while True:
                    random_paragraphs = random.choices([p for p in paras if p != n], k=self.sample_size-1)
                    if len(set(random_paragraphs).intersection(set(negative))) > 0:
                        break

                positive_sampled += [[n] + [p for p in random_paragraphs if p in positive]]
                negative_sampled += [[p for p in random_paragraphs if p in negative]]
                sampled_ids += [n.ix]

        elif self.sampling == "close":
            paras = positive + negative

            positive_sampled, negative_sampled, sampled_ids = [], [], []
            for n in positive:
                close_paragraphs = list(sorted(paras, key=lambda x: doc.tree_distance(n, x, Etype.NEXT)))
                close_paragraphs = [p for p in close_paragraphs if p != n]

                top_close = close_paragraphs[:min(len(close_paragraphs), self.sample_size-1)]

                positive_sampled += [[n] + [p for p in top_close if p in positive]]
                negative_sampled += [[p for p in top_close if p in negative]]
                sampled_ids += [n.ix]

        elif self.sampling == "full":
            positive_sampled = [positive for p in positive]
            negative_sampled = [negative for p in positive]
            sampled_ids = [p.ix for p in positive]

        elif self.sampling == "none":
            positive_sampled = [positive]
            negative_sampled = [negative]
            sampled_ids = ["all"]

        else:
            raise NotImplementedError(f"Given sampling strategy {self.sampling} does not exist.")

        return positive_sampled, negative_sampled, sampled_ids

    def _sample_pos_neg_paragraphs(self, anchors, pdoc: IntertextDocument):
        # all paragraphs
        paras = [n for n in pdoc.nodes if n.ntype == NTYPE_PARAGRAPH]

        # for each anchor, search for the matching paragraph node and add this to the positives
        pos = []
        for a, t in filter(lambda x: self.selected_types is None or x[1] in self.selected_types, anchors):
            n = pdoc.get_node_by_ix(a)
            if type(n) == SpanNode:
                n = n.src_node

            for p in pdoc.breadcrumbs(n, Etype.PARENT):
                if p in paras:
                    pos += [p]
                    break

        # for now: don't reflect frequency of references
        pos = list(set(pos))

        # pick only "relevant paragraphs" of size > 10
        neg = [n for n in paras if n not in pos]

        # sampling strategy
        return self._sample(pos, neg, pdoc)

    def _setup(self):
        self.elink_annos = PaperReviewAnnotations(annotation_type=ANNOTATION_TYPES.ELINKS,
                                                  dataset=self.data)

        elinks_per_paper = {p[0][0]: self._get_linked_paper_nodes(p) for p in self.elink_annos}

        batches = {}
        for pid, sample in elinks_per_paper.items():
            anchors, pdoc = sample

            pos, neg, ix = self._sample_pos_neg_paragraphs(anchors, pdoc)

            for i, ix in enumerate(ix):
                batches[f"{pid}%{ix}"] = pos[i], neg[i]

        self.samples = batches
        self.sample_ids = list(batches.keys())

    def ids(self):
        return self.sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for this dataset")

        sid = self.sample_ids[idx]
        positives, negatives = self.samples[sid]

        if self.target_transform:
            plabel = self.target_transform("positive")
            nlabel = self.target_transform("negative")
        else:
            plabel = "positive"
            nlabel = "negative"

        if self.transform:
            positives = [self.transform(p) for p in positives]
            negatives = [self.transform(n) for n in negatives]

        return positives, negatives, [plabel for p in positives], [nlabel for n in negatives]

    def to_dict(self, idxs: list = None):
        entries = list(range(len(self))) if idxs is None else idxs

        pos0, neg0, pos0_lbls, neg0_lbls = self[entries[0]]
        fields = list(pos0[0].keys())

        df = {"positives": [],
              "negatives": []}

        for i in entries:
            positives, negatives, pos_labels, neg_labels = self[i]

            # create nested data frame for positives and negatives and fill as usual
            pos_samples = {f: [] for f in fields + ["label"]}
            neg_samples = {f: [] for f in fields + ["label"]}
            for f in fields:
                pos_samples[f] += [p[f] for p in positives]
                neg_samples[f] += [p[f] for p in negatives]

            neg_samples["label"] += neg_labels
            pos_samples["label"] += pos_labels

            # append to overall dataframe-style dict
            df["positives"] += [pos_samples]
            df["negatives"] += [neg_samples]

        return df


def get_paragraph_text(with_struct=False, with_meta=False):
    def get_text(paragraph):
        res = [paragraph.content]

        if with_struct or with_meta:
            parent = None
            title = None
            depth = 0
            edges = [e for e in paragraph.incoming_edges if e.etype == Etype.PARENT]
            while len(edges) > 0:
                edge = edges.pop()
                depth += 1

                if edge.src_node.ntype in [NTYPE_HEADING, NTYPE_ABSTRACT] and parent is None:
                    parent = edge.src_node
                elif edge.src_node.ntype in [NTYPE_TITLE]:
                    title = edge.src_node

                edges += [e for e in edge.src_node.incoming_edges if e.etype == Etype.PARENT]

                if title is not None and parent is not None:
                    break

            if with_meta:
                res = [str(depth)] + res if depth > 0 else res

            res = ([(parent.meta["section"] if parent.meta and "section" in parent.meta else "") + "; " + parent.content] + res) if parent is not None else res
            res = ([title.content] + res) if title is not None else res

        return {
            "txt": "<s>".join(res)
        }

    return get_text


def get_class_map_skimming():
    labels = ["negative", "positive"]

    return {
        l: i for i, l in enumerate(labels)
    }
