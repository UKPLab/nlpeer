import argparse
import json
import logging
import os
import re

from os.path import join as pjoin

import fuzzysearch
import pandas as pd
from intertext_graph.itgraph import IntertextDocument, SpanNode
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from nlpeer import NTYPE_ABSTRACT
from nlpeer.data.datasets.utils import get_review_sentences
from nlpeer.data.utils import list_dirs, list_files

DATA_PATH = os.environ.get("F1000_RD_PATH")
OUT_PATH = os.environ.get("OUT_PATH")


def store_collection_meta(meta, base_path):
    if os.path.exists(pjoin(base_path, "meta.json")):
        with open(pjoin(base_path,"meta.json"), "r") as f:
            prev = json.load(f)
    else:
        prev = {}

    prev["F1000RD"] = meta
    with open(pjoin(base_path, "meta.json"), "w+") as f:
        json.dump(prev, f)


def load_review_reports(path):
    revs = {}
    rid_map = {}
    for paper_path in list_dirs(path):
        pid = os.path.basename(paper_path)

        revs[pid] = {}
        for r in list_files(pjoin(paper_path, "reviews")):
            with open(r, "r") as f:
                report = IntertextDocument.load_json(f)

            rid = os.path.basename(r)[:-len(".json")]
            revs[pid][rid] = report
            rid_map[rid] = pid

    return revs, rid_map


def load_papers(path):
    papers = {}
    for paper_path in list_dirs(path):
        pid = os.path.basename(paper_path)

        with open(pjoin(paper_path, "v1.json"), "r") as f:
            papers[pid] = IntertextDocument.load_json(f)

    return papers


def match_and_update_review_sentences(review_rd:IntertextDocument, review_orig):
    new_sentences = []
    matched = {}
    review_text = review_orig["report"]["main"]
    review_sentences = get_review_sentences(review_orig)["main"]
    review_sentences_cleaned = [re.sub("[^A-Za-z1-9 \-]", "", s.lower()).strip() for s in review_sentences]

    unmatched = []
    for snode in review_rd.nodes:
        if type(snode) != SpanNode:
            continue

        sentence = snode.content

        # perfect match -- add to matches, done
        start_match, end_match = -1, -1
        if sentence.lower() in review_text.lower():
            start_match = review_text.lower().find(sentence.lower())
            end_match = start_match + len(sentence)
        else:
            search_res = fuzzysearch.find_near_matches(
                sentence.lower(),
                review_text.lower(),
                max_l_dist=2
            )

            if 0 < len(search_res) <= 3:
                best = list(sorted(search_res, key= lambda x: x.dist))[0]
                start_match = best.start
                end_match = best.end

        if start_match < 0:
            unmatched += [(snode.start, snode.end, sentence)]
            continue

        new_sentences += [(start_match, end_match)]
        matched[snode.ix] = len(new_sentences) - 1

    review_orig["meta"]["sentences"]["main"] = [[nsent[0], nsent[1]] for nsent in new_sentences]

    return matched, unmatched


def annotate_review_sentences(sentence_mapping, label_map):
    errors = []
    annotations = {}
    for six, label in label_map.items():
        if six in sentence_mapping:
            annotations[sentence_mapping[six]] = label
        else:
            errors += [six]

    return annotations, errors


def match_and_update_paper_sentences(paper_rd: IntertextDocument, paper_orig:IntertextDocument):
    def is_in_node(sentence, node):
        return sentence.lower().strip() in node.content.lower()

    def heuristic_match(sentence):
        # todo
        return None

    # add sentence nodes from the F1000RD version
    matched = {}
    matched_nodes = {}
    unmatched = []
    for snode in paper_rd.nodes:
        if type(snode) != SpanNode or snode.ntype != "s":
            continue

        sentence = snode.content

        match = False
        for cnode in paper_orig.nodes:
            if type(cnode) == SpanNode:
                continue

            # account for listings, which are represented differently in F1000RD
            if not is_in_node(sentence, cnode):
                while sentence[0] == "-":
                    sentence = sentence[1:]

            if is_in_node(sentence, cnode):
                start_orig = cnode.content.lower().find(sentence.lower())
                matched[snode.ix] = (cnode.ix, start_orig, start_orig + len(sentence.strip().lower()))
                matched_nodes[snode.src_node.ix] = cnode.ix
                match = True
                break

        if not match:
            # special case "abstract"
            if sentence.lower().strip() == "abstract":
                oanode, osnode = None, None
                oanodes = [n for n in paper_orig.nodes if n.ntype == NTYPE_ABSTRACT]
                if len(oanodes) > 0:
                    oanode = oanodes[0]
                    osnodes = [n for n in paper_orig.nodes if type(n) == SpanNode and n.ntype == "s" and n.src_node.ix == oanode.ix]

                    if len(osnodes) > 0:
                        osnode = osnodes[0]

                if oanode is not None:
                    matched[snode.ix] = (oanode.ix, osnode.start, osnode.end)
                    matched_nodes[snode.src_node.ix] = oanode.ix
            else:
                best_match = heuristic_match(sentence)    # not implemented atm

                if best_match is None:
                    unmatched += [(snode.start, snode.end, sentence)]
                else:
                    matched[snode.ix] = (best_match[0].ix, best_match[1], best_match[2])
                    matched_nodes[snode.src_node.ix] = best_match[0].ix

    # extend node matching by exact texts
    for cnode in paper_rd.nodes:
        if type(cnode) == SpanNode or cnode.ix in matched_nodes:
            continue

        c = cnode.content.strip().lower()

        for conode in paper_orig.nodes:
            if type(conode) == SpanNode:
                continue

            if c == conode.content.strip().lower():
                matched_nodes[cnode.ix] = conode.ix

    # delete previous setence nodes
    to_delete = []
    for n in paper_orig.nodes:
        if type(n) == SpanNode and n.ntype == "s":
            to_delete += [n]

    for n in to_delete:
        paper_orig.remove_node(n)

    match_map = {}
    for rd_ix in matched:
        orig_ix, orig_start, orig_end = matched[rd_ix]
        sentNode = SpanNode(ntype="s",
                            src_node=paper_orig.get_node_by_ix(orig_ix),
                            start=orig_start,
                            end=orig_end,
                            meta=paper_rd.get_node_by_ix(rd_ix).meta)
        #todo update ix to match pattern of pid_ver_nodenum@sentnum
        paper_orig.add_node(sentNode)
        match_map[rd_ix] = sentNode.ix

    return match_map, matched_nodes, unmatched


def create(in_path=None, out_path=None):
    # paths
    in_path = DATA_PATH if in_path is None else in_path
    out_path = OUT_PATH if out_path is None else out_path

    out_f1000_path = pjoin(out_path, "F1000", "data")
    out_annotations_path = pjoin(out_path, "F1000", "annotations")

    assert out_path is not None and in_path is not None, "Cannot create F1000RD dataset. In and/or out paths are " \
                                                         "missing! "

    logging.info(f"Loading data from {in_path}")
    logging.info(f"Storing data at {out_path}")

    if not os.path.exists(out_annotations_path):
        os.mkdir(out_annotations_path)

    # paths
    prag_path = pjoin(in_path, "data", "simple", "prag.csv")
    exp_links_path = pjoin(in_path, "data", "simple", "exp_links.csv")
    imp_links_path = pjoin(in_path, "data", "simple", "imp_links.csv")
    itgs_path = pjoin(in_path, "data", "itg")

    # meta info on dataset
    collection_meta = {
        "dataset_timestamp": os.path.getmtime(itgs_path),
        "origin_dataset": "F1000RD",
        "annotation_types": [
            "rd_review_pragmatics",
            "rd_elinks",
            "rd_ilinks"
        ]
    }
    with open(pjoin(in_path, "LICENSE.txt"), "r") as f:
        collection_meta["license"] = f.read().strip()

    store_collection_meta(collection_meta, out_annotations_path)

    # load review reports per paper and papers
    pid_to_reviews, rid_map = load_review_reports(itgs_path)
    pid_to_papers = load_papers(itgs_path)

    # copying review annotations
    logging.info("Adding annotations from F1000RD to F1000/annotations")

    prags = pd.read_csv(prag_path, header=0)
    elinks = pd.read_csv(exp_links_path, header=0)
    ilinks = pd.read_csv(imp_links_path, header=0)

    match_error = []
    anno_error = []
    prag_annotations, elink_annotations, ilink_annotations = {}, {}, {}
    for rid in tqdm(rid_map, desc="Iterating over reviews"):
        pid = rid_map[rid]
        rd_review = pid_to_reviews[pid][rid]
        rd_paper = pid_to_papers[pid]

        # get reference review from F1000 to transfer annotations
        orig_reviews_path = pjoin(out_f1000_path, pid, "v1", "reviews.json")
        with open(orig_reviews_path, "r") as f:
            orig_reviews = json.load(f)
        orig_review = next(r for r in orig_reviews if r["rid"] == rid)

        # update sentences in the review
        rev_matched_sents, rev_unmatched_sents = match_and_update_review_sentences(rd_review, orig_review)
        if len(rev_unmatched_sents) > 0:
            match_error += [("review", rid, r) for r in rev_unmatched_sents]

        # store changed review sentences
        with open(orig_reviews_path, "w+") as f:
            json.dump(orig_reviews, f)

        # get reference paper from F1000 to transfer annotations
        orig_paper_path = pjoin(out_f1000_path, pid, "v1", "paper.itg.json")
        with open(orig_paper_path, "r") as f:
            orig_paper = IntertextDocument.load_json(f)

        # update paper sentences
        paper_matched_sents, matched_nodes, paper_unmatched_sents = match_and_update_paper_sentences(rd_paper, orig_paper)
        if len(paper_unmatched_sents) > 0:
            match_error += [("paper", pid, r) for r in paper_unmatched_sents]
        paper_matched_sents.update(matched_nodes)

        # store changed paper sentences
        with open(orig_paper_path, "w+")as f:
            orig_paper.save_json(f)

        # pragmatics
        pmap = {}
        for _, pentry in prags[prags.review_id == rid].iterrows():
            pmap[pentry["review_sentence_id"]] = pentry["prag"]
        pannos, errs = annotate_review_sentences(rev_matched_sents, pmap)

        anno_error += [("prag", pid, rid, e) for e in errs]

        if pid not in prag_annotations:
            prag_annotations[pid] = {}
        if rid not in prag_annotations[pid]:
            prag_annotations[pid][rid] = pannos

        # explicit links
        errs = explicit_link_annotations(elink_annotations, elinks, paper_matched_sents, pid, rev_matched_sents, rid, orig_review)
        anno_error += [("elink", pid, rid, e) for e in errs]

        # implicit links
        errs = implicit_link_annotations(ilink_annotations, ilinks, paper_matched_sents, pid,
                                 rev_matched_sents, rid)
        anno_error += [("ilink", pid, rid, e) for e in errs]

    # ignore boilerplate not matched errors
    anno_error = [e for e in anno_error if not(e[0] == "prag" and e[-1].endswith("_0@0"))]

    # write to annotations
    for o, a in [("rd_review_pragmatics", prag_annotations),
                 ("rd_elinks", elink_annotations),
                 ("rd_ilinks", ilink_annotations)]:
        for pid in a:
            if len(a[pid]) == 0:
                continue

            if not os.path.exists(pjoin(out_annotations_path, pid)):
                os.mkdir(pjoin(out_annotations_path, pid))
            if not os.path.exists(pjoin(out_annotations_path, pid, "v1")):
                os.mkdir(pjoin(out_annotations_path, pid, "v1"))

            p_path = pjoin(out_annotations_path, pid, "v1", f"{o}.json")
            with open(p_path, "w+") as f:
                json.dump(a[pid], f)

    # log errors
    logging.info("Matching Errors")
    for e in match_error:
        if e[2][2].strip().startswith("Reviewer response for version"):
            continue

        logging.info(f"Failed to match for {e[0]} {e[1]} with start= {e[2][0]}; end= {e[2][1]} and sentence: {e[2][2]}")

    logging.warning("Annotation Transfer Errors")
    for e in anno_error:
        logging.warning(f"Failed to transfer annotations: {e}")


def implicit_link_annotations(ilink_annotations, ilinks, paper_matched_sents, pid, rev_matched_sents,
                             rid):
    errors = []
    rows = list(ilinks[ilinks.review_id == rid].iterrows())
    if len(rows) > 0:
        if pid not in ilink_annotations:
            ilink_annotations[pid] = {}
        if rid not in ilink_annotations[pid]:
            ilink_annotations[pid][rid] = {}

        for _, entry in rows:
            new_rsent_id = rev_matched_sents[entry["review_sentence_id"]] if entry[
                                                                                 "review_sentence_id"] in rev_matched_sents else None
            new_psent_id = paper_matched_sents[entry["paper_sentence_id"]] if entry[
                                                                                  "paper_sentence_id"] in paper_matched_sents else None

            imp_a = entry["imp_a"] if pd.notna(entry["imp_a"]) else 0
            imp_b = entry["imp_b"] if pd.notna(entry["imp_b"]) else 0
            linked = int(imp_a) + int(imp_b)

            if linked > 0 and (new_psent_id is None or new_rsent_id is None):
                errors += [(entry["review_sentence_id"], entry["paper_sentence_id"])]
                continue

            if linked > 0:
                ilink_annotations[pid][rid][new_rsent_id] = (new_psent_id, linked)

    return errors


def explicit_link_annotations(elink_annotations, elinks, paper_matched_sents, pid, rev_matched_sents, rid, review):
    errors = []
    rows = list(elinks[elinks.review_id == rid].iterrows())
    if len(rows) > 0:
        if pid not in elink_annotations:
            elink_annotations[pid] = {}
        if rid not in elink_annotations[pid]:
            elink_annotations[pid][rid] = {"main": []}

        for _, entry in rows:
            new_rsent_id = rev_matched_sents[entry["review_sentence_id"]] if entry[
                                                                                 "review_sentence_id"] in rev_matched_sents else None
            new_psent_id = paper_matched_sents[entry["paper_sentence_id"]] if entry[
                                                                                  "paper_sentence_id"] in paper_matched_sents else None

            if new_psent_id is None or new_rsent_id is None:
                errors += [(entry["type"], entry["review_sentence_id"], entry["paper_sentence_id"], entry["review_text"], entry["paper_text"])]
                continue

            sent_span = review["meta"]["sentences"]["main"][int(new_rsent_id)]
            sent = review["report"]["main"][sent_span[0]: sent_span[1]]

            elink_annotations[pid][rid]["main"] += [{"type": entry["type"], "rev_span": sent_span, "rev_text": sent}]

    return errors


def arg_parse():
    parser = argparse.ArgumentParser(description="Creating the F1000 dataset within the benchmark")
    parser.add_argument(
        "--data_path", type=str, help="Path to the directory of the F1000 dataset."
    )
    parser.add_argument(
        "--output_directory", type=str, help="Path to the top directory of the benchmark dataset."
    )

    return parser


def main(args):
    logging.basicConfig(level="INFO")

    create(args.data_path, args.output_directory)


if __name__ == "__main__":
    parser = arg_parse()
    main(parser.parse_args())