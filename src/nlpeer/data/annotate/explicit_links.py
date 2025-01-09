import argparse
import json
import logging
import os
import re
from os.path import join as pjoin
from typing import List, Tuple, Dict

import fuzzysearch
import pandas as pd
from intertext_graph.itgraph import IntertextDocument, SpanNode
from tqdm import tqdm

from nlpeer import NTYPE_TITLE, NTYPE_HEADING, NTYPE_ABSTRACT, NTYPE_FIGURE, NTYPE_TABLE, NTYPE_BIB_ITEM, DATASETS
from nlpeer.data.utils import list_dirs, list_files

OUT_PATH = os.environ.get("OUT_PATH")

from external.f1000rd.analysis.exp_linker import replace_ordinal_string_with_number, \
    split_quote, strip_numbers_from_title


def store_collection_meta(meta, base_path):
    if os.path.exists(pjoin(base_path, "meta.json")):
        with open(pjoin(base_path,"meta.json"), "r") as f:
            prev = json.load(f)
    else:
        prev = {}

    prev["ELINKS"] = meta
    with open(pjoin(base_path, "meta.json"), "w+") as f:
        json.dump(prev, f)


# adapted from F1000RD/analysis/exp_linker
def load_static_patterns(path=None):
    def_path = "resources/exp_patterns.tsv" if path is None else path

    logging.info(f"Loading static patterns from {def_path}")

    # static patterns
    patterns_df = pd.read_csv(
        def_path,
        delimiter='\t'
    )
    return [
        (row.pattern, row.type)
        for row in patterns_df.itertuples()
    ]


# adapted from F1000RD/analysis/exp_linker
def load_patterns(paper_doc: IntertextDocument):
    # get headings
    headings = []
    for node in paper_doc.nodes:
        if node.ntype == NTYPE_HEADING:
            headings += [node.content]
        elif node.ntype == NTYPE_ABSTRACT:
            headings += [node.content]

    # generate patterns from headings
    hd_patterns = make_patterns_from_section_titles(headings)

    return hd_patterns

# adapted from F1000RD
def make_patterns_from_section_titles(section_titles: [str]):
    """
    Make patterns from section titles
    strip off numbering
    """

    # Lambda functions that return regex patterns
    TEMPLATES = [
        lambda x: rf'^.{{,3}}(?P<ix>{x}).{{,3}}$', # Review headline with section title
        lambda x: rf'\bthe (?P<ix>{x})', # "the results"
        lambda x: rf'^.{{,3}}(?P<ix>{x}) ?[-:;.]', # "Results: bla bla"
        lambda x: rf'(in|under) (?P<ix>{x})', # "in methods"
        lambda x: rf'\b["”“‘’\'«‹»›„“‟”’❝❞❮❯⹂〝〞〟＂‚‘‛❛❜❟](?P<ix>{x})["”“‘’\'«‹»›„“‟”’❝❞❮❯⹂〝〞〟＂‚‘‛❛❜❟]\b'
        # Section title in quotation marks of various kinds
    ]

    # Make pattern for each combination of section title and pattern template
    patterns = []
    for section_title in section_titles:
        for template_func in TEMPLATES:
            stripped_title = strip_numbers_from_title(re.escape(section_title.lower()))
            pattern = template_func(stripped_title)
            patterns.append((pattern, 'sec-name'))

    return patterns

# adapted from F1000RD/analysis/exp_linker
def find_pointers(txt: str, patterns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out = []
    for pat, cat in patterns:
        res = re.finditer(pat, txt.lower())
        for r in res:
            value = r.group('ix')
            value = replace_ordinal_string_with_number(
                value
            )
            out += [(cat, value, r.regs[0])]

    return out

# extended F1000RD
def find_targets_for_sec_name(tgt_itg: IntertextDocument, match_text: str):
    """Handle "sec-name" matches"""
    ret = []
    META = {
        'exp': ['sec-name']
    }

    # Handle title, abstract and regular section names separately, because
    # they link to different ntypes
    if match_text == 'title':
        for tgt_node in tgt_itg.nodes:
            if tgt_node.ntype == NTYPE_TITLE:
                ret.append(tuple((META, tgt_node)))
    elif match_text == 'abstract':
        for tgt_node in tgt_itg.nodes:
            if tgt_node.ntype == NTYPE_ABSTRACT:
                ret.append(tuple((META, tgt_node)))
    else:
        for tgt_node in tgt_itg.nodes:
            if tgt_node.ntype == NTYPE_HEADING:
                if match_text in tgt_node.content.lower():
                    ret.append(tuple((META, tgt_node)))

    return ret

#extended F1000RD
def find_targets_for_sec_ix(tgt_itg: IntertextDocument, match_text: str):
    """Handle section index matches"""
    ret = []
    META = {
        'exp': ['sec-ix']
    }

    for tgt_node in tgt_itg.nodes:
        if tgt_node.ntype == NTYPE_HEADING:
            try:
                if tgt_node.meta['section'] == match_text:
                    ret.append(tuple((META, tgt_node)))
            except KeyError:
                continue

    return ret

# extended F1000RD
def find_targets_for_fig_ix(tgt_itg: IntertextDocument, match_text: str):
    """Handles figure index matches"""
    ret = []
    META = {
        'exp': ['fig-ix']
    }

    for tgt_node in tgt_itg.nodes:
        if tgt_node.ntype == NTYPE_FIGURE:
            try:
                if tgt_node.meta['label'] == f'{match_text}':
                    ret.append(tuple((META, tgt_node)))
            except KeyError:
                continue

    return ret

# extended F1000RD
def find_targets_for_table_ix(tgt_itg: IntertextDocument,  match_text: str):
    """Handles table index matches"""
    ret = []
    META = {
        'exp': ['table-ix']
    }

    for tgt_node in tgt_itg.nodes:
        if tgt_node.ntype == NTYPE_TABLE:
            try:
                if tgt_node.meta['label'] == f'{match_text}':
                    ret.append(tuple((META, tgt_node)))
            except KeyError:
                continue

    return ret

# from F1000RD
def find_targets_for_quote(tgt_itg: IntertextDocument, match_text: str):
    """Find targets for quotes"""

    # Hard code the hyperparameters
    # The maximum Levenshtein distance for two strings to be counted as a match
    # The higher the value, the more hits are obtained (probably resulting in lower
    # precision)
    MAX_LEVENSHTEIN_DISTANCE = 2
    # The maximum number of hits for a single quote
    # This is to not bet trapped by very unspecific quotes / partial quotes such as "and",
    # which have countless hits
    # A higher number probably increases recall, but reduces precision
    MAX_N_HITS = 2

    ret = []
    META={
        'exp': ['quote']
    }

    # Split quote when it has gaps ("bla [...] bla")
    partial_quotes = split_quote(match_text)

    # For each split check if matches can be found
    for tgt_node in tgt_itg.nodes:
        if type(tgt_node) == SpanNode:
            continue

        partial_quote_idx = 0
        content_start_pos = 0
        match = True

        # Go over each partial quote, look for matches
        # If match found, for the next quote only look for matches after
        # the previously found match
        while (partial_quote_idx < len(partial_quotes)) & match:
            partial_quote = partial_quotes[partial_quote_idx]
            search_res = fuzzysearch.find_near_matches(
                partial_quote,
                tgt_node.content[content_start_pos:].lower(),
                max_l_dist=MAX_LEVENSHTEIN_DISTANCE
            )

            if len(search_res) > 0:
                top_res = sorted(search_res, key=lambda x: x.dist)[0]
                content_start_pos = top_res.end + 1
                partial_quote_idx += 1
            else:
                match = False

        if match:
            ret.append(tuple((META, tgt_node)))

            # Check if more than MAX_N_HITS matches were found for the quote
            # If yes, return no results
            if len(ret) > MAX_N_HITS:
                ret = []
                break

    return ret

def find_targets_for_ref_ix(tgt_itg: IntertextDocument,  match_text: str):
    """Find links to references"""
    ret = []
    META = {
        'exp': ['ref-ix']
    }

    for tgt_node in tgt_itg.nodes:
        if (tgt_node.ntype == NTYPE_BIB_ITEM):
            try:
                if tgt_node.meta['id'] == f'ref-{match_text}':
                    ret.append(tuple((META, tgt_node)))
            except KeyError:
                continue

    return ret


def find_targets_for_line(tgt_itg: IntertextDocument, match_text: str):
    """ Find links for line references"""
    ret = []
    META = {
        'exp': ['line']
    }

    for tgt_node in tgt_itg.nodes:
        if type(tgt_node) == SpanNode and tgt_node.ntype == "line":
            l = tgt_node.meta["line"]
            if int(l) == int(match_text):
                ret += [(META, tgt_node)]
                break  # perfect match, break
        elif type(tgt_node) == SpanNode and tgt_node.ntype == "line_range":
            ls, le = tgt_node.meta["line_start"], tgt_node.meta["line_end"]
            if int(ls) <= int(match_text) <= int(le):
                ret += [(META, tgt_node)]
                # range match, continue looking for perfect matches

    if len(ret) > 1:
        line_node = [(m, t) for (m, t) in ret if "line" in t.meta]
        if len(line_node) > 0:
            return line_node
        else:
            best_range, best_match = int(ret[0][1].meta["line_end"]) - int(ret[0][1].meta["line_start"]), ret[0]
            for m, t in ret[1:]:
                w = int(t.meta["line_end"]) - int(t.meta["line_start"])
                if w < best_range:
                    best_range = w
                    best_match = (m, t)
            return [best_match]
    else:
        return ret


# extended F1000RD
def find_anchors_for_match(doc, match):
    mtype, mtxt = match

    if mtype == 'sec-name':
        return find_targets_for_sec_name(doc, mtxt)
    elif mtype == 'sec-ix':
        return find_targets_for_sec_ix(doc, mtxt)
    elif mtype == 'fig-ix':
        return find_targets_for_fig_ix(doc, mtxt)
    elif mtype == 'table-ix':
        return find_targets_for_table_ix(doc, mtxt)
    elif mtype == 'quote':
        return find_targets_for_quote(doc, mtxt)
    elif mtype == 'ref-ix':
        return find_targets_for_ref_ix(doc, mtxt)
    elif mtype == "line":
        return find_targets_for_line(doc, mtxt)
    else:
        return None

# adapted from F1000RD/analysis/exp_linker
def find_anchors_in_paper(pointers, paper):
    links = {}
    for sec in pointers:
        pointer_types = list(set([(t, txt) for (t, txt, span) in pointers[sec]]))
        links_by_type = []

        for match in pointer_types:
            found_targets = find_anchors_for_match(paper, match)
            if found_targets is None or len(found_targets) == 0:
                links_by_type += [None]
            else:
                links_by_type += [(match, found_targets[0][0], found_targets[0][1])]

        links[sec] = [[(pt,l) for pt, l in zip(pointer_types, links_by_type) if pt[0] == pointer[0] and pt[1] == pointer[1]][0][1] for pointer in pointers[sec]]

    return links


def find_pointers_in_review(review, patterns):
    report = review["report"]

    pointers = {}
    for sec in report:
        section = report[sec]

        if section is None:
            continue

        pointers[sec] = find_pointers(section, patterns)

    return pointers


def create(dataset_type: DATASETS, bm_path=None):
    # paths
    bm_path = OUT_PATH if bm_path is None else bm_path
    dataset_path = pjoin(bm_path, dataset_type.value, "data")
    out_path =pjoin(bm_path, dataset_type.value, "annotations")

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # meta info on dataset
    collection_meta = {
        "dataset": "Benchmark Explicit Links by F1000RD REGEX Linker",
        "annotation_types": [
            "elinks"
        ]
    }
    store_collection_meta(collection_meta, out_path)

    # load patterns
    static_patterns = load_static_patterns()

    # iterate papers and reviews
    for ppath in tqdm(list_dirs(dataset_path), desc="iterating over papers"):
        pid = os.path.basename(ppath)

        for vpath in list_dirs(ppath):
            version = os.path.basename(vpath)

            if "reviews.json" not in [os.path.basename(f) for f in list_files(vpath)]:
                continue

            # input
            with open(pjoin(vpath, "paper.itg.json"), "r") as f:
                pdoc = IntertextDocument.load_json(f)

            with open(pjoin(vpath, "reviews.json"), "r") as f:
                reviews = json.load(f)

            if len(reviews) == 0:
                continue

            # do matching
            dyn_patterns = load_patterns(pdoc)

            links_per_rev = {}
            pointers_per_rev = {}
            for r in reviews:
                pointers = find_pointers_in_review(r, static_patterns + dyn_patterns)
                links = find_anchors_in_paper(pointers, pdoc)

                pointers_per_rev[r["rid"]], links_per_rev[r["rid"]] = pointers, links

            # output
            output = {}
            for rid in pointers_per_rev:
                pointers, links = pointers_per_rev[rid], links_per_rev[rid]
                output[rid] = {}

                for sec in pointers:
                    output[rid][sec] = []

                    for p, l in zip(pointers[sec], links[sec]):
                        ptype, ptxt, pspan = p

                        if l is None:
                            output[rid][sec] += [{
                                "type": ptype,
                                "rev_span": pspan,
                                "rev_text": ptxt,
                                "paper_target": None
                            }]
                        else:
                            lmatch, lmeta, lnode = l

                            output[rid][sec] += [{
                                "type": ptype,
                                "rev_span": pspan,
                                "rev_text": ptxt,
                                "paper_target": lnode.ix
                            }]

            links_out_path = pjoin(out_path, pid, version, "elinks.json")
            if not os.path.exists(pjoin(out_path, pid)):
                os.mkdir(pjoin(out_path, pid))
            if not os.path.exists(pjoin(out_path, pid, version)):
                os.mkdir(pjoin(out_path, pid, version))

            with open(links_out_path, "w+") as f:
                json.dump(output, f)


def arg_parse():
    parser = argparse.ArgumentParser(description="Creating the F1000 dataset within the benchmark")
    parser.add_argument(
        "--bm_path", type=str, required=True, help="Path to the directory of the benchmark dataset"
    )
    parser.add_argument(
        "--dataset_type", type=str, required=False, choices=[d.name for d in DATASETS], default=None, help="Dataset type"
    )

    return parser


def main(args):
    logging.basicConfig(level="INFO")

    if args.dataset_type is None:
        for d in DATASETS:
            create(d, args.bm_path)
    else:
        create(DATASETS[args.dataset_type], args.bm_path)

if __name__ == "__main__":
    parser = arg_parse()
    main(parser.parse_args())