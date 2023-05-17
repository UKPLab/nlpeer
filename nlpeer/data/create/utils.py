import logging
import os
import re
import shutil
from os.path import join as pjoin

import spacy
from spacy import Language
from spacy.symbols import  ORTH
from intertext_graph.itsentsplitter import SpacySplitter


class PDFProcessingError(BaseException):
    def __init__(self, cause, ex=None):
        self.cause = cause
        self.underlying_exception = ex


def create_output_data_path(benchmark_dir_path, dataset_name):
    out_path = pjoin(benchmark_dir_path, dataset_name)

    if not os.path.exists(out_path):
        logging.info(f"Creating {out_path}")
        os.mkdir(out_path)

    # create out path data
    out_path = pjoin(out_path, "data")
    if os.path.exists(out_path):
        logging.info(f"WARNING: Deleting old data directory {out_path}")
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    return out_path


def sentence_split_review(review, splitter=None):
    if splitter is None:
        splitter = SpacySplitter(spacy.load('en_core_sci_sm'))

    res = {}
    for field, text in review["report"].items():
        if text is not None:
            res[field] = splitter.split(text)

    review["meta"]["sentences"] = res

    return review


def get_review_sentences(review):
    res = {}
    for field, sents in review["meta"]["sentences"].items():
        txt = review["report"][field]

        res[field] = [txt[s[0]:s[1]] for s in sents]

    return res


@Language.component("linebreak_component")
def _split_on_special_token(doc):
    for token in doc[:-1]:
        if token.text == "<br>":
            doc[token.i + 1].is_sent_start = True

    return doc


def clean_and_split_review(review):
    # setup special splitting
    def augmented_clean(txt):
        res = txt.strip()  # strip unnecessary whitespaces

        res = re.sub(r"\n{2,}", " <br> ", res)  # clear line break
        res = re.sub(r" <br> [*\-] ", " <br> - ", res)  # replaced line break with an itemize
        res = re.sub(r"\n[*\-] ", " <br> - ", res)  # non-replaced line break with an itemize
        res = re.sub(r"^\* ", "- ", res)

        return res

    # splitter
    nlp = spacy.load('en_core_sci_sm', exclude=["ner", "tagger", "parser", "lemmatizer"])
    nlp.add_pipe("sentencizer")

    nlp.tokenizer.add_special_case("<br>", [{ORTH: "<br>"}])
    nlp.add_pipe("linebreak_component", name="linebreaking", first=True)

    sentences = {}
    report = {}
    for field, text in review["report"].items():
        if text is not None:
            cleaned = augmented_clean(text)
            processed = nlp(cleaned)

            # get sentences and replace the <br> parts by actual line breaks
            new_text = [s.text for s in processed.sents]
            for i, t in enumerate(new_text):
                if t == "<br>" and i > 0:
                    new_text[i-1] = new_text[i-1] + "\n"

            # update sentences to exclude <br> ones and replace any br left in a sentence
            new_text = [t.replace("<br>", "") for t in new_text if t != "<br>"]

            # if line breaks do not occur at the beginning or ending of a sentence, they should be discarded
            tmp = []
            for t in new_text:
                if len(t) > 1:
                    e = t[0]+ t[1:-1].replace("\n", " ") + t[-1]
                else:
                    e = t

                if e[-1] not in [" ", "\n", "\t"]:
                    e += " "

                tmp += [e]

            new_text = tmp
            report[field] = "".join(new_text)

            sentences[field] = []
            ix = 0
            for s in new_text:
                sentences[field] += [(ix, ix+len(s))]
                ix += len(s)

    review["meta"]["sentences"] = sentences
    review["report"] = report

    return review