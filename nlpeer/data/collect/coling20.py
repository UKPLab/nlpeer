import argparse
import json
import os
from os.path import join as pjoin
from os.path import basename
import xml.etree.ElementTree as ET

from tqdm import tqdm

from nlpeer.data.collect import retrieve_matched, fetch_camera_ready_mapping
from nlpeer.data.datasets.parse import pdf_to_tei
from nlpeer.data.datasets.utils import PDFProcessingError
from nlpeer.data.utils import list_files


def load_coling2020_papers(path):
    basepath = pjoin(path, "COLING-2020/COLING2020_XY/COLING2020_XY")

    papers_path = pjoin(basepath, "paper_pdf")
    reviews_path = pjoin(basepath, "reviews")

    reviewed_paper_ids = [int(basename(f).split("_")[0]) for f in list_files(reviews_path)]
    papers = {int(basename(f).split(".")[0]): f for f in list_files(papers_path) if int(basename(f).split(".")[0]) in reviewed_paper_ids}

    result = {}
    for pid, p_path in tqdm(papers.items(), desc="Iterating over papers"):
        status, tei = pdf_to_tei(p_path)

        if status != 200:
            raise PDFProcessingError("GROBID Parsing Error")

        root = ET.fromstring(tei)
        prefix = "{http://www.tei-c.org/ns/1.0}"

        # create article title as root
        title = root.find(f"{prefix}teiHeader/{prefix}fileDesc/{prefix}titleStmt/{prefix}title").text
        title = title if title is not None else ""

        abstract = root.find(f"{prefix}teiHeader/{prefix}profileDesc/{prefix}abstract/{prefix}div")
        if abstract:
            children = list(abstract)
            if children:
                abstract = ' '.join(abstract.itertext()).strip()
            else:
                abstract = abstract.text
        else:
            abstract = ""

        result[pid] = {
            "title": title,
            "abstract": abstract
        }

    print(f"Loaded {len(papers)} papers for matching")

    with open(pjoin(path, "extracted_titles_and_abstracts.json"), "w+") as f:
        json.dump(result, f)

    return result


def main(args):
    if not os.path.isdir(args.data_path):
        raise ValueError(f"The passed data directory {args.data_path} does not exist")

    out_path = pjoin(args.data_path, "COLING2020_camera_ready")

    # create output directories if necessary
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if args.download_approved:
        match_table = pjoin(out_path, "COLING2020_approved.csv")

        if not os.path.exists(match_table):
            raise ValueError(f"The passed output directories do not contain an approved matching table. Make sure "
                             f"that you manually verified the matches produced in the first step. Simply rename the "
                             f"file afterwards -- the resulting file path should be {match_table}.")

        retrieve_matched(match_table, outfolder=out_path)
    else:
        papers = load_coling2020_papers(args.data_path)
        fetch_camera_ready_mapping(papers, "COLING2020", out_path, "external/acl-anthology/data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for matching PeerRead's COLING2020 papers against"
                                                 "the ACL anthology. You need to call this script from the top-most"
                                                 "project directory as a working directory and have installed the"
                                                 "ACL anthology package. As this requires manual evaluation of the "
                                                 "matches, this is a two step script. Start the script without a"
                                                 "download_approved parameter to generate a matching table and after"
                                                 "approving the table (renaming it in <dataset>_approved.csv) you can"
                                                 "run the script again with the download flag set to retrieve the"
                                                 "matched PDFs from the ACL anthology.")
    parser.add_argument(
        "--data_path", type=str, help="Path to the directory containing all datasets including a directory PeerRead "
                                      "with the data to process ",
                        required=True
    )
    parser.add_argument(
        "--download_approved", default=False, required=False, type=bool, help="Downloads the matched PDFs from the "
                                                                              "anthology based on the approved paper "
                                                                              "tables in the output directories. You "
                                                                              "need to run the command twice, "
                                                                              "first without this flag and then with "
                                                                              "this flag set. "
    )

    args = parser.parse_args()

    main(args)