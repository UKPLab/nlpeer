import argparse
import json
import os

from nlpeer.data.collect import retrieve_matched, fetch_camera_ready_mapping
from nlpeer.data.utils import list_files
from os.path import join as pjoin


def load_peerread_papers(datapath):
    papers = {}

    for s in ["train", "dev", "test"]:
        s_path = pjoin(datapath, s)

        parsed_path = pjoin(s_path, "reviews")

        for p in list_files(parsed_path):
            pid = os.path.basename(p)[:-len('.json')]

            with open(pjoin(parsed_path, f"{pid}.json"), "r") as jf:
                reviews_for_paper = json.load(jf)

            papers[pid] = {"title": reviews_for_paper["title"], "abstract": reviews_for_paper["abstract"]}

    print(f"Loaded {len(papers)} papers for matching")

    return papers


def main(args):
    if not os.path.isdir(args.data_path):
        raise ValueError(f"The passed data directory {args.data_path} does not exist")

    if not os.path.isdir(pjoin(args.data_path, "PeerRead")):
        raise ValueError(
            f"The passed data directory {args.data_path} does not contain the peer read corpus under PeerRead")

    acl17_path = pjoin(args.data_path, "PeerRead", "data", "acl_2017")
    conll16_path = pjoin(args.data_path, "PeerRead", "data", "conll_2016")

    if not os.path.exists(acl17_path) or not os.path.exists(conll16_path):
        raise ValueError(
            f"The paths for the ACL17 and CONLL16 don't exist on these paths: {acl17_path}, {conll16_path}")

    acl17_out_path = pjoin(args.data_path, "ACL2017_camera_ready")
    conll16_out_path = pjoin(args.data_path, "CONLL2016_camera_ready")

    # create output directories if necessary
    if not os.path.exists(acl17_out_path):
        os.mkdir(acl17_out_path)

    if not os.path.isdir(conll16_out_path):
        os.mkdir(conll16_out_path)

    if args.download_approved:
        acl_match_table = pjoin(acl17_out_path, "ACL2017_approved.csv")
        conll_match_table = pjoin(conll16_out_path, "CONLL2016_approved.csv")

        if not os.path.exists(acl_match_table) or not os.path.exists(conll_match_table):
            raise ValueError(f"The passed output directories do not contain an approved matching table. Make sure "
                             f"that you manually verified the matches produced in the first step. Simply rename the "
                             f"file afterwards -- the resulting file paths should be {acl_match_table} and "
                             f"{conll_match_table}")

        retrieve_matched(acl_match_table, outfolder=acl17_out_path)
        retrieve_matched(conll_match_table, outfolder=conll16_path)
    else:
        acl17_papers = load_peerread_papers(acl17_path)
        fetch_camera_ready_mapping(acl17_papers, "ACL2017", acl17_out_path, "external/acl-anthology/data")

        conll16_papers = load_peerread_papers(conll16_path)
        fetch_camera_ready_mapping(conll16_papers, "CONLL16", conll16_out_path, "external/acl-anthology/data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for matching PeerRead's ACL2017 and CONLL16 papers against"
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
                                      "with the data to process "
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
