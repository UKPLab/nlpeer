import heapq
import json
import os
import time
import urllib

import pandas as pd
from tqdm import tqdm


if "acl-anthology" not in os.environ["PYTHONPATH"]:
    print("WARNING: The ACL anthology library might be missing from your Python path! Required for matching!")

from anthology import Anthology
from nltk.translate.bleu_score import sentence_bleu
from os.path import join as pjoin

ANTHOLOGY_DATA_PATH = os.environ.get("ACL_ANTHOLOGY_DATAPATH")
ANTHOLOGY_DATA_MAP = {
    "ACL2017": "P17",
    "CONLL2016": "K16",
    "COLING2020": "2020.coling"
}


def in_accepted_venue(accepted_at, paper_id):
    return paper_id.startswith(ANTHOLOGY_DATA_MAP[accepted_at])


def fetch_camera_ready_mapping(papers, venue, outpath, anthology_path=None):
    if anthology_path is None:
        anthology_path = ANTHOLOGY_DATA_PATH

    a = Anthology(anthology_path)

    anthology_papers = filter(lambda x: any(v for v in ANTHOLOGY_DATA_MAP.values() if x[0].startswith(v)),
                              a.papers.items())
    anthology_papers = list(map(lambda p: (
        p[0], p[1].get_title("text"), p[1].get_abstract(), p[1].as_citeproc_json()[0]["author"], p[1].pdf),
                                anthology_papers))

    misses = {}
    matches = {}

    for pid, p in tqdm(papers.items()):
        rel_papers = filter(lambda x: in_accepted_venue(venue, x[0]), anthology_papers)

        top_k = []
        max_k = 5
        for ap in rel_papers:
            apid, aptitle, apabs, apauthors, apurl = ap

            if "title" in p and p["title"] is not None:
                title_score = match_title(p["title"], aptitle)
            else:
                title_score = 0

            if "abstract" in p and p["abstract"] is not None:
                abs_score = match_abstracts(p["abstract"], apabs)
            else:
                abs_score = 0

            perfect_matched = (title_score > 0.9 and abs_score > 0.9)
            certainty = (title_score + abs_score) / 2
            uncertainty = 1 - certainty  # lower uncertainty == better == more certain

            entry = {"anthology_id": apid,
                     "url": apurl,
                     "title": aptitle,
                     "authors": apauthors,
                     "tscore": title_score,
                     "absscore": abs_score
                     }

            if perfect_matched:
                print(f"Perfect match on {pid}")
                top_k = [(uncertainty, title_score, abs_score, entry["anthology_id"], entry)]
                break
            elif uncertainty < 0.9:
                if len(top_k) < max_k:
                    try:
                        heapq.heappush(top_k, (uncertainty, title_score, abs_score, entry["anthology_id"], entry))
                    except TypeError as e:
                        # could not replace, because there is an element with the same score
                        # ignore
                        heapq.heapify(top_k)
                elif top_k[0][0] > uncertainty:
                    try:
                        heapq.heapreplace(top_k, (uncertainty, title_score, abs_score, entry["anthology_id"], entry))
                    except TypeError as e:
                        # could not replace, because there is an element with the same score
                        # ignore
                        heapq.heapify(top_k)

        if len(top_k) == 0:
            print(f"Miss on {pid}")
            misses[pid] = {
                "title": p["title"] if "title" in p else None,
                "accepted": venue
            }
        else:
            matches[pid] = {
                "title": p["title"] if "title" in p else None,
                "accepted": venue,
                "matches": [e[4] for e in top_k]
            }

    with open(os.path.join(outpath, f"{venue}_matching.json"), "w+") as file:
        json.dump(matches, file)

    if len(misses) > 0:
        with open(os.path.join(outpath, f"{venue}_missed.json"), "w+") as file:
            json.dump(misses, file)

    df = pd.DataFrame({"sid": list(papers.keys()),
                       "title": [p["title"] for p in papers.values()],
                       "match": [(matches[pid]["matches"][0]["url"] if pid in matches else "") for pid in
                                 papers],
                       "tscore": [(matches[sid]["matches"][0]["tscore"] if sid in matches else "") for sid in
                                  papers],
                       "absscore": [(matches[sid]["matches"][0]["absscore"] if sid in matches else "") for sid in
                                    papers],
                       "other_matches": [(len(matches[sid]["matches"]) if sid in matches else "") for sid in
                                         papers]
                       })
    df.to_csv(os.path.join(outpath, f"{venue}_to_be_approved.csv"))


def match_abstracts(absPR, absACL):
    absPR = absPR.strip().lower()
    absACL = absACL.strip().lower()

    if absPR == absACL:
        return 1
    else:
        return sentence_bleu([absPR.split(" ")], absACL.split(" "), weights=(1 / 3, 1 / 3, 1 / 3))  # uses 3-grams


def match_title(titlePR, titleACL):
    titlePR = titlePR.strip().lower()
    titleACL = titleACL.strip().lower()

    if titlePR == titleACL:
        return 1
    else:
        return sentence_bleu([titlePR.split(" ")], titleACL.split(" "), weights=(2 / 3, 1 / 3))  # uses pairs


def aggregate_mappings(paths):
    res = []

    for p in paths:
        df = pd.read_csv(p)
        sid_to_match = df[["sid", "match"]]
        res += [sid_to_match]

    res = pd.concat(res, ignore_index=True)

    return list(res.transpose().to_dict().values())


def get_pdf(url, out):
    for i in range(4):
        try:
            return urllib.request.urlretrieve(url, out)
        except urllib.error.URLError as e:
            print(e)
            time.sleep(3)

    print(f">> Failed on {url}")


def retrieve_matched(mapping_file, outfolder):
    sid_to_aclurl = aggregate_mappings([mapping_file])
    for e in tqdm(sid_to_aclurl, desc="Iterating over matching pairs"):
        sid = e["sid"]
        aclurl = e["match"]

        get_pdf(aclurl, pjoin(outfolder, f"{sid}.pdf"))

    pd.DataFrame.from_records(sid_to_aclurl).to_csv(pjoin(outfolder, "sid_to_url.csv"))
