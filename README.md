# NLPeer: A Unified Resource for the Computational Study of Peer Review

This is the official code repository for NLPeer introduced in the paper 
"NLPeer: A Unified Resource for the Computational Study of Peer Review".

The associated dataset is intended for the study of peer review and approaches to NLP-based assistance
to peer review. We stress that review author profiling violates the intended use of this dataset.
Please also read the associated dataset card.

## Quickstart
1. Install the package from github.
```bash
pip install git+https://github.com/UKPLab/nlpeer
```

2. Download the newest version of the NLPeer dataset (here)[https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/3618].

3. Load e.g. the ARR22 dataset
```python
from nlpeer import DATASETS, PAPERFORMATS, PaperReviewDataset

dataset_type = DATASETS.ARR22
paper_format = PAPERFORMATS.ITG
version = 1

# load data paperwise
data = PaperReviewDataset("<path_to_top_dir_of_nlpeer>", dataset_type, version, paper_format)

# iterate over papers with associated reviews
paperwise = [(paper_id, meta, paper, reviews) for paper_id, meta, paper, reviews in data]
```


## NLPeer Data Format

### Dataset File Structure
```             
> DATASET
    > data
      > PAPER-ID
          meta.json               = meta data on the general paper
          
          > VERSION-NUM
              paper.pdf           =  raw article pdfs of the dataset
              paper.itg           =  article in parsed ITG format
              paper.tei           =  article in prased GROBID format
              # ... 
              # more parsed paper types go here (e.g. latex)
              
              meta.json           =  metadata on the specific article version (if any)
              
              reviews.json        =  review texts with meta-data (can be empty)
          ...
          # more versions go here
      > ...
      meta.json                   = meta data on the dataset
    
    > annotations
       > PAPER-ID
          <anno>.json        = cross version annotations go here
          diff_1_2.json      = e.g. diffs
          ...
          > v1
             <anno>.json        = within-version annotations go here
             links.json         = e.g. links
             discourse.json     = e.g. discourse of reviews
          > v2
             ...
```

### Paper File Formats
#### raw.pdf
PDF as is

#### paper.itg.json
Paper parsed in ITG format. ALWAYS present.

#### paper.tei
Paper parsed in GROBID format. Should be close to ALWAYS present.

#### VERSION_NUM/meta.json
```json
{
  "title": "title of paper",
  "authors": ["author1", "author2"],
  "abstract": "abstract of paper",
  "license": "LICENSE TEXT"
}
```
Any additional fields may occur. The authors field may be empty if the paper is anonymous.

### Review File Format
```json
[
  {
    "rid": "Review ID, at least unique per paper",
    "reviewer": "Reviewer name or null if not given",
    "report": {
       "main": "main text (empty if structured; if only structured, no fields)",
       "fieldX": "textX",
       "fieldY": "textY"
    },
    "scores" : {
       "scoreX": "value",
       "scoreY": "value", 
       "overall": "value"
    },
    "meta": {
        "more metadata": "val",
        "sentences": [
          [0, 1],
          [2, 3]
        ]
    }
  }
]
```

### Standardized ITG NodeTypes
| NLPeer ITG     | F1000RD ITG   | GROBID TEI            | Semantics                           |
|----------------|---------------|-----------------------|-------------------------------------|
| title          | article-title |                       | title of the article                |
| heading        | title         | head                  | heading of a section/subsection/... |
| paragraph      | p             | p                     | paragraph of text                   |
| abstract       | abstract      | -                     | abstract of the article             |
| list           | list          | list                  | list of (enumerated) items          |
| list_item      |               | item                  | item of a list                      |
| elem_reference |               | ref(@type=figure/...) | reference to a text element         |
| bib_reference  |               | ref(@type=bibr)       | reference to a bibliography item    |
| headnote       |               | note(@place=headnote) | headnote                            |
| footnote       |               | note(@place=footnote) | footnote                            |
| figure         | label (fig)   | figure                | figure                              |
| table          | label (table) | table                 | table                               |
| formula        | label (eq)    | formula               | formula                             |
| caption        | label (*)     |                       | caption of a figure/table           |
| bib_item       | ref           |                       | bibliography entry                  |

*Note: Currently F1000 does not include specific citation spans, but only links to the bibliography or
text media from the full paragraph node. We plan to fix this in the future.*

Please check out the nlpeer/__init__.py for an overview of the node types, reviewing scales etc.

## Data Loading

### General

```python

from nlpeer import DATASETS, PAPERFORMATS, PaperReviewDataset, ReviewPaperDataset

dataset_type = DATASETS.ARR22
paper_format = PAPERFORMATS.ITG
version = 1

# load data paperwise
data = PaperReviewDataset("<path_to_top_dir_of_nlpeer>", dataset_type, version, paper_format)

# iterate over papers with associated reviews
paperwise = [(paper_id, meta, paper, reviews) for paper_id, meta, paper, reviews in data]

# use a review-wise data view
data_r = ReviewPaperDataset("<path_to_top_dir_of_nlpeer>", dataset_type, version, paper_format)

# iterate over reviews with associated paper
reviewwise = [(paper_id, meta, paper, review) for paper_id, meta, paper, review in data]
```

### Task Specific

Assuming `data` is a ReviewPaperDataset previously loaded.

**E.g. Review Score Prediction**

```python

from nlpeer.tasks import ReviewScorePredictionDataset, abstract_with_review_only

# data = initialized paper review dataset
in_transform = abstract_with_review_only()
target_transform = lambda x: x  # no normalization

rsp_data = ReviewScorePredictionDataset(data, transform=in_transform, target_transform=target_transform)
```

## Training

**E.g. Pragmatic Labeling**

```python
from nlpeer.tasks.pragmatic_labeling.data import PragmaticLabelingDataModule
from nlpeer.tasks.pragmatic_labeling.models.TransformerBased import LitTransformerPragmaticLabelingModule
from nlpeer import DATASETS
from nlpeer.tasks import get_class_map_pragmatic_labels

import torchmetrics
from torch.nn import NLLLoss
from torch.optim import AdamW

from pytorch_lightning import Trainer, seed_everything

# model
mtype = "roberta-base"
model = LitTransformerPragmaticLabelingModule(model_type=mtype,
                                              train_loss=NLLLoss(),
                                              dev_loss=torchmetrics.Accuracy(task="multiclass", num_classes=len(get_class_map_pragmatics())),
                                              optimizer=AdamW,
                                              lr=2e-5,
                                              eps=1e-8,
                                              num_labels=len(get_class_map_pragmatics()))

# preprocessing for model
input_transform, target_transform = model.get_prepare_input()
tok = model.get_tokenizer()

# general
path = "<path to nlpeer>"
dataset_type = DATASETS.ARR22
version = 1

data_loading = {
    "num_workers": 8,
    "shuffle": True
}

# load lightning data module
data_module = PragmaticLabelingDataModule(path,
                                          dataset_type=dataset_type,
                                          in_transform=input_transform,
                                          target_transform=target_transform,
                                          tokenizer=tok,
                                          data_loader_config=data_loading,
                                          paper_version=version)

# train
trainer = Trainer()

# fit the model
trainer.fit(model, data_module)
```

## Experiments

**E.g. Guided Skimming**
1. Train multiple models on different seeds
```shell
BMPATH="somepath"
WANDB_PROJ="somename"
OUTPATH="somepath"
dataset="ARR22"

python tasks/skimming/train.py --benchmark_path $BMPATH --project $WANDB_PROJ --store_results $OUTPATH --dataset $dataset --model roberta --lr 2e-5 --batch_size 3  --repeat 3
```

2. Evaluate these models
```shell
MPATH="somepath to checkpoints"

python tasks/skimming/evaluate.py --benchmark_dir $BMPATH --project $WANDB_PROJ --store_results $OUTPATH --dataset $dataset --model roberta --chkp_dir $MPATH
```

3. Process output: The output is a dict of performance measures. Check for the desired metric in the dict. The random baseline is reported along.

## Citation

Please use the following citation:

```
@inproceedings{dycke-etal-2023-nlpeer,
    title = "{NLP}eer: A Unified Resource for the Computational Study of Peer Review",
    author = "Dycke, Nils  and
      Kuznetsov, Ilia  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.277",
    pages = "5049--5073"
}

```

Contact Persons: Nils Dycke, Ilia Kuznetsov

<https://www.ukp.tu-darmstadt.de>

<https://www.tu-darmstadt.de>

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
