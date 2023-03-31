# BPART5

This is the source code for the experiments of the paper

*Activity Recommendation for Business Process Modeling with Pre-trained Language Models*

by Diana Sola, Han van der Aa, Christian Meilicke, and Heiner Stuckenschmidt.

Link to the paper: tbd.

## License

The source code in this repository is licensed as follows. 
```
Copyright (c) 2022 by SAP.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Setup

The python package requirements for this project are specified in `./requirements.txt`. We used Python 3.8. 

To reproduce our experiments, you need to download pre-processed data and the fine-tuned model from [here](https://drive.google.com/drive/folders/1FDiznHldAz9QedCqacav2YGoD_eTWDDz?usp=share_link).
Place the files in `./data/interim/` and `./models/t5small_finetuned/checkpoint-138000/` into the corresponding folders in this repo, respectively.

To make everything reproducible, we provide:
- the data split (processes_split.json), 
- verbalized sequences (train.json, valid.json, test.json), 
- simulated models for the evaluation cases (simulated_models.json),
- and the evaluation cases (recommendation_cases.json).

To perform new experiments, you need to download the [SAP-SAM dataset](https://zenodo.org/record/7012043#.Y5u_yXbMKPo) and place the csv files into the folder `./data/raw/sapsam_2022/csv`.
For more information on SAP-SAM, see [this repo](https://github.com/signavio/sap-sam) or [this paper](https://arxiv.org/abs/2208.12223).

## Getting started

To reproduce the training, you can run `training.py` in `./src`.

To reproduce the recommendation and evaluation, you can run `recommend_and_evaluate.py` in `./src`.

For both, training and recommendation, we used the fine-tuned T5 model in `./models/t5small_finetuned/checkpoint-138000/`.

The recommendation and evaluation results for both, the [rule-based method](https://hanvanderaa.com/wp-content/uploads/2022/04/IS2022-Exploiting-label-semantics-for-rule-based-activity-recommendation-in-business-process-modeling.pdf) from our earlier work and BPART5, can be found in `./reports/recommendation/`.
The files that end in `_rec_cases.json` contain the generated recommendation lists per evaluation case.
The files that end in `_evaluation.json` contain the evaluation results, i.e., the values of the metrics, per case and over all cases.

In `notebooks/evaluation.ipynb`, you can reproduce the analysis of the experimental results that can be found in the paper.

If you want to change settings in the data pre-processing, please check `args.py` in `./src/transformer4bpm/`.

## Project Organization

    ├── data
    │   ├── interim           <- Intermediate data should be placed in this folder.
    │   └── raw               <- The raw dataset should be placed in this folder.
    ├── notebooks             <- Jupyter notebooks.
    ├── reports            
    │   └── figures           <- Generated graphics and figures used in the paper.
    |   └── recommendation    <- Recommendation and evaluation results for BPART5 and the rule-based method from our earlier work.
    ├── src                   <- Source code for use in this project.
    │   └── transformer4bpm   <- More source code.
    ├── LICENSE               <- License that applies to the source code in this repository.
    ├── README.md             <- The top-level README for developers using this project.
    ├── requirements.txt      <- Python package requirements for this project.
    └── setup.py              <- Makes project pip installable (pip install -e .) such that src can be imported.
