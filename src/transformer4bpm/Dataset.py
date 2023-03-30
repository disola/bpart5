# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""This is SAP-SAM."""

import json
import datasets

from transformer4bpm.constants import DATA_INTERIM

_DESCRIPTION = """\
https://zenodo.org/record/7012043#.Yyx2-nZByPo"""
_HOMEPAGE = "https://zenodo.org/record/7012043#.Yyx2-nZByPo"

_URL = str(DATA_INTERIM) 
_URLS = {
    "train": _URL + "/train.json",
    "validate": _URL + "/validate.json",
    "test": _URL + "/test.json"
}

class Dataset(datasets.GeneratorBasedBuilder):
    """SAP-SAM dataset."""

    def _info(self):
        features = datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    #"model_id": datasets.Value("string")
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("context", "target"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE)

    def _split_generators(self,dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": downloaded_files["validate"],
                    "split": "validate",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        key=0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            # Yields examples as (key, example) tuples
            for example in data["data"]:
                context = example["context"]
                target = example["target"]
                yield key, {
                        "context": context,
                        "target": target
                    }
                    #"second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                key+=1