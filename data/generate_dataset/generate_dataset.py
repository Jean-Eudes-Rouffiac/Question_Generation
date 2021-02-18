# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
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

# Lint as: python3

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import nltk
nltk.download('punkt')

import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--path_to_data", type=str, help="path to json files.")

args = parser.parse_args()


QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]


def process_e2e_qg(paragraph):
    source_text = f"generate questions: {paragraph['context'].strip()}"
    questions = [qas['question'].strip() for qas in paragraph['qas']]
    target_text = " {sep_token} ".join(questions)
    target_text = f"{target_text} {{sep_token}}"
    return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

def _generate_examples(filepath, qg_format):
    """This function returns the examples in the raw (text) form."""
    logging.info("generating examples from = %s", filepath)
    count = 0
    tasks = ['e2e_qg']

    with open(filepath, encoding="utf-8") as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()

                yield count, process_e2e_qg(paragraph)
                count += 1


fquad_generator = _generate_examples(os.path.join(args.path_to_data, "train.json"), "highlight")
sources, targets, tasks = [], [], []
elem = next(fquad_generator, None)
while elem is not None:
    sources.append(elem[1]['source_text'])
    targets.append(elem[1]['target_text'])
    tasks.append(elem[1]['task'])
    elem = next(fquad_generator, None)
df_train_fquad = pd.DataFrame({"source_text": sources, 'target_text': targets, "task": tasks})

piaf_generator = _generate_examples(os.path.join(args.path_piaf_data, "piaf-v1.1.json"), "highlight")
sources, targets, tasks = [], [], []
elem = next(piaf_generator, None)
while elem is not None:
    sources.append(elem[1]['source_text'])
    targets.append(elem[1]['target_text'])
    tasks.append(elem[1]['task'])
    elem = next(piaf_generator, None)
df_train_piaf = pd.DataFrame({"source_text": sources, 'target_text': targets, "task": tasks})

df_mix = pd.concat([df_train_fquad, df_train_piaf])
df_mix = df_mix.reset_index(drop=True)
df_mix.to_csv("data/train.csv")

fquad_generator = _generate_examples(os.path.join(args.path_to_data, "valid.json"), "highlight")
sources, targets, tasks = [], [], []
elem = next(fquad_generator, None)
while elem is not None:
    sources.append(elem[1]['source_text'])
    targets.append(elem[1]['target_text'])
    tasks.append(elem[1]['task'])
    elem = next(fquad_generator, None)
df_valid = pd.DataFrame({"source_text": sources, 'target_text': targets, "task": tasks})
df_valid.to_csv("data/valid.csv")
