# IMPORTS:
import pandas as pd

import time
import random
random.seed(2020)
import os
import regex as re
import torch
import tqdm

from transformers import (
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration
    )

from transformers_utils import (
    generate_questions
    )

import argparse
import json
import copy

# ARGUMENT PARSING

parser = argparse.ArgumentParser()

# Compulsory arguments
parser.add_argument("--file_data", help="name of the csv file containing the contexts", type=str)
parser.add_argument("--output_dir", help="name of the directory where to export the generated questions", type=str)
parser.add_argument("--file_name", help="name of the output csv file that will be saved", type=str)
parser.add_argument("--checkpoint_path", help='name or path of where to find the checkpoint folder', type=str, default=None, action="store")
parser.add_argument("--tokenizer_name_or_path", help='name or path of where to find the tokenizer', type=str, default='t5_qg_tokenizer', action="store")
# Optional arguments
parser.add_argument("--max_length_input", help="max length of input sequence, default 256", type=int, default=512, action="store")
parser.add_argument("--max_length_output", help="max_length of output sequence, defaut 50", type=int, default=50, action="store")
parser.add_argument("--batch_size", help="batch size for training, default 16", type=int, default=16, action='store')
parser.add_argument("--repetition_penalty", help='repetition penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("--length_penalty", help='length penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("--num_beams", help="number of beams, parameter for generation, default 1", type=int, default=1, action="store")
parser.add_argument("--temperature", help="temperature parameter for softmax in generation, default 1.0", type=float, default=1.0, action="store")

args = parser.parse_args()


def add_string(contexts, string):
        context_bis = []
        for text in contexts:
            context_bis.append(string + text)
        return context_bis


def main(args_file=None):

    # LOADING MODEL & TOKENIZER:
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    print("Loading model and tokenizer...", end="", flush=True)

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

    config = T5Config.from_json_file(args.checkpoint_path + "/config.json")
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path + "/pytorch_model.bin", config=config)

    print("Done.")


    model.to(device)

    # LOADING & PREPROCESSING DATA
    print("Loading and preprocessing data...", end="", flush=True)
    df_generation = pd.read_csv(args.file_data)
    print(df_generation.head())

    # GENERATION
    print("Generating...")
    max_length_seq = args.max_length_input
    max_length_label = args.max_length_output

    contexts = df_generation["context"].tolist()

    generation_hyperparameters = {
        'min_length': 5,
        'max_length': max_length_label,
        'repetition_penalty': args.repetition_penalty,
        'length_penalty': args.length_penalty,
        'num_beams': args.num_beams,
        'temperature': args.temperature,
    }

    nb_batch = len(contexts)//args.batch_size

    generated_questions = []

    index_for_context = 0

    for i in tqdm.tqdm(range(nb_batch)):
        batch_contexts = contexts[i*args.batch_size:(i+1)*args.batch_size] if i != nb_batch - 1 else contexts[i*args.batch_size:]
        batch_hl_contexts = add_string(batch_contexts, "generate questions: ")
        if len(batch_hl_contexts) > 0:
            list_inputs = tokenizer.batch_encode_plus(batch_hl_contexts, padding=True, max_length=max_length_seq, truncation=True)
            input_ids = torch.tensor(list_inputs["input_ids"], ).to(device)
            attention_mask = torch.tensor(list_inputs["attention_mask"]).to(device)
            generation_hyperparameters["input_ids"] = input_ids
            generation_hyperparameters["attention_mask"] = attention_mask
            batch_generated_tokens = model.generate(**generation_hyperparameters)
            batch_generated_questions = tokenizer.batch_decode(batch_generated_tokens)
            for j in range(len(batch_generated_questions)):
                final_generated = []
                generated_without_space = []
                generated = re.split("<[^<]*>", batch_generated_questions[j])
                for l in generated :
                    if len(l)>0:
                        generated_without_space.append(l.strip())
                for m in generated_without_space:
                    if len(m)>0:
                        if m[-1]== "?":
                            final_generated.append(m)
                batch_generated_questions[j] = list(set(final_generated))
            generated_questions += batch_generated_questions

    # SAVING
    dict_to_save = {}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df_ = pd.DataFrame({"context": contexts, "questions": generated_questions})
    df_.to_excel(os.path.join(args.output_dir, args.file_name)+ ".xlsx", index = False , encoding="utf-8")


def run_generate(args_dict):
    with open("args_generate.json", 'w') as f:
        json.dump(args_dict, f)

    main(args_file="args_generate.json")

if __name__ == "__main__":
    main()
