import argparse
import copy
import functools
import os
from typing import List, Optional

import datasets as hf_datasets
import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk

# make sure to install promptsource, transformers, and datasets!
from promptsource.templates import DatasetTemplates
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import api.util as util

from api.data_classes import Distribution
from api.model import Model


def main(
    model_dir: str,
    output_dir: str,
    training_distribution_dir: str,
    test_distribution_dirs: List[str],
    max_examples: int = 100,
):
    train_dataset = Distribution(training_distribution_dir).training_dataset
    if max_examples != None:
        train_dataset.set_max_examples(max_examples)
    train_dataset.convert_to_pairs(one_pair_per_instruction=True)
    neg_hs, pos_hs, y = get_hs(model_dir, train_dataset)

    def save_generations(generation, save_dir, generation_type):
        """
        Input:
            generation: numpy array (e.g. hidden_states or labels) to save
            args: arguments used to generate the hidden states. This is used for the filename to save to.
            generation_type: one of "negative_hidden_states" or "positive_hidden_states" or "labels"

        Saves the generations to an appropriate directory.
        """
        # construct the filename based on the args
        # arg_dict = vars(args)
        # exclude_keys = ["save_dir", "cache_dir", "device"]
        # filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)
        filename = generation_type + ".npy"

        # create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save
        np.save(os.path.join(save_dir, filename), generation)

    save_generations(neg_hs, output_dir, generation_type="negative_hidden_states")
    save_generations(pos_hs, output_dir, generation_type="positive_hidden_states")
    save_generations(y, output_dir, generation_type="labels")
    n = len(y)
    neg_hs_train, neg_hs_test = neg_hs[: n // 2], neg_hs[n // 2 :]
    pos_hs_train, pos_hs_test = pos_hs[: n // 2], pos_hs[n // 2 :]
    y_train, y_test = y[: n // 2], y[n // 2 :]

    # for simplicity we can just take the difference between positive and negative hidden states
    # (concatenating also works fine)
    x_train = neg_hs_train - pos_hs_train
    x_test = neg_hs_test - pos_hs_test

    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)

    args = {
        "base_model_dir": model_dir,
    }
    util.save_json(args, output_dir + "/configuration.json")
    print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))


def get_hs(model_dir, dataset):
    hf_classification_format = {"test": []}
    for example in dataset.examples:
        for response in example["responses"]:
            hf_classification_format["test"].append(
                {
                    "content": example["prompt"] + response,
                    "label": example["responses"][response],
                }
            )
    data = hf_datasets.Dataset.from_dict(hf_classification_format)["test"]

    cache_dir = None
    layer = -1

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        max_memory = {0: "{0}MB".format(70 * 1024), 1: "0MB"}
    else:
        max_memory = {1: "{0}MB".format(70 * 1024), 0: "0MB"}
    device = f"cuda:{local_rank}"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, cache_dir=cache_dir, quantization_config=bnb_config
    )
    # model.cuda()
    model_type = "decoder"

    def get_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
        """
        Given a decoder model and some text, gets the hidden states (in a given layer, by default the last) on that input text

        Returns a numpy array of shape (hidden_dim,)
        """
        # tokenize (adding the EOS token this time)
        input_ids = tokenizer(
            input_text + tokenizer.eos_token, return_tensors="pt"
        ).input_ids.to(model.device)

        # forward pass
        with torch.no_grad():
            output = model(input_ids, output_hidden_states=True)

        # get the last layer, last token hidden states
        hs_tuple = output["hidden_states"]
        hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

        return hs

    def get_hidden_states(model, tokenizer, input_text, layer=-1, model_type="decoder"):
        return get_decoder_hidden_states(model, tokenizer, input_text, layer=layer)

    def format_imdb(text, label):
        """
        Given an imdb example ("text") and corresponding label (0 for negative, or 1 for positive),
        returns a zero-shot prompt for that example (which includes that label as the answer).

        (This is just one example of a simple, manually created prompt.)
        """
        return (
            text
            + [
                "\n\n### Does the responder successfully follow the instruction?\nYes",
                "\n\n###Does the responder successfully follow the instruction?\nNo",
            ][label]
        )  # 62%

    def get_hidden_states_many_examples(model, tokenizer, data, model_type, n=100):
        """
        Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
        Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
        with the ground truth labels

        This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
        """
        # setup
        model.eval()
        all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

        # loop
        for _ in tqdm(range(n)):
            # for simplicity, sample a random example until we find one that's a reasonable length
            # (most examples should be a reasonable length, so this is just to make sure)
            while True:
                idx = np.random.randint(len(data))
                text, true_label = data[idx]["content"], data[idx]["label"]
                # the actual formatted input will be longer, so include a bit of a marign
                if len(tokenizer(text)) < 400:
                    break

            # get hidden states
            neg_hs = get_hidden_states(
                model,
                tokenizer,
                format_imdb(text, 0),
                model_type=model_type,
                layer=layer,
            )
            pos_hs = get_hidden_states(
                model,
                tokenizer,
                format_imdb(text, 1),
                model_type=model_type,
                layer=layer,
            )

            # collect
            all_neg_hs.append(neg_hs)
            all_pos_hs.append(pos_hs)
            all_gt_labels.append(true_label)

        all_neg_hs = np.stack(all_neg_hs)
        all_pos_hs = np.stack(all_pos_hs)
        all_gt_labels = np.stack(all_gt_labels)

        return all_neg_hs, all_pos_hs, all_gt_labels

    neg_hs, pos_hs, y = get_hidden_states_many_examples(
        model, tokenizer, data, model_type, n=len(data)
    )
    return neg_hs, pos_hs, y


if __name__ == "__main__":
    fire.Fire(main)
