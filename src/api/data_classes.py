import copy
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import api.util as util


class Distribution:
    def __init__(self, dir: str):
        self.dir = dir
        if not os.path.exists(dir):
            raise ValueError(f"Distribution directory {dir} does not exist")

        training_examples = util.load_json(f"{dir}/train.json")
        test_examples = util.load_json(f"{dir}/test.json")

        self.id = os.path.basename(dir)

        self.training_dataset = MCDataset(
            training_examples,
            distribution_id=self.id,
            data_dir=f"{self.dir}/train.json",
        )
        self.test_dataset = MCDataset(
            test_examples, distribution_id=self.id, data_dir=f"{self.dir}/test.json"
        )

    def create_hps_copy(self, num_train, num_eval, dir):
        if num_train + num_eval > len(self.training_dataset):
            raise Exception(
                f"num_train + num_eval ({num_train + num_eval}) must be less than the number of training examples ({len(self.training_dataset)})"
            )
        hps_train_examples = self.training_dataset.examples[:num_train]
        hps_eval_examples = self.training_dataset.examples[
            num_train : num_train + num_eval
        ]
        util.save_json(hps_train_examples, f"{dir}/train.json")
        util.save_json(hps_eval_examples, f"{dir}/test.json")
        return Distribution(dir)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class MCDataset(Dataset):
    @staticmethod
    def validate_example(
        example, desired_num_responses=None, desired_num_responses_warning_only=True
    ):
        if "prompt" not in example:
            raise Exception(f"Example does not have a prompt:\n{example}")
        if "responses" not in example:
            raise Exception(f"Example does not have responses:\n{example}")
        best_response = [
            r for r in example["responses"] if example["responses"][r] == 1
        ]
        if len(best_response) == 0:
            raise Exception(
                f"Example does not have a preferred response, i.e. no values are 1:\n{example}"
            )
        if len(best_response) > 1:
            raise Exception(
                f"Example has more than one preferred response, i.e. multiple values are 1:\n{example}"
            )
        worse_responses = [
            r for r in example["responses"] if example["responses"][r] == 0
        ]
        if len(worse_responses) == 0:
            raise Exception(
                f"Example does not have any dispreferred responses, i.e. no values are 0:\n{example}"
            )
        num_responses_for_example = len(best_response) + len(worse_responses)
        if (
            desired_num_responses != None
            and num_responses_for_example != desired_num_responses
        ):
            raise Exception(
                f"Example does not have the desired number of responses ({desired_num_responses}):\n{example}"
            )

    def validate_examples(self, desired_num_responses=None):
        if len(self.examples) == 0:
            print(f"WARNING: No examples in dataset {self.data_dir}.")
            return
        num_responses = len(list(self.examples[0]["responses"].keys()))
        for example in self.examples:
            MCDataset.validate_example(
                example,
                desired_num_responses=desired_num_responses
                if desired_num_responses != None
                else num_responses,
                desired_num_responses_warning_only=(desired_num_responses == None),
            )

    def __init__(self, examples, distribution_id, data_dir, max_examples=None):
        self.data_dir = data_dir
        self.distribution_id = distribution_id
        self.examples = examples
        self.validate_examples()
        self.max_examples = max_examples

    def __len__(self):
        if self.max_examples != None:
            return min(len(self.examples), self.max_examples)
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def set_max_examples(self, max_examples):
        self.max_examples = max_examples
        if max_examples != None:
            self.examples = self.examples[:max_examples]

    def convert_to_pairs(self, one_pair_per_instruction=False):
        new_examples = []
        for example in self.examples:
            best_response = [
                r for r in example["responses"] if example["responses"][r] == 1
            ][0]
            worse_responses = [
                r for r in example["responses"] if example["responses"][r] == 0
            ]
            response_pairs = [
                {best_response: 1, worse_response: 0}
                for worse_response in worse_responses
            ]
            if one_pair_per_instruction:
                if len(response_pairs) == 0:
                    print(example)
                pair = random.choice(response_pairs)
                new_examples.append(
                    {
                        "prompt": example["prompt"],
                        "responses": pair,
                    }
                )
            else:
                new_examples.extend(
                    [
                        {"prompt": example["prompt"], "responses": response_pair}
                        for response_pair in response_pairs
                    ]
                )
        self.examples = new_examples

    def filter_out_long_examples(self, tokenizer: PreTrainedTokenizer):
        filtered_examples = []
        for example in self.examples:
            tokenized = [
                tokenizer.encode(example["prompt"] + r + tokenizer.eos_token)
                for r in example["responses"]
            ]
            if max([len(t) for t in tokenized]) <= tokenizer.model_max_length:
                filtered_examples.append(example)
        num_examples_filtered_out = len(self.examples) - len(filtered_examples)

        if num_examples_filtered_out > 0:
            util.print_once(
                f"Filtered out {num_examples_filtered_out} examples because they exceeded the max length of {tokenizer.model_max_length}"
            )

        self.examples = filtered_examples
        return filtered_examples


@dataclass
class MCDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    # Returns a dict with the keys inputs, labels, attention_mask, and scores. Each key is an array that represents the flattened MC options. scores is a list of lists. Each sublist represents options for a single example and the values of are the scores for the response.
    def __call__(self, instances: Sequence[Dict]):
        response_labels = torch.tensor(
            [list(instance["responses"].values()) for instance in instances]
        )
        instances = [
            [
                {"prompt": instance["prompt"], "response": response}
                for response in instance["responses"]
            ]
            for instance in instances
        ]

        collator = SupervisedDataCollator(tokenizer=self.tokenizer)
        model_inputs = {}
        model_inputs["example_inputs"] = [collator(response) for response in instances]
        model_inputs.update({"response_labels": response_labels})

        return model_inputs


@dataclass
class SupervisedDataCollator:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    @staticmethod
    def tokenize_prompts_and_responses(
        prompts: Sequence[str],
        responses: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        prompts_tokenized = []
        responses_tokenized = []
        prompt_lens = []
        for prompt, response in zip(prompts, responses):
            if prompt == "":
                prompt_tokenized = torch.tensor([], dtype=torch.int64)
            else:
                prompt_tokenized = tokenizer(
                    prompt, return_tensors="pt", padding=False, add_special_tokens=False
                ).input_ids[0]
            response_tokenized = tokenizer(
                response, return_tensors="pt", padding=False, add_special_tokens=False
            ).input_ids[0]

            responses_tokenized.append(response_tokenized)
            prompts_tokenized.append(prompt_tokenized)

        input_ids = [
            torch.cat((s, t), dim=0)
            for s, t in zip(prompts_tokenized, responses_tokenized)
        ]
        prompt_lens = [len(s) for s in prompts_tokenized]
        labels = copy.deepcopy(input_ids)
        for label, prompt_len in zip(labels, prompt_lens):
            label[:prompt_len] = util.IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [instance["prompt"] for instance in instances]
        responses = [instance["response"] for instance in instances]

        suffix = self.tokenizer.eos_token
        responses = [f"{response}{suffix}" for response in responses]

        data_dict = self.tokenize_prompts_and_responses(
            prompts, responses, self.tokenizer
        )

        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        def pad_on_left(sequences, pad_id):
            max_seq_length = max(len(seq) for seq in sequences)
            padded_input_ids = [
                torch.cat(
                    [
                        torch.tensor(
                            [pad_id] * (max_seq_length - len(seq)) + seq.tolist()
                        )
                    ]
                )
                for seq in sequences
            ]
            padded_input_ids = torch.stack(padded_input_ids)
            return padded_input_ids

        input_ids = pad_on_left(input_ids, self.tokenizer.pad_token_id)
        labels = pad_on_left(labels, util.IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
