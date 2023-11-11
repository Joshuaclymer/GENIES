import transformers
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer
from api.model import Model
import re
import copy
from torch.utils.data import Dataset
import api.util as util
import torch
from torch.utils.data import DataLoader
from typing import Dict, Sequence, Any, List, Optional
from dataclasses import dataclass


class Distribution:
    def __init__(self, dir: str):
        self.dir = dir
        training_examples = util.load_json(f"{dir}/train.json")
        test_examples = util.load_json(f"{dir}/test.json")
        meta_data = util.load_json(f"{dir}/meta_data.json")

        self.formats = meta_data["formats"]
        self.id = meta_data["id"]

        # Add prompts for each example
        for example in training_examples:
            self.format_example(self.formats, example)
        for example in test_examples:
            self.format_example(self.formats, example)

        self.training_dataset = MCDataset(
            training_examples, distribution_id=self.id, distribution_dir=self.dir
        )
        self.test_dataset = MCDataset(
            test_examples, distribution_id=self.id, distribution_dir=self.dir
        )

    @staticmethod
    def format_example(formats: List[str], example: dict) -> dict:
        all_format_keys = [re.findall(r"\{(.+?)\}", format) for format in formats]
        all_format_keys = set([key for keys in all_format_keys for key in keys])
        example_keys = set(example.keys())
        example_format_keys = example_keys.intersection(all_format_keys)

        for format in formats:
            keys_for_format_string = set(re.findall(r"\{(.+?)\}", format))
            if keys_for_format_string == example_format_keys:
                example["prompt"] = format.format(**example)
                return example
        raise Exception(
            f"An example could not be matched to a format string.\nExample: {example}\nFormat strings: {formats}"
        )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class MCDataset(Dataset):
    def __init__(self, examples, distribution_id, distribution_dir, max_examples=None):
        self.distribution_dir = distribution_dir
        self.distribution_id = distribution_id
        self.examples = examples
        for i, example in enumerate(self.examples):
            example["id"] = i
        self.max_examples = max_examples

    def __len__(self):
        if self.max_examples != None:
            return min(len(self.examples), self.max_examples)
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def set_max_examples(self, max_examples):
        self.max_examples = max_examples
        self.examples = self.examples[:max_examples]

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

    def get_example_scores(
        self, model: Model, accelerator: Optional[Accelerator], per_device_batch_size
    ) -> List[dict]:
        # First filter out examples that exceed the models maximum sequence lenght
        self.filter_out_long_examples(model.tokenizer)

        dataloader = DataLoader(
            self,
            batch_size=per_device_batch_size,
            shuffle=False,
            collate_fn=MCDataCollator(tokenizer=model.tokenizer),
        )

        model.hf_model.eval()
        model.hf_model, dataloader = accelerator.prepare(model.hf_model, dataloader)

        example_scores = []  # a list of tuples. Each tuple is (score, example id)
        for batch in tqdm(dataloader):
            batch_scores = self.get_scores_for_batch(model.hf_model, batch)
            ids = batch.pop("ids")
            example_scores.extend(list(zip(batch_scores, ids)))

        # Gather tensors from different devices
        example_scores = torch.tensor(example_scores, dtype=torch.float32).to(model.hf_model.device)
        example_scores = accelerator.gather(example_scores).cpu()

        accelerator.free_memory()

        # Reorder flattened scores to ensure indices match
        ordered_example_scores = [None] * len(self.examples)
        for i in range(example_scores.shape[0]):
            ordered_example_scores[int(example_scores[i, 1])] = float(example_scores[i, 0])
        return ordered_example_scores

    @staticmethod
    def get_scores_for_batch(model, batch):
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = output.logits

        # Shift so that tokens < n predict n
        logits_for_batch = logits[:, :-1, :]
        shift_labels_batch = batch["labels"][:, 1:]

        # Flatten the tokens
        shift_logits_batch = logits_for_batch.contiguous().view(-1, logits.shape[-1])
        shift_labels_batch = shift_labels_batch.contiguous().view(-1)

        # Cross-entropy loss
        loss_fct = CrossEntropyLoss(reduction="none")  # use "none" to get loss per item
        batch_losses = loss_fct(shift_logits_batch, shift_labels_batch)
        avg_log_probs = -batch_losses.view(logits.size(0), -1).mean(dim=1)

        # Unflatten avg log probs
        response_labels = batch["response_labels"]
        response_labels = response_labels.to(model.device).to(dtype=avg_log_probs.dtype)
        response_labels.requires_grad = True
        avg_log_probs_unflattened = avg_log_probs.contiguous().view(
            response_labels.shape[0], response_labels.shape[1]
        )
        response_probabilities = torch.softmax(avg_log_probs_unflattened, dim=1).to(model.device)
        example_scores = (response_probabilities * response_labels).sum(dim=1)
        return example_scores


@dataclass
class MCDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    # Returns a dict with the keys inputs, labels, attention_mask, and scores. Each key is an array that represents the flattened MC options. scores is a list of lists. Each sublist represents options for a single example and the values of are the scores for the response.
    def __call__(self, instances: Sequence[Dict]):
        response_labels = torch.tensor(
            [list(instance["responses"].values()) for instance in instances]
        )
        ids = [instance["id"] for instance in instances]
        flattened_instances = [
            {"prompt": instance["prompt"], "response": response}
            for instance in instances
            for response in instance["responses"]
        ]

        collator = SupervisedDataCollator(tokenizer=self.tokenizer)
        model_inputs = collator(flattened_instances)
        model_inputs.update({"response_labels": response_labels})
        model_inputs.update({"ids": ids})

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
            torch.cat((s, t), dim=0) for s, t in zip(prompts_tokenized, responses_tokenized)
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

        data_dict = self.tokenize_prompts_and_responses(prompts, responses, self.tokenizer)

        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=util.IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
