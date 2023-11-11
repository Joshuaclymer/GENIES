import random
import traceback
from copy import deepcopy

import fire

from api.data_classes import Distribution, MCDataset
from api.model import Model
from api.util import import_class, load_json, repo_path, save_json
from gen_data.gen_utils.distribution_generator import DistributionGenerator
import re

def get_format(formats, example):
    all_format_keys = [re.findall(r"\{(.+?)\}", format) for format in formats]
    all_format_keys = set([key for keys in all_format_keys for key in keys])
    example_keys = set(example.keys())
    example_format_keys = example_keys.intersection(all_format_keys)

    matching_format = None
    for format in formats:
        keys_for_format_string = set(re.findall(r"\{(.+?)\}", format))
        if keys_for_format_string == example_format_keys:
            matching_format = format
    assert matching_format != None
    return format

def main(
    dir_to_input_data,
    dir_to_output_data,
    path_to_class_def=None,
    class_name=None,
    desired_num_responses=4,
    keep_not_selected=False,
    keep_invalid=False,
    num_test_examples=250,
    num_train_examples=600,
):
    data = load_json(dir_to_input_data)
    if path_to_class_def is None or class_name is None:
        formats = data["formats"]
        name = data["name"]
    else:
        distribution = import_class(class_name, path_to_class_def)
        formats = distribution.formats
        name = distribution.name
    examples = data["examples"]
    print(f"Loaded {len(examples)} examples")
    new_examples = []
    tokenizer = Model.get_tokenizer("../../models/pythia-410m")
    for example in examples:
        if "completions" not in example or "preferred_completion" not in example:
            print("Skipping example without completions or preferred completion")
            continue
        example["responses"] = {
            c: float(c == example["preferred_completion"])
            for c in example["completions"]
        }
        if len(example["responses"]) != desired_num_responses:
            print("Skipping example with wrong number of responses")
            continue
        del example["completions"]
        del example["preferred_completion"]
        if "difficulty_to_grade" in example:
            del example["difficulty_to_grade"]
        if "discard" in example:
            if (
                example.get("discard_reason")
                == "Preferred completion was not selected by GPT after three attempts"
                and keep_not_selected
            ):
                pass
            elif (
                example.get("discard_reason")
                == "GPT returned an invalid response when generating options three times."
                and keep_invalid
            ):
                pass
            elif example["discard"]:
                print("Skipping discarded example")
                continue
            del example["discard"]
        if "input" in example and example["input"] == "":
            del example["input"]
        try:
            example = Distribution.format_example(formats, example)
            MCDataset.validate_example(
                example, desired_num_responses=desired_num_responses
            )
            max_len = max(
                [
                    len(tokenizer(example["prompt"] + r)["input_ids"])
                    for r in example["responses"]
                ]
            )
            if max_len > 600:
                print("Skipping example with too many tokens")
                continue
            del example["prompt"]
            new_examples.append(example)
        except:
            traceback.print_exc()
            return
        example_format = get_format(formats, example)
        example["prompt_format"] = example_format


    print(f"{len(new_examples)} examples passed checks")

    random.shuffle(new_examples)
    test_examples = new_examples[:num_test_examples]
    train_examples = new_examples[
        num_test_examples : num_train_examples + num_test_examples
    ]
    print("Split lengths:", len(test_examples), len(train_examples))
    id = name
    save_json(train_examples, f"{dir_to_output_data}/{id}/train.json")
    save_json(test_examples, f"{dir_to_output_data}/{id}/test.json")
    print(f"Splits are in {dir_to_output_data}/{id}")


if __name__ == "__main__":
    fire.Fire(main)
