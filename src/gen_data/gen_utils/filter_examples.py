import ast
import copy
import json
import os
import random
import time
import traceback
from typing import List

import fire
from mpi4py import MPI
from tqdm import tqdm

import api.util as util
import gen_data.gen_utils.generate_options
from api.data_classes import Distribution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


def gpt_filter_prompt(examples, distribution):
    example_strings = []
    for example in examples:
        possible_responses = example["completions"]
        possible_responses = [
            {"response": response, "id": i}
            for i, response in enumerate(possible_responses)
        ]

        prompt = Distribution.format_example(distribution.formats, example)["prompt"]
        example_string = ""
        example_string += f"#Prompt\n{prompt}\n"
        for response in possible_responses:
            example_string += (
                f"###Response with id {response['id']}\n {response['response']}\n"
            )
        example_strings.append(example_string)
    examples_str = "\n\n".join(example_strings)

    labels_string = "\n".join(
        [f"{l['label']}: {l['description']}" for l in distribution.filter_labels]
    )
    return f"""\
I'm generating a language model multiple-choice dataset of prompts and responses. Exactly one of the responses is meant to be good and the others are meant to be worse (in potentially in subtle ways). \
For each below example, try to predict which response is the best and also assign a label to the example. I will use these labels to filter out bad examples: 

{labels_string}

The examples are as follows. The input is optional.

{examples_str}

Return your answer as a list of python dictionaries -- each dictionary corresponding to an example above. Here's an example:
[
    {{
        "id_of_best_response": <int>,
        "label": "<str>",
    }},
    {{
        "id_of_best_response": <int>,
        "label": "<str>",
    }}, ...
]
"""


def gpt_filter(model, examples, distribution):
    system_message = (
        "You are a helpful assistant that only returns python dictionaries."
    )
    for i, example in enumerate(examples):
        random.shuffle(examples[i]["completions"])
    prompt = gpt_filter_prompt(examples, distribution)
    response = gen_data.gen_utils.generate_options.query_gpt(
        prompt, model, system_message, max_tokens=150, temperature=0.0
    )
    time.sleep(20)
    try:
        response_dict = ast.literal_eval(response)
    except:
        print(response)
        traceback.print_exc()
        raise TypeError

    if not len(response_dict) == len(examples):
        print(response_dict)
        print("response length does not match number of examples")
        raise TypeError("Response length does not match number of examples.")

    for i, example in enumerate(examples):
        response = response_dict[i]
        if not response["id_of_best_response"] in list(
            range(0, len(example["completions"]))
        ):
            print(examples[i])
            print("------")
            print(response)
            raise TypeError(
                "id_of_best_response is not in range of possible responses."
            )

    # add 'does match' key to indicate whether GPT-4 chose the completion marked preferred
    for i, example in enumerate(examples):
        try:
            examples[i]["preferred_completion_matches"] = (
                example["preferred_completion"]
                == example["completions"][int(response_dict[i]["id_of_best_response"])]
            )
        except:
            raise TypeError("id_of_best_response is not in ")

    # add label to examples
    for i, example in enumerate(examples):
        examples[i]["label"] = response_dict[i]["label"]

    return examples


def manager(input_dir, output_dir, max_examples, distribution, keep_label=False):
    print("Filtering examples...")
    input_data = util.load_json(input_dir)
    input_data["examples"] = [
        e for e in input_data["examples"] if not e["discard"] and "completions" in e
    ]
    if not os.path.exists(output_dir):
        util.save_json({"examples": []}, output_dir)
    output_examples = util.load_json(output_dir)["examples"]
    input_examples = input_data["examples"]
    is_same = (
        lambda x, y: x["instruction"] == y["instruction"]
        and x["preferred_completion"] == y["preferred_completion"]
    )
    num_processed_examples = len(output_examples)
    unprocessed_examples = [
        e for e in input_examples if not any([is_same(e, o) for o in output_examples])
    ]
    examples_to_process = copy.deepcopy(unprocessed_examples)[:max_examples]
    num_examples_to_process = len(examples_to_process)

    def save_results(results, distribution):
        def label_is_keep(label):
            if label == "invalid_gpt_response":
                return False
            label_dicts = distribution.filter_labels
            is_keep = [l["keep"] for l in label_dicts if l["label"] == label]
            if len(is_keep) == 0:
                return False
            return is_keep[0]

        [e.update({"discard": not label_is_keep(e["label"])}) for e in results]
        input_data["examples"] = results
        util.save_json(input_data, output_dir)
        output_path_for_cleaned_up_json = os.path.join(
            os.path.dirname(output_dir), "filtered.json"
        )

        cleaned_up_examples = []
        for example in results:
            if not example["discard"]:
                example_copy = copy.deepcopy(example)
                if not keep_label:
                    del example_copy["label"]
                if "prompt" in example_copy:
                    del example_copy["prompt"]
                del example_copy["discard"]
                del example_copy["preferred_completion_matches"]
                cleaned_up_examples.append(example_copy)

        input_data["examples"] = cleaned_up_examples
        util.save_json(input_data, output_path_for_cleaned_up_json)

    # Add prompt to each example using format string
    formats = input_data["formats"]
    for i, example in enumerate(examples_to_process):
        prompt = Distribution.format_example(formats, example)
        examples_to_process[i]["prompt"] = prompt

    results = output_examples
    progress_bar = tqdm(total=num_examples_to_process + num_processed_examples)
    progress_bar.update(len(results))

    # Create request objects for checking if workers are free
    free_requests = [comm.irecv(source=i, tag=10) for i in range(1, world_size)]

    while len(results) < num_examples_to_process + num_processed_examples:
        # Delegate examples to workers
        for i in range(1, world_size):
            # Check if worker is free
            is_free_status = free_requests[i - 1].test()
            if is_free_status[0]:
                is_free = is_free_status[1]
                # Re-initiate the non-blocking receive for this worker
                free_requests[i - 1] = comm.irecv(source=i, tag=10)
                # If the worker is free, collect processed examples send it a new example
                if is_free:
                    # Collect processed example
                    result = comm.recv(source=i, tag=11)
                    if result != "worker_just_initialized":
                        results.extend(result)
                        save_results(results, distribution)
                        progress_bar.update(len(result))
                    # Send the worker a new example
                    if len(examples_to_process) > 0:
                        examples = []
                        for j in range(5):
                            if len(examples_to_process) == 0:
                                break
                            examples.append(examples_to_process.pop(0))
                        print(f"sent {len(examples)} examples to worker {i}")
                        comm.send(examples, dest=i, tag=11)
            time.sleep(0.1)
    save_results(results, distribution)
    print("\nFinished filtering examples.")


def kill_workers():
    for i in range(1, world_size):
        comm.send("kill", dest=i, tag=11)


def worker(model, distribution):
    # Send is free message and None for result to manager
    comm.send(True, dest=0, tag=10)
    comm.send("worker_just_initialized", dest=0, tag=11)

    while True:
        # Receive example from manager
        message = comm.recv(source=0, tag=11)
        comm.send(False, dest=0, tag=10)
        if message == "kill":
            break
        examples = message
        labeled_examples = (
            gen_data.gen_utils.generate_options.retry_if_response_is_invalid(
                gpt_filter, (model, examples, distribution), tries=1
            )
        )
        if labeled_examples == None:
            [e.update({"label": "invalid_gpt_response"}) for e in examples]
        else:
            examples = labeled_examples

        comm.send(True, dest=0, tag=10)
        comm.send(examples, dest=0, tag=11)


def main(
    path_to_class_def,
    distribution_class_name,
    dir_to_output_data,
    max_examples=None,
    model="gpt-4",
    keep_label=False,
):
    if world_size == 1:
        raise ValueError("Must run with more than 1 process.")
    input_dir = f"./{dir_to_output_data}/with_options.json"
    output_dir = f"./{dir_to_output_data}/filter_info.json"

    distribution = util.import_class(distribution_class_name, path_to_class_def)
    try:
        if rank == 0:
            manager(
                input_dir, output_dir, max_examples, distribution, keep_label=keep_label
            )
            kill_workers()
            MPI.Finalize()
        else:
            worker(model, distribution)
    except:
        traceback.print_exc()
        kill_workers()
        MPI.Finalize()


if __name__ == "__main__":
    fire.Fire(main)
