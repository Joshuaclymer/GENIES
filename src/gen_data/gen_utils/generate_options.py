### Generation options
# In the end, there will be 5 options of varying quality, each assigned a rank.
# Generation process:
# - chatGPT generates 5 outpus of varying quality (at once). The lowest would be 'obviously a bad answer' and the highest would be 'the best answer you can possibly come up with'
# - now we give a different chatGPT instance all 5 outputs in a random order and ask it to rank them. If they don't match, the cycle starts again until a match is found or 5 iterations have passed, in which case, the example is discarded (there's a 96% chance that random orderings are discarded).

# Parallelization
# rank 0 is the manager. As soon as a process is free, it is given another example to compute and sends the result back to rank 0

import ast
import copy
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from typing import List

import fire
import openai
from mpi4py import MPI
from tqdm import tqdm

import api.util as util
from api.data_classes import Distribution
from gen_data.gen_utils.generate_instructions import OpenAIDecodingArguments

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


def gpt_completion_generation_prompt(example, distribution):
    requirements_string = "\n".join(
        [
            f"{idx + 1}. {req}\n"
            for idx, req in enumerate(distribution.option_requirements)
        ]
    )

    # Get demonstrations
    if distribution.options_demonstrations == None:
        num_examples = 1
        examples_string = """\
Return your response in the following format:

## Worse response 1
<str>
## Worse response 2
<str>
## Worse response 3
<str>
"""
    else:
        demonstrations = distribution.options_demonstrations

        examples_prompt = []
        examples_best = []
        examples_worse = []
        num_examples = len(examples_prompt) + 1
        for d in demonstrations:
            d = Distribution.format_example(distribution.formats, d)
            examples_prompt.append(d["prompt"])
            worse = [c for c in d["responses"] if d["responses"][c] == 0]
            best = [c for c in d["responses"] if d["responses"][c] == 1]
            examples_best.append(best[0])
            examples_worse.append(worse)
            assert len(worse) == 3

        examples_string = "\n".join(
            [
                f"""
## Prompt 
{prompt}
## Good response
{best}
## Worse response 1
{worse[0]}
## Worse response 2
{worse[1]}
## Worse response 3
{worse[2]}
"""
                for prompt, best, worse in zip(
                    examples_prompt, examples_best, examples_worse
                )
            ]
        )

    return f"""\
I'm generating a multiple choice dataset. Please produce three worse responses for each prompt. They should all be worse than the 'good' response. 

## Additional requirements for worse responses:
{requirements_string}

# {num_examples} multiple choice examples
{examples_string}

## Prompt 
{example["prompt"]}
## Good response
{example["preferred_completion"]}
"""


def gpt_select_preferred_prompt(prompt, responses):
    formatted_list = json.dumps(
        [{"responses": c, "id": i} for i, c in enumerate(responses)], indent=4
    )
    return f"""\
Please determine which of the following responses is the best.

Here is the prompt:

"{prompt}"

Here are the responses:

{formatted_list}

Return the id of the best response as a single python dictionary like so:
{{
    "best_id": <int>
}}
"""


def parse_list(string):
    # Get text after each separator
    matches = re.findall(r"(?<=## Worse response \d)[\s\S]+?(?=\n#|$)", string)
    matches = [m.strip() for m in matches]
    return matches


def parse_dict(string):
    return ast.literal_eval(string)


def query_gpt(prompt, model, system_message, temperature=0.2, max_tokens=600):
    openai.api_key = util.openai_api_key()
    decoding_args = OpenAIDecodingArguments(
        temperature=temperature,
        n=1,
        max_tokens=max_tokens,  # hard-code to maximize the length. the requests will be automatically adjusted
    )
    while True:
        try:
            time.sleep(0.1)
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                **decoding_args.__dict__,
            )
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                decoding_args.max_tokens = int(decoding_args.max_tokens * 0.8)
                logging.warning(
                    f"Reducing target length to {decoding_args.max_tokens}, Retrying..."
                )
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(5)  # Annoying rate limit on requests.

    completion = response.choices[0].message.content
    return completion


def generate_dispreferred_completions_with_gpt(gpt_model, example, distribution):
    system_message = "You generate (subtly) bad responses to instructions that are all creative and diverse."
    gpt_prompt = gpt_completion_generation_prompt(example, distribution)
    result = query_gpt(gpt_prompt, gpt_model, system_message)
    try:
        list_result = parse_list(result)
    except:
        print("0")
        raise TypeError()
    if len(list_result) != 3:
        print("1")
        # print(gpt_prompt)
        raise TypeError()
    example["completions"] = [example["preferred_completion"]] + list_result
    if not distribution.options_check(example):
        print("2")
        # print(example)
        raise TypeError()
    return list_result


def select_preferred_completion_with_gpt(gpt_model, prompt, responses):
    system_message = "You only return python dictionaries."
    gpt_prompt = gpt_select_preferred_prompt(prompt, responses)
    result = query_gpt(gpt_prompt, gpt_model, system_message)
    try:
        dict_result = parse_dict(result)
    except:
        print("2")
        raise TypeError()
    if "best_id" not in dict_result:
        print("3")
        raise TypeError()
    try:
        int_id = int(dict_result["best_id"])
    except:
        print("4")
        raise TypeError()
    return int_id


def make_progress_bar(iterable, total):
    if rank == world_size - 1:
        return tqdm(iterable, total=total)
    else:
        return iterable


def retry_if_response_is_invalid(func, args, tries=3):
    for i in range(tries):
        try:
            return func(*args)
        except TypeError:
            traceback.print_exc()
            continue
    print("Discarded example because GPT-3 returned an invalid response three times")


def manager(input_dir, output_dir, max_examples, distribution):
    print("Generating options...")
    input_data = util.load_json(input_dir)

    print(output_dir)
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

    def save_results(results):
        input_data["examples"] = results
        util.save_json(
            input_data,
            output_dir,
        )

    # Add prompt to each example using format string
    formats = input_data["formats"]
    for i, example in enumerate(examples_to_process):
        examples_to_process[i] = Distribution.format_example(formats, example)

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
                        results.append(result)
                        save_results(results)
                        progress_bar.update(1)
                    # Send the worker a new example
                    if len(examples_to_process) > 0:
                        example = examples_to_process.pop()
                        comm.send(example, dest=i, tag=11)
            time.sleep(0.1)
    save_results(results)
    print("\nFinished generating options.")


def kill_workers():
    for i in range(1, world_size):
        comm.send("kill", dest=i, tag=11)


def worker(model, distribution):
    # Send is free message and None for result to manager
    comm.send(True, dest=0, tag=10)
    comm.send("worker_just_initialized", dest=0, tag=11)

    def send_example(example):
        del example["prompt"]
        comm.send(True, dest=0, tag=10)
        comm.send(example, dest=0, tag=11)

    while True:
        # Receive example from manager
        message = comm.recv(source=0, tag=11)
        comm.send(False, dest=0, tag=10)
        if message == "kill":
            break
        example = message

        # Then generate dispreferred completions
        for i in range(3):
            dispreferred_completions = retry_if_response_is_invalid(
                generate_dispreferred_completions_with_gpt,
                (model, example, distribution),
            )
            if dispreferred_completions == None:
                example["discard"] = True
                example[
                    "discard_reason"
                ] = "GPT returned an invalid response when generating options three times."
                break
            all_completions = dispreferred_completions + [
                example["preferred_completion"]
            ]
            random.shuffle(all_completions)
            correct_preferred_id = all_completions.index(
                example["preferred_completion"]
            )
            preferred_id = retry_if_response_is_invalid(
                select_preferred_completion_with_gpt,
                (model, example["prompt"], all_completions),
            )
            if preferred_id == None:
                example["discard"] = True
                example[
                    "discard_reason"
                ] = "GPT returned an invalid response when selecting preferred completion three times."
                break
            if preferred_id == correct_preferred_id:
                example["discard"] = False
                break
        if "discard" not in example:
            example["discard"] = True
            example[
                "discard_reason"
            ] = "Preferred completion was not selected by GPT after three attempts"
        send_example(example)  # Send the example back


def main(
    path_to_class_def,
    distribution_class_name,
    dir_to_output_data,
    input_path=None,
    max_examples=None,
    model="gpt-3.5-turbo",
):
    if world_size == 1:
        raise ValueError("Must run with more than 1 process.")
    if input_path == None:
        input_path = f"./{dir_to_output_data}/examples.json"
    output_dir = f"./{dir_to_output_data}/with_options.json"

    distribution = util.import_class(distribution_class_name, path_to_class_def)
    try:
        if rank == 0:
            manager(input_path, output_dir, max_examples, distribution)
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
