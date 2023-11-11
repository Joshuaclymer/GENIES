import time
import ast
import copy
import sys
import traceback
from typing import List
import random
import math
import traceback
import json
import openai
import api.util as util
from tqdm import tqdm
import re
import os
import fire
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


def gpt_translation_prompt(list_of_strings):
    dict_of_strings = [{"text" : s} for s in list_of_strings]
    dictionary_string = json.dumps(dict_of_strings, indent=4)
    return \
f"""Translate the following dictionary values to Spanish. Return a list of python dictionaries with the translated values. If there are numbers, translate the numbers to spanish words. Ensure the output is in proper python syntax. e.g. be sure that you use ' marks inside of " quotations. When translating code, translate variable names but ensure that the code remains executable.

Input list of dictionaries:
{dictionary_string}

Output list of dictionaries:
"""

def parse_dict(string):
    return ast.literal_eval(string)

def query_gpt(prompt, model, system_message):
    openai.api_key = util.openai_api_key()
    while True:
        try:
            time.sleep(0.1)
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            )
            break
        except:
            time.sleep(2)

    completion = response.choices[0].message.content
    return completion



def translate(gpt_model, list_of_strings):
    system_message = "You are a helpful assistant that only returns lists of python dictionaries."
    indicies_not_empty = [i for i, s in enumerate(list_of_strings) if s != ""]
    items_not_empty = [list_of_strings[i] for i in indicies_not_empty]
    gpt_prompt = gpt_translation_prompt(items_not_empty)
    result = query_gpt(gpt_prompt, gpt_model, system_message)
    try:
        dict_result = parse_dict(result)
    except:
        raise TypeError()
    try:
        result_strings = [d["text"] for d in dict_result]
    except:
        raise TypeError()
    for i, s in zip(indicies_not_empty, result_strings):
        list_of_strings[i] = s
    return list_of_strings
    
def make_progress_bar(iterable, total):
    if rank == world_size - 1:
        return tqdm(iterable, total=total)
    else:
        return iterable


def retry_if_response_is_invalid(func, args, tries = 3):
    for i in range(tries):
        try:
            return func(*args)
        except TypeError:
            #traceback.print_exc()
            continue
    print("Discarded example because GPT-3 returned an invalid response three times")

def manager(input_dir, output_dir, max_examples): 
    print("Translating to spanish...") 
    input_data = util.load_json(input_dir)

    print(output_dir)
    if not os.path.exists(output_dir):
        util.save_json([], output_dir)

    output_examples = util.load_json(output_dir)
    input_examples = input_data

    is_same = (
        lambda x, y: x["english"] == y["english"]
    )
    unprocessed_examples = [
        e for e in input_examples if not any([is_same(e, o) for o in output_examples])
    ]
    examples_to_process = copy.deepcopy(unprocessed_examples)
    num_examples_to_process = len(examples_to_process)

    def save_results(results):
        output_examples_copy = copy.deepcopy(output_examples)
        output_examples_copy.extend(results)
        util.save_json(output_examples_copy, output_dir)

    results = []
    progress_bar = tqdm(total=num_examples_to_process)
    progress_bar.update(len(results))

    # Create request objects for checking if workers are free
    free_requests = [comm.irecv(source=i, tag=10) for i in range(1, world_size)]

    while len(results) < num_examples_to_process:
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
                        save_results(results)
                        progress_bar.update(len(result))
                    # Send the worker a new example
                    if len(examples_to_process) > 0:
                        examples = []
                        for j in range(5):
                            if len(examples_to_process) > 0:
                                examples.append(examples_to_process.pop(0))
                        comm.send(examples, dest=i, tag=11)
            time.sleep(0.1)
    save_results(results)
    print("\nFinished translating.")


def kill_workers():
    for i in range(1, world_size):
        comm.send("kill", dest=i, tag=11)


def worker(model): 
    # Send is free message and None for result to manager
    comm.send(True, dest=0, tag=10)
    comm.send("worker_just_initialized", dest=0, tag=11)

    def send_example(example):
        comm.send(True, dest=0, tag=10)
        comm.send(example, dest=0, tag=11)

    while True:
        # Receive example from manager
        message = comm.recv(source=0, tag=11)
        comm.send(False, dest=0, tag=10)
        if message == "kill":
            break
        examples = message
        translated = translate(model, [e["english"] for e in examples])
        for example, t in zip(examples, translated):
            example["spanish"] = t
        send_example(examples)  # Send the example back

def main(
    input_path,
    output_path,
    max_examples=None,
    model="gpt-3.5-turbo",
):
    if world_size == 1:
        raise ValueError("Must run with more than 1 process.")

    try:
        if rank == 0:
            manager(input_path, output_path, max_examples)
            kill_workers()
            MPI.Finalize()
        else:
            worker(model) 
    except:
        traceback.print_exc()
        kill_workers()
        MPI.Finalize()


if __name__ == "__main__":
    fire.Fire(main)
