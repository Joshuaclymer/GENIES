import copy
import dataclasses
import importlib
import io
import json
import logging
import math
import os
import random
import re
import string
import sys
import time
import traceback
from functools import partial
from multiprocessing import Pool
from typing import Optional, Sequence, Union

import fire
import numpy as np
import openai
import tqdm
from mpi4py import MPI
from openai import openai_object

# import tqdm
from rouge_score import rouge_scorer

import api.model
import api.util
from api.model import Model
from gen_data.gen_utils.distribution_generator import DistributionGenerator

openai.api_key = api.util.openai_api_key()

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.5  # Caution: changed from 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model="gpt-3.5-turbo",
    sleep_time=5,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[
    StrOrOpenAIObject,
    Sequence[StrOrOpenAIObject],
    Sequence[Sequence[StrOrOpenAIObject]],
]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    if batch_size > 1:
        raise Exception(
            "batch_size > 1 is not supported anymore due to OpenAI API changes."
        )

    completions = []

    for batch_id, prompt_batch in enumerate(prompt_batches):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                messages = [
                    {
                        "role": "system",
                        "content": "You are part of an api and are expected to output text exactly as expected and specified.",
                    },
                    {"role": "user", "content": prompt_batch[0]},
                ]
                completion_batch = openai.ChatCompletion.create(
                    messages=messages, **shared_kwargs
                )
                choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.append(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(
                        batch_decoding_args.max_tokens * 0.8
                    )
                    logging.warning(
                        f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying..."
                    )
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [
            completions[i : i + decoding_args.n]
            for i in range(0, len(completions), decoding_args.n)
        ]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def encode_prompt(distribution: DistributionGenerator, prompt_instructions):
    batch_size = distribution.batch_size

    if distribution.leading_sentence == None and distribution.requirements is None:
        prompt = ""
    else:
        requirements_string = "\n".join(
            [f"{idx + 1}. {req}\n" for idx, req in enumerate(distribution.requirements)]
        )
        prompt = f"""{distribution.leading_sentence}\n\nHere are the requirements:\n{requirements_string}\n\n"""
    prompt += f"List of {len(prompt_instructions) + batch_size} tasks:"

    for idx, task_dict in enumerate(prompt_instructions):
        prompt += f"###\n"
        for heading in distribution.headings:
            if heading["key"] in task_dict:
                key = heading["key"]
                prompt += f"{idx + 1}. {heading['heading']}:\n{task_dict[key]}\n"

    prompt += f"###\n"
    if distribution.headings[0]["key"] in task_dict:
        key = distribution.headings[0]["key"]
        prompt += f"{idx + 2}. {distribution.headings[0]['heading']}:"
    return prompt


def post_process_gpt3_response(
    num_prompt_instructions,
    response,
    distribution: DistributionGenerator,
):
    if response is None:
        return []
    raw_examples = (
        f"{num_prompt_instructions+1}. {distribution.headings[0]['heading']}:\n"
        + response[0]["message"]["content"]
    )
    raw_examples = re.split("###", raw_examples)
    instructions = []
    for idx, example in enumerate(raw_examples):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_examples) - 1 and response[0]["finish_reason"] == "length":
            continue
        strip_chars = "\n` \t"
        heading_strings = [heading["heading"] for heading in distribution.headings]
        keys = [heading["key"] for heading in distribution.headings]
        splitted_data = re.split(f"\d\.\s+({'|'.join(heading_strings)}):", example)
        values = [splitted_data[i] for i in range(2, len(splitted_data), 2)]
        values = [s.strip(strip_chars) for s in values]
        if len(values) != len(keys):
            print("Discarded example because the number of values is not equal to the number of keys")
            continue
        example_dict = dict(zip(keys, values))
        for key in example_dict:
            if example_dict[key] == "<empty>":
                del example_dict[key]

        instructions.append(example_dict)
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def main(
    distribution_class_name,
    path_to_class_def,
    dir_to_output_data,
    output_filepath=None,
    num_instructions_to_generate=100,
    model="gpt-3.5-turbo",
    num_prompt_instructions=3,
    request_batch_size=1,
    rogue_threshold=0.7,
    temperature=1.0,
    top_p=1.0,
):
    if request_batch_size > 1:
        raise Exception(
            "request_size > 1 is not supported anymore due to OpenAI API changes."
        )

    distribution = api.util.import_class(distribution_class_name, path_to_class_def)
    primary_key = distribution.primary_key
    if output_filepath is None:
        output_filepath = f"{dir_to_output_data}/gen.json"
    seed_instruction_data = distribution.examples
    if rank == 0:
        print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    num_seed_examples = len(seed_instruction_data)

    request_idx = 0
    # load the LM-generated instructions
    if rank == 0:
        if not os.path.exists(output_filepath):
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, "w") as f:
                print("Created output file.")
                json.dump([], f)
    time.sleep(0.1)
    if rank == 0:
        print(f"output_filepath={output_filepath}")
    machine_instruction_data = api.util.load_json(output_filepath)
    if rank == 0:
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    if rank == 0:
        progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
        if machine_instruction_data:
            progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d[primary_key] for d in seed_instruction_data] + [
        d[primary_key] for d in machine_instruction_data
    ]
    all_instruction_tokens = [
        scorer._tokenizer.tokenize(inst) for inst in all_instructions
    ]

    while len(all_instructions) - num_seed_examples < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            if "difficulty_to_grade" in seed_instruction_data[0]:
                # Sample examples with different values for 'difficulty_to_grade'
                prompt_instructions = []
                difficulty_values = distribution.difficulty_values

                # Create separate lists for different difficulty levels
                difficulty_examples = {value: [] for value in difficulty_values}
                for e in seed_instruction_data:
                    difficulty = e["difficulty_to_grade"]
                    if difficulty in difficulty_examples:
                        difficulty_examples[difficulty].append(e)

                num_prompt_instructions = min(
                    num_prompt_instructions, len(seed_instruction_data)
                )
                if num_prompt_instructions < len(difficulty_values):
                    print(
                        "WARNING: Not enough seed instructions to provide one for each difficulty level."
                    )

                prompt_instructions = []
                i = 0
                while len(prompt_instructions) < num_prompt_instructions:
                    current_difficulty = difficulty_values[i % len(difficulty_values)]
                    examples_for_difficulty = difficulty_examples[current_difficulty]
                    if examples_for_difficulty:
                        while True:
                            s = random.choice(examples_for_difficulty)
                            if not s in prompt_instructions:
                                prompt_instructions.append(s)
                                i += 1
                                break
            else:
                prompt_instructions = random.sample(
                    seed_instruction_data, num_prompt_instructions
                )
            prompt = encode_prompt(distribution, prompt_instructions)
            batch_inputs.append(prompt)
        decoding_args = OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=1000,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )
        results = openai_completion(
            prompts=batch_inputs,
            model=model,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={
                "100257": -100
            },  # prevent the <|endoftext|> token from being generated
        )
        # print(f"Results: {results}")

        instruction_data_produced_by_gpt = []
        for result in results:
            # print(f"\nnum_prompt_instructions={num_prompt_instructions}\n")
            # print(f"\nresult={result}\n")
            # print(f"\ndistribution={distribution}\n")
            new_instructions = post_process_gpt3_response(
                num_prompt_instructions,
                result,
                distribution=distribution,
            )
            instruction_data_produced_by_gpt += new_instructions

        new_instruction_data_entries = instruction_data_produced_by_gpt
        new_instruction_tokens_list = [
            scorer._tokenizer.tokenize(instruction_data_entry[primary_key])
            for instruction_data_entry in instruction_data_produced_by_gpt
        ]

        assert len(new_instruction_data_entries) == len(new_instruction_tokens_list)

        new_instruction_data_entries = comm.allgather(new_instruction_data_entries)
        new_instruction_tokens_list = comm.allgather(new_instruction_tokens_list)

        new_instruction_data_entries = [
            item for sublist in new_instruction_data_entries for item in sublist
        ]
        new_instruction_tokens_list = [
            item for sublist in new_instruction_tokens_list for item in sublist
        ]

        total = len(new_instruction_data_entries)
        filtered_instruction_data_entries = []
        filtered_instruction_tokens_list = []
        print(f"\nTotal={total}\n")
        keep = 0
        if total > 0:
            assert isinstance(new_instruction_data_entries[0], dict)

            for instruction_data_entry, new_instruction_tokens in zip(
                new_instruction_data_entries, new_instruction_tokens_list
            ):
                # computing similarity with the pre-tokenzied instructions

                rouge_scores = []
                for instruction in all_instruction_tokens:
                    rouge_scores.append(
                        rouge_scorer._score_lcs(new_instruction_tokens, instruction)
                    )

                rouge_scores = [score.fmeasure for score in rouge_scores]
                sorted_indices = np.argsort(rouge_scores)[-10:][::-1]
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in sorted_indices
                }
                if max(rouge_scores) > rogue_threshold:
                    print(f"Rouge scores too high, skipping.")
                    continue
                else:
                    keep += 1
                    # print(f"Keep = {keep}")
                    instruction_data_entry[
                        "most_similar_instructions"
                    ] = most_similar_instructions
                    instruction_data_entry["avg_similarity_score"] = float(
                        np.mean(rouge_scores)
                    )

                    filtered_instruction_data_entries.append(instruction_data_entry)
                    filtered_instruction_tokens_list.append(new_instruction_tokens)

            machine_instruction_data.extend(filtered_instruction_data_entries)
            if distribution.resample:
                seed_instruction_data.extend(filtered_instruction_data_entries)
            all_instructions.extend(
                [d[primary_key] for d in filtered_instruction_data_entries]
            )
            if rank == 0:
                progress_bar.update(len(filtered_instruction_data_entries))
                print(f"Kept {keep} out of {total} instructions")
            all_instruction_tokens.extend(filtered_instruction_tokens_list)

            assert len(all_instructions) == len(all_instruction_tokens)

            if rank == 0:
                api.util.save_json(machine_instruction_data, output_filepath)
                examples_path = os.path.join(
                    os.path.dirname(output_filepath), "examples.json"
                )
                convert_gen_to_distribution(
                    output_filepath,
                    examples_path,
                    distribution_class=distribution,
                )
                print(f"Kept {keep} out of {total} instructions")

    if rank == 0:
        examples_path = os.path.join(
            os.path.dirname(output_filepath), "examples.json"
        )
        convert_gen_to_distribution(
            output_filepath,
            examples_path,
            distribution_class=distribution,
        )
        print("\nFinished generating instructions and preferred completions.")


def convert_gen_to_distribution(path, output_path, distribution_class):
    examples = None
    examples = api.util.load_json(path)

    for i, e in enumerate(examples):
        del e["most_similar_instructions"]
        del e["avg_similarity_score"]
        examples[i] = distribution_class.post_process(e)

    data = {
        "name": distribution_class.name,
        "formats": distribution_class.formats,
        "examples": examples,
    }
    api.util.save_json(data, output_path)
    return data


if __name__ == "__main__":
    fire.Fire(main)
    MPI.Finalize()
