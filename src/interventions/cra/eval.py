import datetime
import os
import pickle
import random
from typing import List, Optional

import fire
import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from trl import RewardTrainer

import api.util as util
import interventions.mms.llama as llama
from api.data_classes import Distribution, MCDataCollator
from api.evaluate import compute_metrics, evaluate_with_trainer
from api.model import Model
from interventions.mms.train import get_llama_activations_bau
from interventions.pro.train import MCTrainer


def main(
    distribution_dirs: List[str],
    eval_datasets=None,
    model_dir: str = None,
    model: Model = None,
    output_paths: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    do_calibration: bool = True,
    seed=42,
    alpha=1,
):
    device = f"cuda"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    source_directions = torch.tensor(torch.load(model_dir + "/directions.pt")).to(
        device
    )
    config = util.load_json(model_dir + "/config.json")
    with open(model_dir + "/calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    model_dir = config["model_dir"]

    if model == None:
        hf_model = llama.LLaMAForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = Model(model_dir, hf_model=hf_model, quantization_config=None)

    heads = config["heads"]
    num_heads = model.hf_model.config.num_attention_heads

    evaluations = []
    for output_path, dataset in zip(
        output_paths, (distribution_dirs if eval_datasets == None else eval_datasets)
    ):
        if eval_datasets == None:
            dataset = Distribution(dataset).test_dataset
            dataset.convert_to_pairs(one_pair_per_instruction=False)
            dataset.set_max_examples(max_examples)

        # Tokenize prompts and get labels
        predictions = []
        true_labels = []
        for e in tqdm(dataset.examples):
            responses = list(e["responses"].keys())
            random.shuffle(responses)
            prompts = [e["prompt"] + r for r in responses]
            postfix1 = (
                "\n\n### Does the responder successfully follow the instruction?\nYes"
            )
            postfix2 = (
                "\n\n### Does the responder successfully follow the instruction?\nNo"
            )
            prompts_1 = [
                model.tokenizer(
                    p + postfix1, return_tensors="pt", padding=False, truncation=False
                )["input_ids"]
                for p in prompts
            ]
            prompts_2 = [
                model.tokenizer(
                    p + postfix2, return_tensors="pt", padding=False, truncation=False
                )["input_ids"]
                for i, p in enumerate(prompts)
            ]
            values = [e["responses"][r] for r in responses]

            label = values.index(1)

            def get_activations(tokenized_prompt):
                activations = []
                _, head_wise_activations, _ = get_llama_activations_bau(
                    model.hf_model, tokenized_prompt, device
                )
                head_wise_activations = head_wise_activations[:, -1, :]
                head_wise_activations = rearrange(
                    head_wise_activations, "l (h d) -> l h d", h=num_heads
                )
                for layer, head in heads:
                    activations.append(head_wise_activations[layer, head])
                return torch.stack(activations)

            target_activations_response_1 = get_activations(
                prompts_1[0]
            ) - get_activations(prompts_2[0])
            target_activations_response_2 = get_activations(
                prompts_1[1]
            ) - get_activations(prompts_2[1])

            target_directions = (
                target_activations_response_2 - alpha * target_activations_response_1
            )
            cosine_sims = F.cosine_similarity(
                target_directions, source_directions, dim=1
            )
            score = float(0.5 * (cosine_sims.mean() + 1))
            if do_calibration:
                prediction = calibrator.predict_proba([[score]])[0]
            else:
                prediction = [1 - score, score]
            predictions.append(prediction)
            true_labels.append(label)

        metrics = compute_metrics((predictions, true_labels))
        metrics = {f"eval_{k}": v for k, v in metrics.items()}

        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        eval_data = {
            "model_dir": model.dir,
            "distribution_id": dataset.distribution_id,
            "timestamp": current_time_str,
        }

        eval_data.update(metrics)
        evaluations.append(eval_data)
        util.save_json(eval_data, output_path)
        util.print_once(f"Saved evaluation at {output_path}")

    return evaluations


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def fire_wrap(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
