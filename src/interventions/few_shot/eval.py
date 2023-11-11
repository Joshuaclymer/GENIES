import os
import random
from typing import List, Optional

import fire
import torch
import transformers
from transformers import BitsAndBytesConfig

import api.util as util
from api.data_classes import Distribution, MCDataCollator
from api.evaluate import evaluate_with_trainer
from api.model import Model
from interventions.pro.train import MCTrainer


def main(
    distribution_dirs: List[str],
    model_dir: str = None,
    output_paths: Optional[List[str]] = None,
    per_device_batch_size: int = 32,
    max_examples: Optional[int] = None,
    num_shots: int = 4,
    **kwargs,
):
    config = util.load_json(f"{model_dir}/config.json")
    model_dir = config["model_dir"]
    source_dir = config["training_distribution_dir"]
    training_distribution = Distribution(source_dir).training_dataset
    eval_datasets = [Distribution(d).test_dataset for d in distribution_dirs]

    deliminator = "\n\n"

    def sample_prefix():
        shot_examples = random.sample(training_distribution.examples, num_shots)
        return "".join(
            [
                deliminator
                + e["prompt"]
                + [r for r in e["responses"] if e["responses"][r] == 1][0]
                for e in shot_examples
            ]
        )

    # Edit eval datasets to include few-shot prefix from the training dataset
    for d in eval_datasets:
        for e in d.examples:
            e["prompt"] = sample_prefix() + deliminator + e["prompt"]
    print(e["prompt"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    if os.environ.get("WORLD_SIZE") == None or os.environ.get("WORLD_SIZE") == "1":
        device_map = None
    else:
        device_map = {"": device}
    model = Model(
        model_dir,
        quantization_config=bnb_config,
        type=transformers.AutoModelForCausalLM,
        device_map=device_map,
    )
    data_collator = MCDataCollator(model.tokenizer)

    return evaluate_with_trainer(
        trainer=MCTrainer,
        intervention_name="few_shot",
        datasets=eval_datasets,
        model=model,
        output_paths=output_paths,
        per_device_batch_size=per_device_batch_size,
        max_examples=max_examples,
        data_collator=data_collator,
    )


def fire_wrap(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
