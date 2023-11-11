import os
from typing import List, Optional

import fire
import torch
import transformers
from api.data_classes import Distribution
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

from api.evaluate import evaluate_with_trainer
from api.model import Model
from interventions.full_fine_tune.train import MCRewardCollator, RewardTrainer


def main(
    distribution_dirs: List[str],
    model_dir: str = None,
    output_paths: Optional[List[str]] = None,
    per_device_batch_size: int = 32,
    max_examples: Optional[int] = None,
    intervention_name="lora_fine_tune",
    **kwargs,
) -> List[dict]:
    config = PeftConfig.from_pretrained(model_dir)
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
    if os.environ.get("WORLD_SIZE") == None or os.environ.get("WORLD_SIZE") == "1":
        device_map = None
        max_memory = None
    else:
        device_map = {"": device}
    original_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        num_labels=2,
    )
    model_loaded = PeftModel.from_pretrained(
        original_model,
        model_dir,
        is_trainable=False,
    ).to("cuda")
    model_loaded.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.base_model_name_or_path
    )
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Model(model_dir, hf_model=model_loaded, tokenizer=tokenizer)

    data_collator = MCRewardCollator(model.tokenizer)
    return evaluate_with_trainer(
        trainer=RewardTrainer,
        intervention_name=intervention_name,
        datasets=[Distribution(d).test_dataset for d in distribution_dirs],
        model=model,
        output_paths=output_paths,
        per_device_batch_size=per_device_batch_size,
        max_examples=max_examples,
        data_collator=data_collator,
        **kwargs,
    )


def fire_wrap(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
