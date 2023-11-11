from api.model import Model 
import fire
from typing import List, Optional
import os
from api.data_classes import MCDataCollator, Distribution
from interventions.log_prob_classify.train import MCTrainer
from api.evaluate import evaluate_with_trainer
from transformers import BitsAndBytesConfig
import torch
import transformers

def main(distribution_dirs: List[str], model_dir : str = None, output_paths : Optional[List[str]] = None, per_device_batch_size : int = 32, max_examples : Optional[int] = None) -> List[dict]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f'cuda:{local_rank}'
    if os.environ.get("WORLD_SIZE") == None or os.environ.get("WORLD_SIZE") == "1":
        device_map = None
    else:
        device_map = {"": device}
    model = Model(model_dir, quantization_config=bnb_config, type = transformers.AutoModelForCausalLM, device_map = device_map)
    data_collator = MCDataCollator(model.tokenizer)

    return evaluate_with_trainer(
        trainer=MCTrainer,
        intervention_name="pro",
        datasets=[Distribution(d).test_dataset for d in distribution_dirs],
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