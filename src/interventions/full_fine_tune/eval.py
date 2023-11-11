from typing import List, Optional

import fire
import transformers
from data_classes import Distribution

from api.evaluate import evaluate_with_trainer
from api.model import Model
from interventions.full_fine_tune.train import MCRewardCollator, RewardTrainerStorageEfficient


def main(
    distribution_dirs: List[str],
    model_dir: str = None,
    output_paths: Optional[List[str]] = None,
    per_device_batch_size: int = 32,
    max_examples: Optional[int] = None,
) -> List[dict]:
    model = Model(model_dir, type=transformers.AutoModelForSequenceClassification)
    data_collator = MCRewardCollator(model.tokenizer)
    return evaluate_with_trainer(
        trainer=RewardTrainerStorageEfficient,
        intervention_name="full_fine_tune",
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
