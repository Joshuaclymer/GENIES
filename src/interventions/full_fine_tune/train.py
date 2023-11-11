import copy
import os
import shutil
from dataclasses import dataclass 
from typing import Any, Dict, List, Optional 

import fire
import transformers
from transformers import PreTrainedTokenizerBase 
from trl import RewardTrainer

import api.train
import api.util as util
from api.data_classes import (
    Distribution,
    SupervisedDataCollator,
)
from api.evaluate import compute_metrics
from api.model import Model
from api.train import train_with_trainer


@dataclass
class MCRewardCollator:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, mc_dataset_instances: List[dict]) -> Dict[str, Any]:
        chosen_instances = []
        for e in mc_dataset_instances:
            e["response"] = [r for r in e["responses"] if e["responses"][r] == 1][0]
            chosen_instances.append(copy.deepcopy(e))

        rejected_instances = []
        for e in mc_dataset_instances:
            e["response"] = [r for r in e["responses"] if e["responses"][r] == 0][0]
            rejected_instances.append(copy.deepcopy(e))

        supervised_collator = SupervisedDataCollator(self.tokenizer)
        chosen_tokenized = supervised_collator(chosen_instances)
        rejected_tokenized = supervised_collator(rejected_instances)
        batch = {
            "input_ids_chosen": chosen_tokenized["input_ids"],
            "attention_mask_chosen": chosen_tokenized["attention_mask"],
            "input_ids_rejected": rejected_tokenized["input_ids"],
            "attention_mask_rejected": rejected_tokenized["attention_mask"],
            "return_loss": True,
        }
        return batch


def main(
    model_dir: str,
    output_dir: str,
    training_distribution_dir: str,
    test_distribution_dir: Optional[List[str]] = None,
    max_eval_examples: int = 100,
    max_train_examples: int = None,
    peft_config=None,
    **kwargs,
) -> Optional[List[float]]:
    train_dataset = Distribution(training_distribution_dir).training_dataset
    if test_distribution_dir != None:
        eval_dataset = Distribution(test_distribution_dir).test_dataset
    else:
        eval_dataset = None
    train_dataset.convert_to_pairs(one_pair_per_instruction=True)
    eval_dataset.convert_to_pairs(one_pair_per_instruction=True)
    model = Model(model_dir, type=transformers.AutoModelForSequenceClassification)
    train_args = api.train.default_training_arguments
    train_args.update(kwargs)

    return train_with_trainer(
        model=model,
        output_dir=output_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer=RewardTrainerStorageEfficient,
        train_args=train_args,
        compute_metrics=compute_metrics,
        data_collator=MCRewardCollator(tokenizer=model.tokenizer),
        max_eval_examples=max_eval_examples,
        max_train_examples=max_train_examples,
        peft_config=peft_config,
    )


def fire_wrapper(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrapper)


class RewardTrainerStorageEfficient(RewardTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if util.is_main():
            best_checkpoint_dir = self.state.best_model_checkpoint
            util.print_once(f"Deleting all checkpoints except {best_checkpoint_dir}")
            all_checkpoint_dirs = [
                f"{self.args.output_dir}/{d}"
                for d in os.listdir(self.args.output_dir)
                if "checkpoint" in d
            ]
            if best_checkpoint_dir != None and len(all_checkpoint_dirs) > 0:
                for checkpoint_dir in all_checkpoint_dirs:
                    if os.path.basename(checkpoint_dir) != os.path.basename(
                        best_checkpoint_dir
                    ):
                        shutil.rmtree(checkpoint_dir)
        super()._save_checkpoint(model, trial, metrics)
