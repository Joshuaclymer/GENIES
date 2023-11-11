import os
import shutil
import time
import traceback
import warnings
from dataclasses import FrozenInstanceError, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fire
import torch
import torch.nn as nn
import api.util as util
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from trl import RewardTrainer
from trl.trainer.utils import RewardDataCollatorWithPadding

import api.train
from api.data_classes import Distribution
from api.evaluate import compute_metrics
from api.model import Model
from api.train import train_with_trainer
from interventions.full_fine_tune.train import MCRewardCollator, main


class RewardTrainer(Trainer):
    r"""
    The RewardTrainer can be used to train your custom Reward Model. It is a subclass of the
    `transformers.Trainer` class and inherits all of its attributes and methods. It is recommended to use
    an `AutoModelForSequenceClassification` as the reward model. The reward model should be trained on a dataset
    of paired examples, where each example is a tuple of two sequences. The reward model should be trained to
    predict which example in the pair is more relevant to the task at hand.

    The reward trainer expects a very specific format for the dataset. The dataset should contain two 4 entries at least
    if you don't use the default `RewardDataCollatorWithPadding` data collator. The entries should be named
    - `input_ids_chosen`
    - `attention_mask_chosen`
    - `input_ids_rejected`
    - `attention_mask_rejected`

    Optionally, you can also pass a `margin` entry to the dataset. This entry should contain the margin used to modulate the
    loss of the reward model as outlined in https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/.
    If you don't pass a margin, no margin will be used.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        max_length: Optional[int] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`RewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`List[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
        """
        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default RewardDataCollatorWithPadding"
                )
            if type(args) == TrainingArguments:
                if max_length is None:
                    warnings.warn(
                        "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
                        " It will be set to `512` by default, but you should do it yourself in the future.",
                        UserWarning,
                    )
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    warnings.warn(
                        "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
                        " It will be set to `512` by default, but you should do it yourself in the future.",
                        UserWarning,
                    )
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )[0]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        )[0]
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - inputs["margin"]
            ).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum to 1
        logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels

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
                    if os.path.basename(checkpoint_dir) != os.path.basename(best_checkpoint_dir):
                        shutil.rmtree(checkpoint_dir)
        super()._save_checkpoint(model, trial, metrics)


def main(
    model_dir: str,
    output_dir: str,
    training_distribution_dir: str,
    test_distribution_dirs: Optional[List[str]] = None,
    max_eval_examples: int = 250,
    peft_config=None,
    num_train_steps: int = 150,
    **kwargs,
) -> Optional[List[float]]:
    train_dataset = Distribution(training_distribution_dir).training_dataset
    if test_distribution_dirs != None:
        eval_datasets = [Distribution(d).test_dataset for d in test_distribution_dirs]
        eval_datasets.append(Distribution(training_distribution_dir).test_dataset)
        for e in eval_datasets:
            e.convert_to_pairs(one_pair_per_instruction=True)
    else:
        eval_datasets = [Distribution(training_distribution_dir).test_dataset]
    train_dataset.convert_to_pairs(one_pair_per_instruction=True)

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
        kwargs.update({"deepspeed": None})
    else:
        device_map = {"": device}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_origin = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        return_dict=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        num_labels=2,
        device_map=device_map,
        max_memory=max_memory,
    )

    model = prepare_model_for_kbit_training(model_origin)
    if peft_config != None:
        config = peft_config
    else:
        config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,  # Changed
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )

    peft_model = get_peft_model(model, config)
    model = Model(hf_model=peft_model, tokenizer=tokenizer, dir=model_dir)

    train_args = api.train.default_training_arguments
    train_args.update(kwargs)

    training_from_checkpoint = False
    if os.path.exists(model_dir):
        training_from_checkpoint = True

    try:
        return train_with_trainer(
            model=model,
            output_dir=output_dir,
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            trainer=RewardTrainer,
            train_args=train_args,
            compute_metrics=compute_metrics,
            data_collator=MCRewardCollator(tokenizer=tokenizer),
            max_eval_examples=max_eval_examples,
            num_train_steps=num_train_steps,
        )
    except:
        if training_from_checkpoint:
            traceback.print_exc()
            if util.is_main():
                print("Removing existing directory in case this resolves the error")
                shutil.rmtree(output_dir)
            time.sleep(30)
            return train_with_trainer(
                model=model,
                output_dir=output_dir,
                train_dataset=train_dataset,
                eval_datasets=eval_datasets,
                trainer=RewardTrainer,
                train_args=train_args,
                compute_metrics=compute_metrics,
                data_collator=MCRewardCollator(tokenizer=tokenizer),
                max_eval_examples=max_eval_examples,
                num_train_steps=num_train_steps,
            )


def fire_wrap(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
