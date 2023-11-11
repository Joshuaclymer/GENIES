import datetime
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import fire
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

import api.util as util
import wandb
from api.data_classes import Distribution, MCDataset
from api.model import Model


def train(
    model_dir: str,
    train_distribution: str,
    intervention_dir: str,
    eval_distributions: List[str] = None,
    dry_run: bool = False,
    train_kwargs: bool = {},
    retrain: dict = False,
) -> Optional[List[dict]]:
    model_name = os.path.basename(model_dir)
    train_distribution_id = Distribution(train_distribution).id
    intervention_name = os.path.basename(intervention_dir)
    output_model_name = f"{intervention_name}/{model_name}-{train_distribution_id}"
    output_model_dir = f"models/{output_model_name}"
    if eval_distributions == None:
        eval_distributions = [train_distribution]
    else:
        eval_distributions = [train_distribution] + eval_distributions

    util.print_once("")

    if not retrain:
        if os.path.exists(output_model_dir):
            model_exists = "pytorch_model.bin" in os.listdir(
                output_model_dir
            ) or "adapter_model.bin" in os.listdir(output_model_dir)
            if model_exists:
                util.print_once(
                    f"Model {output_model_name} already exists. Skipping training."
                )
                return output_model_dir
            else:
                util.print_once(
                    "WARNING: model directory exists but no model can be found there. Training the model anyway. Note that another process might be training the same model and the two processes might conflict."
                )

    util.print_once(
        f"# Training {model_name} on {train_distribution_id} with strategy '{intervention_name}'"
    )
    if dry_run:
        util.print_once("Skipping training because dry_run=True.")
        return output_model_dir
    else:
        args = {
            "model_dir": model_dir,
            "output_dir": output_model_dir,
            "training_distribution_dir": train_distribution,
            "test_distribution_dirs": eval_distributions,
        }
        args.update(train_kwargs)
        port = random.randint(10000, 20000)
        command = (
            f"accelerate launch --main_process_port {port} {intervention_dir}/train.py"
        )

        for key, value in args.items():
            if isinstance(value, list):
                command += f' --{key} "{value}"'
            else:
                command += f" --{key} {value}"
        util.execute_command(command)

        return output_model_dir


default_training_arguments = {
    "gradient_accumulation_steps": 1,
    "evaluation_strategy": "steps",
    "save_steps": 30,
    "save_total_limit": 1,
    "eval_steps": 10,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "ddp_find_unused_parameters": False,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 1,
    "weight_decay": 0.00,
    "save_total_limit": 0,
    "deepspeed": "configs/ds_zero_1.json",
    "gradient_checkpointing": False,
    "max_grad_norm": 0.3,
    "adam_beta2": 0.999,
    "optim": "paged_adamw_32bit",
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "constant",
    "logging_steps": 1,
    "weight_decay": 0.00,
    "report_to": "wandb",
    "remove_unused_columns": False,
}


def train_with_trainer(
    model: Model,
    output_dir: str,
    train_dataset: MCDataset,
    trainer: Trainer,
    data_collator,
    eval_datasets: Optional[List[Dataset]] = None,
    compute_metrics=None,
    max_eval_examples: int = None,
    max_train_examples: int = None,
    train_args=default_training_arguments,
    num_train_steps: int = None,
    **kwargs,
):
    print("training with trainer")
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    wandb.login(key=util.wandb_api_key())
    train_args.update({"run_name": f"train|{output_dir.replace('/', '-')}"})

    if eval_datasets != None:
        for e in eval_datasets:
            e.set_max_examples(max_eval_examples)
            e.filter_out_long_examples(model.tokenizer)
        if len(eval_datasets) > 1:
            eval_datasets = {d.distribution_id: d for d in eval_datasets}
            train_args[
                "metric_for_best_model"
            ] = f"eval_{train_dataset.distribution_id}_score"
        else:
            eval_datasets = eval_datasets[0]

    # Adjust the epochs and max_steps so that the number of training examples is roughly consistent across datasets with different numbers of examples
    if num_train_steps != None:
        num_pairs_train = len(train_dataset.examples)
        effective_batch_size = (
            train_args["per_device_train_batch_size"]
            * len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            * train_args["gradient_accumulation_steps"]
        )
        num_steps_per_epoch = math.ceil(num_pairs_train / effective_batch_size)
        train_args["num_train_epochs"] = num_train_steps // num_steps_per_epoch + 1
        train_args["max_steps"] = num_train_steps

    hf_training_arguments = TrainingArguments(output_dir=output_dir, **train_args)

    model.hf_model.config.use_cache = False
    model.hf_model.train()
    train_dataset.set_max_examples(max_train_examples)
    train_dataset.filter_out_long_examples(model.tokenizer)

    trainer = trainer(
        model=model.hf_model,
        tokenizer=model.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        args=hf_training_arguments,
        **kwargs,
    )
    wandb.finish()

    # if "checkpoint" in model.dir:
    #     resume_from_checkpoint = True
    # else:
    resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save log data
    logs = trainer.state.log_history
    if util.is_main():
        util.save_json(logs, f"{hf_training_arguments.output_dir}/training_logs.json")

    model.tokenizer.save_pretrained(hf_training_arguments.output_dir)
    trainer.save_model()

    (f"Model saved in {hf_training_arguments.output_dir}")

    # Delete checkpoints to save space
    if util.is_main():
        dirs = os.listdir(output_dir)
        checkpoint_dirs = [f"{output_dir}/{d}" for d in dirs if "checkpoint" in d]
        [shutil.rmtree(d) for d in checkpoint_dirs]

    # Save the training arguments with the model for future reference
    reordered_train_args = {}
    reordered_train_args["initial_model_dir"] = model.dir
    reordered_train_args["distribution_id"] = train_dataset.distribution_id
    reordered_train_args["date_trained"] = datetime.datetime.now().strftime(
        r"%d/%m/%Y %H:%M:%S"
    )
    reordered_train_args.update(asdict(hf_training_arguments))
    util.save_json(
        reordered_train_args, f"{hf_training_arguments.output_dir}/train_args.json"
    )

    return logs


def fire_wrap(*args, **kwargs):
    train(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
