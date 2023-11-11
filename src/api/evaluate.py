import datetime
import os
import random
import time
import numpy as np
from typing import List, Optional, Union

import fire
from transformers import Trainer, TrainingArguments

import api.util as util
import wandb
from api.data_classes import Distribution, MCDataset
from api.model import Model


def evaluate(
    model_dir: str,
    distributions: Union[List[str], str, List[Distribution]],
    intervention_dir: str,
    use_cached: bool = True,
    dry_run: bool = False,
    eval_kwargs: dict = {},
    use_accelerate: bool = True,
) -> List[dict]:
    intervention_name = os.path.basename(intervention_dir)
    if isinstance(distributions, str):
        distributions = [Distribution(distributions)]
    if isinstance(distributions[0], str):
        distributions = [Distribution(dir) for dir in distributions]

    distribution_dirs = [distribution.dir for distribution in distributions]
    model_name = os.path.basename(model_dir)
    output_paths = [
        f"results/evaluations/{intervention_name}/{model_name}/{distribution.id}.json"
        for distribution in distributions
    ]

    util.print_once("")
    util.print_once(f"# Evaluating {model_name} using strategy '{intervention_name}'")
    evaluations = [None] * len(output_paths)

    if use_cached:
        for i, path in enumerate(output_paths):
            if os.path.exists(path):
                eval_data = util.load_json(path)
                evaluations[i] = eval_data
        distributions_string = ", ".join(
            [
                distribution.id
                for distribution, evaluation in zip(distributions, evaluations)
                if evaluation != None
            ]
        )
        util.print_once(
            f"Cached evaluations found for the following distributions: {distributions_string}"
        )

    if not all(evaluations):
        distributions_string = ", ".join(
            [
                distribution.id
                for distribution, evaluation in zip(distributions, evaluations)
                if evaluation == None
            ]
        )
        util.print_once(
            f"Computing evaluations for the following distributions: {distributions_string}"
        )
        util.print_once(f"Running {intervention_name}/eval.py")

        indices_with_no_cache = [i for i, e in enumerate(evaluations) if e == None]
        distribution_dirs = [distribution_dirs[i] for i in indices_with_no_cache]
        output_paths = [output_paths[i] for i in indices_with_no_cache]

        if dry_run:
            util.print_once(
                "Skipping evaluation because dry_run=True. Returned random numbers for all scores."
            )
            result = [{"eval_accuracy": random.random()}] * len(indices_with_no_cache)
        elif not use_accelerate:
            print("NOT USING ACCELERATE")
            eval_module = util.import_module_from_path(intervention_dir + "/eval.py")
            result = eval_module.main(
                model_dir=model_dir,
                distribution_dirs=distribution_dirs,
                output_paths=output_paths,
                **eval_kwargs,
            )
        else:
            args = {
                "model_dir": model_dir,
                "distribution_dirs": distribution_dirs,
                "output_paths": output_paths,
            }
            args.update(eval_kwargs)
            port = random.randint(10000, 20000)
            command = f"accelerate launch --main_process_port {port} {intervention_dir}/eval.py"

            for key, value in args.items():
                if isinstance(value, list):
                    command += f' --{key} "{value}"'
                else:
                    command += f" --{key} {value}"
            util.execute_command(command)
            time.sleep(10)
            result = [util.load_json(path) for path in output_paths]
        for i, eval_data in zip(indices_with_no_cache, result):
            evaluations[i] = eval_data
    util.print_once(
        "\n".join(
            [
                f"{distribution.id}: {e['eval_accuracy']}"
                for distribution, e in zip(distributions, evaluations)
            ]
        )
    )
    return evaluations


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    correct_probabilities = np.array(
        [p[int(label)] for p, label in zip(predictions, labels)]
    )
    choices = np.argmax(predictions, axis=1)
    accuracy = np.array(choices == labels, dtype=float).mean().item()
    brier_score = np.mean(np.square(1 - correct_probabilities))

    random.seed(42)
    randomized_labels = np.array([random.choice([0,1]) for _ in range(len(correct_probabilities))])
    randomized_probabilities = np.array([(1 - p) if l == 0 else p for l,p in zip(randomized_labels, correct_probabilities)])
    num_bins = 5
    bins = np.linspace(0, 1, num_bins+1)
    bin_assignments = np.digitize(randomized_probabilities, bins)
    bin_scores = []
    
    for bin in range(1, num_bins + 1):
        probabilities_in_bin = randomized_probabilities[bin_assignments == bin]
        bin_size = probabilities_in_bin.shape[0]
        if bin_size < 20:
            continue
        bin_score = np.mean((np.mean(randomized_labels) - np.mean(probabilities_in_bin))**2)
        bin_scores.append(bin_score)
    calibration = np.sqrt(np.mean(np.array(bin_scores)))
    return {
        "accuracy": accuracy,
        "rms_calibration_error": calibration,
        "average_probability": np.mean(correct_probabilities),
        "brier_score": brier_score,
        "probabilities": list(correct_probabilities),
    }


def evaluate_with_trainer(
    trainer: Trainer,
    intervention_name: str,
    datasets: List[MCDataset],
    model: Model = None,
    output_paths: Optional[List[str]] = None,
    per_device_batch_size: int = 32,
    max_examples: Optional[int] = None,
    data_collator=None,
    **trainer_kwargs,
) -> List[dict]:
    model_name = model.dir.replace("/", "-")
    if output_paths == None:
        output_paths = [
            f"results/evaluations/{intervention_name}/{model_name}/{dataset.distribution_id}.json"
            for dataset in datasets
        ]

    evaluations = []
    for output_path, dataset in zip(output_paths, datasets):
        dataset.filter_out_long_examples(model.tokenizer)
        dataset.convert_to_pairs(one_pair_per_instruction=False)
        dataset.set_max_examples(max_examples)
        run_name = f"eval|{intervention_name}|{model_name}|{dataset.distribution_id}"
        metrics = get_metrics(
            model,
            per_device_batch_size,
            dataset,
            trainer,
            data_collator,
            trainer_kwargs,
            run_name,
        )

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


def get_metrics(
    model,
    per_device_batch_size,
    dataset,
    trainer,
    data_collator,
    trainer_kwargs,
    run_name,
) -> float:
    # First filter out examples that exceed the models maximum sequence length
    args = {
        "output_dir": "tmp",
        "evaluation_strategy": "epoch",
        "per_device_eval_batch_size": per_device_batch_size,
        "do_eval": True,
        # "bf16": True,
        "report_to": "wandb",
        # "report_to": "none",
        "remove_unused_columns": False,
    }

    args.update({"run_name": run_name})

    trainer = trainer(
        compute_metrics=compute_metrics,
        model=model.hf_model,
        tokenizer=model.tokenizer,
        eval_dataset=dataset,
        data_collator=data_collator,
        args=TrainingArguments(**args),
        **trainer_kwargs,
    )
    metrics = trainer.evaluate()
    wandb.finish()
    return metrics


def fire_wrap(*args, **kwargs):
    evaluate(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
