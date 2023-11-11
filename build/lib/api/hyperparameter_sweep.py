import fire
from api.data_classes import Distribution
from typing import Union, List, Optional
import wandb
import copy
import api.util as util
import os
import wandb

# Example sweep configuration:
default_sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "average_score"},
    "parameters": {
        "learning_rate": {"max": 1e-4, "min": 8e-6},
    },
}


def hyper_parameter_sweep(
    model_dir: str,
    distribution: Union[str, Distribution],
    intervention_dir: str,
    sweep_configuration: dict = default_sweep_configuration,
    train_kwargs: dict = {},
    eval_kwargs: dict = {},
    count: int = 4,
    wandb_project="project",
) -> Optional[List[dict]]:
    if isinstance(distribution, str):
        distribution = Distribution(distribution)

    (key=util.wandb_api_key())

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project)

    wandb.agent(
        sweep_id,
        function=lambda: train_with_params(
            model_dir, distribution, intervention_dir, train_kwargs, eval_kwargs
        ),
        count=count,
    )


def train_with_params(model_dir, distribution, intervention_dir, train_kwargs, eval_kwargs):
    wandb.init()
    parameters = wandb.config
    run_train_kwargs = copy.deepcopy(train_kwargs)
    run_train_kwargs.update(parameters)
    if isinstance(distribution, str):
        distribution = Distribution(distribution)

    model_name = os.path.basename(model_dir)
    intervention_name = os.path.basename(intervention_dir)
    output_model_name = f"{intervention_name}/{model_name}-{distribution.id}-{'-'.join([f'{k}={v}' for k,v in parameters.items()])}"
    output_model_dir = f"models/hps/{output_model_name}"
    eval_output_path = f"results/evaluations/hps/{output_model_name}.json"

    train_command = f"accelerate launch --deepspeed_config_file configs/ds_zero_3.json --use_deepspeed {intervention_dir}/train.py"
    run_train_kwargs.update(
        {
            "model_dir": model_dir,
            "output_dir": output_model_dir,
            "training_distribution_dir": distribution.dir,
        }
    )
    for k, v in run_train_kwargs.items():
        train_command += f' --{k} "{str(v)}"'

    util.execute_command(train_command)

    eval_command = f"accelerate launch --deepspeed_config_file configs/ds_zero_3.json --use_deepspeed {intervention_dir}/eval.py"
    eval_kwargs.update(
        {
            "model_dir": model_dir,
            "distribution_dirs": [distribution.dir],
            "output_paths": [eval_output_path],
        }
    )
    for k, v in eval_kwargs.items():
        eval_command += f' --{k} "{str(v)}"'
    util.execute_command(eval_command)

    # Open json to get evaluation
    result = util.load_json(eval_output_path)
    wandb.log({"average_score": result["average_score"]})


if __name__ == "__main__":
    fire.Fire(hyper_parameter_sweep)
