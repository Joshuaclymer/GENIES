import copy
import os
import random
from typing import List, Optional, Union

import fire
import numpy as np

import api.util as util
import wandb
from api.data_classes import Distribution

# Example sweep configuration:
default_sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "eval/loss"},
    "parameters": {
        "learning_rate": {"values": list(np.logspace(-5, -3.5, 8))},
    },
}


def hyper_parameter_sweep(
    model_dir: str,
    distribution: Union[str, Distribution],
    intervention_dir: str,
    sweep_configuration: dict = default_sweep_configuration,
    train_kwargs: dict = {},
    eval_kwargs: dict = {},
    count: int = 8,
    wandb_project="project",
    ds_config_dir="configs/ds_zero_3.json",
) -> Optional[List[dict]]:
    if isinstance(distribution, str):
        distribution = Distribution(distribution)
    hps_distribution = distribution.create_hps_copy(
        num_train=400, num_eval=200, dir=f"{distribution.dir}/hps"
    )

    os.environ["WANDB_DIR"] = "./wandb"
    os.environ["WANDB_CACHE_DIR"] = "./wandb"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb"

    wandb.login(key=util.wandb_api_key())

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project)

    wandb.agent(
        sweep_id,
        function=lambda: train_with_params(
            model_dir,
            hps_distribution,
            intervention_dir,
            train_kwargs,
            eval_kwargs,
            ds_config_dir,
        ),
        count=count,
    )


def train_with_params(
    model_dir, distribution, intervention_dir, train_kwargs, eval_kwargs, ds_config_dir
):
    wandb.init()
    parameters = wandb.config
    print(train_kwargs)
    run_train_kwargs = copy.deepcopy(train_kwargs)
    run_train_kwargs.update(parameters)
    if isinstance(distribution, str):
        distribution = Distribution(distribution)

    model_name = os.path.basename(model_dir)
    intervention_name = os.path.basename(intervention_dir)
    output_model_name = f"{intervention_name}/{model_name}-{distribution.id}-{'-'.join([f'{k}={v}' for k,v in parameters.items()])}"
    output_model_dir = f"models/hps/{output_model_name}"
    # eval_output_path = f"results/evaluations/hps/{output_model_name}.json"

    port = random.randint(10000, 20000)
    train_command = (
        f"accelerate launch --main_process_port {port} {intervention_dir}/train.py"
    )
    run_train_kwargs.update(
        {
            "model_dir": model_dir,
            "output_dir": output_model_dir,
            "training_distribution_dir": distribution.dir,
            "test_distribution_dirs": [distribution.dir],
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "save_steps": 100,
        }
    )
    for k, v in run_train_kwargs.items():
        train_command += f' --{k} "{str(v)}"'
    # prefix = "source scl_source enable devtoolset-10; "
    prefix = ""
    train_command = prefix + train_command
    print("executing command:", train_command)
    util.execute_command(train_command)


if __name__ == "__main__":
    fire.Fire(hyper_parameter_sweep)
