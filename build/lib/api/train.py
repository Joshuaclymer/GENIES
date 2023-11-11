import os
import api.util as util
from typing import List, Optional, Union
import fire
from api.data_classes import Distribution


def train(
    model_dir: str,
    distribution: Union[str, Distribution],
    intervention_dir: str,
    dry_run: bool = False,
    train_kwargs: bool = {},
    retrain: dict = False,
) -> Optional[List[dict]]:

    train_module = util.import_module_from_path(f"{intervention_dir}/train.py")
    model_name = os.path.basename(model_dir)
    if isinstance(distribution, str):
        distribution = Distribution(distribution)
    intervention_name = os.path.basename(intervention_dir)
    output_model_name = f"{intervention_name}/{model_name}-{distribution.id}"
    output_model_dir = f"models/{output_model_name}"

    util.print_once("")
    if os.path.exists(output_model_dir) and not retrain:
        util.print_once(f"Model {output_model_name} already exists. Skipping training.")
        return output_model_dir, None

    util.print_once(
        f"# Training {model_name} on {distribution.id} with strategy '{intervention_name}'"
    )
    if dry_run:
        util.print_once("Skipping training because dry_run=True.")
        return output_model_dir, None
    else:
        logs = train_module.main(
            model_dir=model_dir,
            output_dir=output_model_dir,
            training_distribution_dir=distribution.dir,
            test_distribution_dir=distribution.dir,
            **train_kwargs,
        )
        return output_model_dir, logs


def fire_wrap(*args, **kwargs):
    train(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
