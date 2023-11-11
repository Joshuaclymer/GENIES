from typing import List, Optional

import fire
from data_classes import Distribution

import api.util as util
from api.evaluate import compute_metrics


def main(
    distribution_dirs: List[str],
    model_dir: str = None,
    output_paths: Optional[List[str]] = None,
    **kwargs
):
    for distribution_dir, output_path in zip(distribution_dirs, output_paths):
        dataset = Distribution(distribution_dir).test_dataset
        dataset.convert_to_pairs()
        num_examples = len(dataset.examples)
        predictions = [[0.5, 0.5]] * num_examples
        labels = [1] * num_examples
        metrics = compute_metrics((predictions, labels))
        metrics = {"eval_" + key: value for key, value in metrics.items()}
        util.save_json(metrics, output_path)
        print("Saved at", output_path)
    return metrics


def fire_wrap(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
