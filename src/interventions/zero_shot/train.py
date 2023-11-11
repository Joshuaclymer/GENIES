from typing import List, Optional

import fire

import api.util as util

def main(
    model_dir: str,
    output_dir: str,
    training_distribution_dir: str,
    test_distribution_dirs: List[str],
) -> Optional[List[float]]:
    config = {
        "model_dir": model_dir,
        "training_distribution_dir": training_distribution_dir,
    }
    util.save_json(config, f"{output_dir}/config.json")


def fire_wrap(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
