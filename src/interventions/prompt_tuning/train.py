import os
import shutil
import time
from typing import List, Optional

import fire
from peft import PromptTuningConfig, TaskType

import api.util as util
import interventions.classify_lora.train as classify_train


def main(
    model_dir: str,
    output_dir: str,
    training_distribution_dir: str,
    test_distribution_dirs: Optional[List[str]] = None,
    max_eval_examples: int = 100,
    max_train_examples: int = None,
    peft_config=None,
    num_train_steps: int = 150,
    **kwargs,
) -> Optional[List[float]]:
    peft_config = PromptTuningConfig(
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=10,
    )
    if os.path.exists(output_dir) and util.is_main():
        print(
            "Resume from checkpoint not supported for prompt tuning. Deleting old checkpoints."
        )
        shutil.rmtree(output_dir)
    time.sleep(30)
    return classify_train.main(
        model_dir=model_dir,
        output_dir=output_dir,
        training_distribution_dir=training_distribution_dir,
        test_distribution_dirs=test_distribution_dirs,
        max_eval_examples=max_eval_examples,
        max_train_examples=max_train_examples,
        peft_config=peft_config,
        num_train_steps=num_train_steps,
        **kwargs,
    )


def fire_wrap(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(fire_wrap)
