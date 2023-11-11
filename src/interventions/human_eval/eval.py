import datetime
from typing import List, Optional

import fire

import api.util as util
from api.data_classes import Distribution
from api.evaluate import compute_metrics


def main(
    distribution_dirs: List[str],
    model_dir: str = None,
    output_paths: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
) -> List[dict]:
    if len(distribution_dirs) > 1:
        raise ValueError(
            "Only one distribution can be evaluated at a time with human eval."
        )
    dataset = Distribution(distribution_dirs[0]).test_dataset
    dataset.convert_to_pairs(one_pair_per_instruction=True)
    dataset.set_max_examples(max_examples)

    evaluation = {
        "model_name": "human",
        "distribution_id": dataset.distribution_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    predictions = []
    labels = []
    for i, example in enumerate(dataset.examples):
        print("-----------------------------------------------------------------")
        print(f"Example {i + 1}/{len(dataset.examples)}")
        print("-----------------------------------------------------------------")

        prompt = example["prompt"]
        answers = list(example["responses"].keys())
        correct_answer = [
            r for r in example["responses"] if example["responses"][r] == 1
        ][0]
        correct_index = answers.index(correct_answer)
        letters = ["A", "B", "C", "D"][: len(answers)]

        print(prompt)

        # show answers in terminal with letters
        for i, answers in enumerate(answers):
            print(f"{letters[i]}) {answers}\n\n")
        passed = False
        while True:
            user_input = input(
                f"Which of the above is the best response? (type {', '.join(letters)}, or P for pass) "
            )
            user_input = user_input.upper().strip()
            if user_input in letters + ["P"]:
                break
            else:
                print("Invalid input. Try again.")
        if user_input == "P":
            passed = True
            continue

            # if user_input == letters[correct_index]:
            #     break
            # else:
            #     while True:
            #         confirmation = input(f"The labeled option is {letters[correct_index]}. Type Y to confirm that the answer you typed ({user_input}) is the best completion. Type N to go back and change your answer.")
            #         if confirmation in ["Y", "N"]:
            #             break
            #         else:
            #             print("Invalid input. Try again.")
            #     if confirmation == "Y":
            #         break
        if not passed:
            is_correct = float(user_input == letters[correct_index])
            predictions.append([1 - is_correct, is_correct])
            labels.append(1)

    metrics = compute_metrics((predictions, labels))
    metrics["eval_score"] = metrics["score"]
    evaluation.update(metrics)
    util.save_json(evaluation, output_paths[0])
    return [evaluation]


if __name__ == "__main__":
    fire.Fire(main)
