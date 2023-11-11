import os
import random
from api.data_classes import Distribution
import api.util as util
import fire
import json
from typing import Optional, List
from api.evaluate import evaluate
from api.train import train


def main(
    base_model_dir: str,
    intervention_dir: str,
    output_path: str,
    path_to_distribution_shift_pairs: Optional[str] == None,
    source_dirs: Optional[str] = None,
    target_dirs: Optional[str] = None,
    dry_run: bool = False,
    baseline_intervention_dir: str = "src/interventions/pro",
    use_cached_evaluations: bool = True,
    retrain_models: bool = False,
    train_kwargs: Optional[dict] = {},
    eval_kwargs: Optional[dict] = {},
):

    # Check that the arguments are valid
    if path_to_distribution_shift_pairs == None and (source_dirs == None or target_dirs == None):
        raise ValueError(
            "Must provide either path_to_distribution_shift_pairs or paths_to_source and target_dirs"
        )

    # Parse pairs.json file if it was provided into source_dirs and target_dirs
    pairs_data = util.load_json(path_to_distribution_shift_pairs)
    relative_source_dirs = [pair["source"] for pair in pairs_data]
    relative_target_dirs = [pair["target"] for pair in pairs_data]
    source_dirs = [f"distributions/{dir}" for dir in relative_source_dirs]
    target_dirs = [f"distributions/{dir}" for dir in relative_target_dirs]

    # Load distributions to obtain meta data
    source_distributions = [Distribution(dir) for dir in source_dirs]
    target_distributions = [Distribution(dir) for dir in target_dirs]

    # Initialize output data with empty values
    pgc_result = [
        {
            "source": source.id,
            "target": target.id,
            "pgc": None,
            "performance_of_base_model": None,
            "performance_trained_on_source": None,
            "performance_trained_on_target": None,
        }
        for source, target in zip(source_distributions, target_distributions)
    ]

    util.print_once("")
    util.print_once("----------- Measuring baselines -----------")
    util.print_once("")

    unique_target_distributions = list(set(target_distributions))
    unique_target_ids = [t.id for t in unique_target_distributions]

    # Evaluate the base model on the target distributions

    evaluations = evaluate(
        model_dir=base_model_dir,
        distribution_dirs=unique_target_distributions,
        intervention_dir=baseline_intervention_dir,
        use_cached=use_cached_evaluations,
        dry_run=dry_run,
        eval_kwargs=eval_kwargs,
    )

    # Write evaluation to the pgc result file
    for target_id, evaluation in zip(unique_target_ids, evaluations):
        for item in pgc_result:
            if item["target"] == target_id:
                item["performance_of_base_model"] = evaluation["average_score"]

    util.save_json(pgc_result, output_path)
    util.print_once(f"Saved intermediate results in {output_path}")

    # For each target distribution, train the base model on the distribution and then evaluate its performance.
    for target_distribution in unique_target_distributions:
        output_model_dir, logs = train(
            base_model_dir,
            target_distribution,
            intervention_dir,
            dry_run=dry_run,
            train_kwargs=train_kwargs,
            retrain=retrain_models,
        )
        evaluations = evaluate(
            output_model_dir,
            [target_distribution],
            intervention_dir,
            use_cached=use_cached_evaluations,
            dry_run=dry_run,
            eval_kwargs=eval_kwargs,
        )
        for item in pgc_result:
            if item["target"] == target_distribution.id:
                item["performance_trained_on_target"] = evaluations[0]["average_score"]
        util.save_json(pgc_result, output_path)
        util.print_once(f"Saved intermediate results in {output_path}")

    util.print_once("")
    util.print_once("----------- Measuring generalization -----------")
    util.print_once("")

    # Train the base model on each source distribution and then evaluate its performance on all target distributions that it is paired with
    unique_source_distributions = list(set(source_distributions))

    for source_distribution in unique_source_distributions:
        output_model_dir, logs = train(
            base_model_dir,
            source_distribution,
            intervention_dir,
            train_kwargs=train_kwargs,
            retrain=retrain_models,
            dry_run=dry_run,
        )

        # Get the target distributions that this source distribution is paired with
        indices_of_pairs_source_is_part_of = [
            i for i, s in enumerate(source_distributions) if source_distribution == s
        ]
        target_distributions_for_source = [
            target_distributions[i] for i in indices_of_pairs_source_is_part_of
        ]

        evaluations = evaluate(
            output_model_dir,
            target_distributions_for_source,
            intervention_dir,
            use_cached=use_cached_evaluations,
            dry_run=dry_run,
            eval_kwargs=eval_kwargs,
        )

        for evaluation, target_distribution in zip(evaluations, target_distributions_for_source):
            for item in pgc_result:
                if (
                    item["source"] == source_distribution.id
                    and item["target"] == target_distribution.id
                ):
                    item["performance_trained_on_source"] = evaluations[0]["average_score"]
        util.save_json(pgc_result, output_path)
        util.print_once(f"Saved intermediate results in {output_path}")

    for item in pgc_result:
        item["pgc"] = (
            item["performance_trained_on_source"] - item["performance_of_base_model"]
        ) / (item["performance_trained_on_target"] - item["performance_of_base_model"])
    pgcs_to_string = "\n".join(
        [
            f"{item['source']} -> {item['target']}: pgc: {item['pgc']}, base: {item['performance_of_base_model']}, source: {item['performance_trained_on_source']}, target: {item['performance_trained_on_target']}"
            for item in pgc_result
        ]
    )

    util.save_json(pgc_result, output_path)
    util.print_once("")
    util.print_once(f"Results saved at {output_path}")
    util.print_once(f"PGC: {pgcs_to_string}")
    util.print_once("")


if __name__ == "__main__":
    fire.Fire(main)
