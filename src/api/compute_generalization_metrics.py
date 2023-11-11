import os
from typing import Optional

import fire

import api.util as util
from api.data_classes import Distribution
from api.evaluate import evaluate
from api.train import train

def main(
    base_model_dir: str,
    intervention_dir: str,
    output_path: str,
    path_to_distribution_shift_pairs: Optional[str] == None,
    source_dirs: Optional[str] = None,
    target_dirs: Optional[str] = None,
    target_tuned_capability_intervention: str = "src/interventions/lora_fine_tune",
    baseline_intervention: str = "src/interventions/zero_shot",
    train_kwargs: Optional[dict] = {},
    eval_kwargs: Optional[dict] = {},
    retrain_models: bool = False,
    dry_run: bool = False,
    use_cached_evaluations: bool = True,
):
    # Check that the arguments are valid
    if path_to_distribution_shift_pairs == None and (
        source_dirs == None or target_dirs == None
    ):
        raise ValueError(
            "Must provide either path_to_distribution_shift_pairs or paths_to_source and target_dirs"
        )
    if not os.path.exists(intervention_dir):
        raise ValueError(f"Intervention directory {intervention_dir} does not exist")

    if not os.path.exists(target_tuned_capability_intervention):
        raise ValueError(
            f"Upper reference intervention directory {target_tuned_capability_intervention} does not exist"
        )

    target_tuned_capability_intervention_name = os.path.basename(target_tuned_capability_intervention)
    baseline_intervention_name = os.path.basename(baseline_intervention)

    # Parse pairs.json file if it was provided into source_dirs and target_dirs
    pairs_data = util.load_json(path_to_distribution_shift_pairs)
    relative_source_dirs = [pair["source"] for pair in pairs_data]
    relative_target_dirs = [pair["target"] for pair in pairs_data]
    relative_target_reference_dirs = [
        pair["target_reference"] if "target_reference" in pair else pair["target"]
        for pair in pairs_data
    ]
    source_dirs = [f"distributions/{dir}" for dir in relative_source_dirs]
    target_dirs = [f"distributions/{dir}" for dir in relative_target_dirs]
    target_reference_dirs = [
        f"distributions/{dir}" for dir in relative_target_reference_dirs
    ]

    # Load up the source and target distributions to retrieve their meta data
    source_distributions = [Distribution(dir) for dir in source_dirs]
    target_distributions = [Distribution(dir) for dir in target_dirs]
    target_reference_distributions = [
        Distribution(dir) for dir in target_reference_dirs
    ]

    # Initialize output data with empty values
    generalization_metrics = [
        {
            "source": source.id,
            "target": target.id,
            "target_reference": target_reference.id,
            "generalization_accuracy": None,
            "baseline_accuracy": None,
            "baseline_intervention": baseline_intervention_name,
            "target_tuned_capability": None,
            "target_tuned_capability_intervention": target_tuned_capability_intervention_name,
            "source_ID_accuracy": None,
            "RMS_calibration_error": None,
            "differential_elicitation": None,
        }
        for source, target, target_reference in zip(
            source_distributions, target_distributions, target_reference_distributions
        )
    ]

    print("")
    print("----------- Measuring Baseline Target Accuracy -----------")
    print("")

    unique_target_distributions = list(set(target_distributions))
    unique_target_ids = [t.id for t in unique_target_distributions]

    unique_target_reference_distributions = list(set(target_reference_distributions))

    # Evaluate the base model on the target distributions
    evaluations = evaluate(
        model_dir=base_model_dir,
        distributions=unique_target_distributions,
        intervention_dir=baseline_intervention,
        use_cached=use_cached_evaluations,
        dry_run=dry_run,
        eval_kwargs=eval_kwargs,
    )

    # Write evaluation to the pgc result file
    for target_id, evaluation in zip(unique_target_ids, evaluations):
        for item in generalization_metrics:
            if item["target"] == target_id:
                item[f"baseline_accuracy"] = evaluation["eval_accuracy"]

    util.save_as_csv(generalization_metrics, output_path)
    print(f"Saved intermediate results in {output_path}")

    print("")
    print("----------- Measuring Target-tuned Capability -----------")
    print("")
    # For each target distribution, train the base model on the distribution and then evaluate its performance.
    for target_distribution in unique_target_reference_distributions:
        output_model_dir = train(
            base_model_dir,
            target_distribution.dir,
            target_tuned_capability_intervention,
            dry_run=dry_run,
            train_kwargs=train_kwargs,
            retrain=retrain_models,
        )
        evaluations = evaluate(
            output_model_dir,
            [target_distribution],
            target_tuned_capability_intervention,
            use_cached=use_cached_evaluations or retrain_models,
            dry_run=dry_run,
            eval_kwargs=eval_kwargs,
        )
        for item in generalization_metrics:
            if item["target_reference"] == target_distribution.id:
                item["target_tuned_capability"] = evaluations[0]["eval_accuracy"]

        util.save_as_csv(generalization_metrics, output_path)
        print(f"Saved intermediate results in {output_path}")

    print("")
    print("----------- Measuring Generalization -----------")
    print("")

    # Train the base model on each source distribution and then evaluate its performance on all target distributions that it is paired with
    unique_source_distributions = list(set(source_distributions))

    for source_distribution in unique_source_distributions:
        # Get the target distributions that this source distribution is paired with
        indices_of_pairs_source_is_part_of = [
            i for i, s in enumerate(source_distributions) if source_distribution == s
        ]
        target_distributions_for_source = [
            target_distributions[i] for i in indices_of_pairs_source_is_part_of
        ]

        output_model_dir = train(
            base_model_dir,
            source_distribution.dir,
            intervention_dir,
            eval_distributions=[d.dir for d in target_distributions_for_source],
            train_kwargs=train_kwargs,
            retrain=retrain_models,
            dry_run=dry_run,
        )

        evaluations = evaluate(
            output_model_dir,
            target_distributions_for_source + [source_distribution],
            intervention_dir,
            use_cached=use_cached_evaluations or retrain_models,
            dry_run=dry_run,
            eval_kwargs=eval_kwargs,
        )

        source_evaluation = evaluations[-1]
        for evaluation, target_distribution in zip(
            evaluations[:-1], target_distributions_for_source
        ):
            for item in generalization_metrics:
                if (
                    item["source"] == source_distribution.id
                    and item["target"] == target_distribution.id
                ):
                    item[f"generalization_accuracy"] = evaluation[
                        "eval_accuracy"
                    ]
                    item[f"source_ID_accuracy"] = source_evaluation[
                        "eval_accuracy"
                    ]
                    if "eval_rms_calibration_error" in evaluation:
                        item["RMS_calibration_error"] = evaluation[
                            "eval_rms_calibration_error"
                        ]

        util.save_as_csv(generalization_metrics, output_path)
        print(f"Saved intermediate results in {output_path}")

    print([item['generalization_accuracy'] for item in generalization_metrics])
    print([item['baseline_accuracy'] for item in generalization_metrics])
    for item in generalization_metrics:
        baseline_accuracy = item["baseline_accuracy"]
        target_tuned_capability = item["target_tuned_capability"]
        generalization_accuracy = item["generalization_accuracy"]
        item["differential_elicitation"] = (generalization_accuracy - baseline_accuracy) / target_tuned_capability

    generalization_metrics_stringified = "\n".join(
        [
            f"{item['source']} -> {item['target']}: {item['differential_elicitation']}"
            for item in generalization_metrics
        ]
    )

    util.save_as_csv(generalization_metrics, output_path)
    print("")
    print(f"Results saved at {output_path}")
    print(generalization_metrics_stringified)
    print("")


if __name__ == "__main__":
    fire.Fire(main)
