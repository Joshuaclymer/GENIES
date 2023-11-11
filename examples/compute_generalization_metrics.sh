export INTERVENTION='lora_fine_tune'
export SPLIT='test'
export EVALKWARGS='{"max_examples": 100}'
export TRAINKWARGS='{"num_train_steps": 5, "per_device_train_batch_size": 32}'
export MODEL='pythia-410m'

python src/api/compute_generalization_metrics.py \
--base_model_dir models/$MODEL \
--intervention_dir src/interventions/$INTERVENTION \
--target_tuned_capability_intervention src/interventions/lora_fine_tune \
--output_path results/generalization_metrics/$MODEL/$INTERVENTION/$SPLIT.csv \
--path_to_distribution_shift_pairs distribution_shifts/$SPLIT.json \
--eval_kwargs "${EVALKWARGS}" \
--train_kwargs "${TRAINKWARGS}" \
--use_cached_evaluations True \
--retrain_models True \