export MODEL=openllama-3b
export INTERVENTION=classify_lora
export EVALKWARGS='{"per_device_batch_size": 32}'
export TRAINKWARGS='{"per_device_train_batch_size": 16, "per_device_eval_batch_size": 32, "gradient_accumulation_steps": 1, "learning_rate": 7.2e-5, "num_train_steps": 100}'

CATEGORY=$1

sbatch --gpus 1 --job-name "$CATEGORY-$MODEL-$INTERVENTION" --output "logs/$CATEGORY-$MODEL-$INTERVENTION.out" scripts/compute_generalization_metrics.sh "$MODEL" "$INTERVENTION" "$EVALKWARGS" "$TRAINKWARGS" "$CATEGORY"