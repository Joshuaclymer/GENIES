export MODEL=llama-13b
export INTERVENTION=classify_lora
export EVALKWARGS='{"per_device_batch_size": 16}'
export TRAINKWARGS='{"per_device_train_batch_size": 16, "per_device_eval_batch_size": 16, "gradient_accumulation_steps": 1, "learning_rate": 2e-4, num_train_steps: 150}'

CATEGORY=$1
sbatch --gpus 1 --job-name "$CATEGORY-$MODEL-$INTERVENTION" --output "logs/$CATEGORY-$MODEL-$INTERVENTION.out" scripts/compute_generalization_metrics.sh "$MODEL" "$INTERVENTION" "$EVALKWARGS" "$TRAINKWARGS" "$CATEGORY"