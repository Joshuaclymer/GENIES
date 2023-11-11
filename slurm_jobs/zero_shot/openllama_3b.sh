export MODEL=openllama-3b
export INTERVENTION=zero_shot
export EVALKWARGS='{"per_device_batch_size": 32}'
export TRAINKWARGS='{}'

CATEGORY=$1
sbatch --gpus 1 --job-name "$CATEGORY-$MODEL-$INTERVENTION" --output "logs/$CATEGORY-$MODEL-$INTERVENTION.out" scripts/compute_generalization_metrics.sh "$MODEL" "$INTERVENTION" "$EVALKWARGS" "$TRAINKWARGS" "$CATEGORY"