export MODEL=llama-30b
export INTERVENTION=ccs
export EVALKWARGS='{}'
export TRAINKWARGS='{"max_examples": 300}'

CATEGORY=$1
sbatch --gpus 1 --job-name "$CATEGORY-$MODEL-$INTERVENTION" --output "logs/$CATEGORY-$MODEL-$INTERVENTION.out" scripts/compute_generalization_metrics.sh "$MODEL" "$INTERVENTION" "$EVALKWARGS" "$TRAINKWARGS" "$CATEGORY"