# SLURM configuration should go here

export MODEL=$1
export INTERVENTION=$2
export EVALKWARGS=$3
export TRAINKWARGS=$4
export SPLIT=$5

python src/api/compute_generalization_metrics.py \
--base_model_dir models/$MODEL \
--intervention_dir src/interventions/$INTERVENTION \
--target_tuned_capability_intervention src/interventions/$INTERVENTION \
--output_path results/generalization_metrics/$MODEL/$INTERVENTION/$SPLIT.csv \
--path_to_distribution_shift_pairs distribution_shifts/$SPLIT.json \
--eval_kwargs "${EVALKWARGS}" \
--train_kwargs "${TRAINKWARGS}" \