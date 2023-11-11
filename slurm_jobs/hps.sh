# SLURM configuration should go here

export INTERVENTION=$2

python src/api/hyperparameter_sweep.py \
--model_dir models/$1 \
--intervention_dir src/interventions/$2 \
--distribution distributions/alpaca_mmlu \
--train_kwargs '{"max_steps": 100, "per_device_train_batch_size": 32}' \
--count 8