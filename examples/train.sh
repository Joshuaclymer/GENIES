
export TRAINKWARGS='{"num_train_steps": 20, "per_device_train_batch_size": 32}'
python src/api/train.py \
--model_dir models/pythia-410m \
--intervention_dir src/interventions/lora_fine_tune \
--train_distribution distributions/alpaca_mmlu \
--train_kwargs "$TRAINKWARGS" \
--retrain