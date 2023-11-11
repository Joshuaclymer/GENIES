
python src/api/evaluate.py \
--model_dir models/lora_fine_tune/pythia-410m-alpaca_mmlu \
--intervention_dir src/interventions/lora_fine_tune \
--distributions '["distributions/alpaca_mmlu"]' \
--use_cached False \
--eval_kwargs '{"max_examples": 100}' \