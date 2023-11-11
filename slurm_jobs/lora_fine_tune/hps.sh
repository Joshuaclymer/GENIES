export MODEL=tinyllama-1b
export INTERVENTION=classify_lora
sbatch --job-name "hps-$MODEL" --output "logs/hps/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION
# export MODEL=openllama-3b
# sbatch --job-name "hps-$MODEL" --output "logs/hps/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION