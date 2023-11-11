export MODEL=pythia-410m
export INTERVENTION=prompt_tuning
sbatch --job-name "hps-$MODEL-$INTERVENTION" --output "logs/hps/$INTERVENTION/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION 

export MODEL=pythia-1b
sbatch --job-name "hps-$MODEL-$INTERVENTION" --output "logs/hps/$INTERVENTION/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION 

export MODEL=pythia-2.7b
sbatch --job-name "hps-$MODEL-$INTERVENTION" --output "logs/hps/$INTERVENTION/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION 

export MODEL=pythia-6.7b
sbatch --job-name "hps-$MODEL-$INTERVENTION" --output "logs/hps/$INTERVENTION/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION 

export MODEL=llama-7b
sbatch --job-name "hps-$MODEL-$INTERVENTION" --output "logs/hps/$INTERVENTION/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION 

export MODEL=llama-13b
sbatch --job-name "hps-$MODEL-$INTERVENTION" --output "logs/hps/$INTERVENTION/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION 

export MODEL=llama-30b
sbatch --job-name "hps-$MODEL-$INTERVENTION" --output "logs/hps/$INTERVENTION/$MODEL.out" experiments/hps.sh $MODEL $INTERVENTION 