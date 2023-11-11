# IMPORTANT NOTE: these scripts must be run from the src/gen_data directory

source activate newenv
export path_to_class_def=distribution_definitions/pursue_goals.py 
export distribution_class_name=PursueGoals
export dir_to_output_data=data/pursue_goals 

mpirun --allow-run-as-root --oversubscribe -np 10 python gen_utils/generate_instructions.py \
    --distribution_class_name $distribution_class_name \
    --path_to_class_def $path_to_class_def \
    --dir_to_output_data $dir_to_output_data \
    --num_instructions_to_generate 1200 \
    --num_prompt_instructions 3 \

python gen_utils/make_splits.py \
    --dir_to_input_data $dir_to_output_data/examples.json \
    --dir_to_output_data ./../../distributions \
    --path_to_class_def $path_to_class_def \
    --class_name $distribution_class_name \
    --desired_num_responses 2