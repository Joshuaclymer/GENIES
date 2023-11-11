# IMPORTANT NOTE: these scripts must be run from the src/gen_data directory

export PATH_TO_CLASS_DEF=distribution_definitions/us_history.py 
export DISTRIBUTION_CLASS_NAME=USHistory
export DIR_TO_OUTPUT_DATA=data/us_history 

mpirun --allow-run-as-root --oversubscribe -np 10 python gen_utils/generate_instructions.py \
    --distribution_class_name $DISTRIBUTION_CLASS_NAME \
    --path_to_class_def $PATH_TO_CLASS_DEF \
    --dir_to_output_data $DIR_TO_OUTPUT_DATA \
    --num_instructions_to_generate 1200 \

mpirun --allow-run-as-root --oversubscribe -np 10 python gen_utils/generate_options.py \
    --distribution_class_name $DISTRIBUTION_CLASS_NAME \
    --path_to_class_def $PATH_TO_CLASS_DEF \
    --dir_to_output_data $DIR_TO_OUTPUT_DATA \

mpirun --allow-run-as-root --oversubscribe -np 2 python gen_utils/filter_examples.py \
    --distribution_class_name $DISTRIBUTION_CLASS_NAME \
    --path_to_class_def $PATH_TO_CLASS_DEF \
    --dir_to_output_data $DIR_TO_OUTPUT_DATA \

python gen_utils/make_splits.py \
    --dir_to_input_data $DIR_TO_OUTPUT_DATA/filtered.json \
    --dir_to_output_data ./../../distributions\
    --path_to_class_def $PATH_TO_CLASS_DEF \
    --class_name $DISTRIBUTION_CLASS_NAME \