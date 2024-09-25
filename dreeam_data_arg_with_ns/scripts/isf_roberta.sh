NAME=$1
MODEL_DIR=$2
SPLIT=$3

python run.py --data_dir data/docred \
--transformer_type roberta \
--model_name_or_path bert_models/roberta-large \
--display_name  ${NAME} \
--load_value_path ${MODEL_DIR} \
--load_path ${MODEL_DIR} \
--eval_mode single \
--test_file ${SPLIT}.json \
--test_batch_size 8 \
--evi_thresh 0.2 \
--num_labels 4 \
--num_class 97 

