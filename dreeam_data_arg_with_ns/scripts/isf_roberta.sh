NAME=$1
MODEL_DIR=$2
SPLIT=$3

python run.py --data_dir /scratch/nlp/jiazixia/rl_enhanced/dreeam_pos_neg_rl/dataset/docred \
--transformer_type roberta \
--model_name_or_path /scratch/nlp/jiazixia/dreeam/bert_models/roberta-large \
--display_name  ${NAME} \
--load_value_path ${MODEL_DIR} \
--load_path ${MODEL_DIR} \
--eval_mode single \
--test_file ${SPLIT}.json \
--test_batch_size 8 \
--evi_thresh 0.2 \
--num_labels 4 \
--num_class 97 

#python run.py --data_dir dataset/docred \
#--transformer_type roberta \
#--model_name_or_path roberta-large \
#--display_name  ${NAME} \
#--load_path ${MODEL_DIR} \
#--eval_mode fushion \
#--test_file ${SPLIT}.json \
#--test_batch_size 8 \
#--num_labels 4 \
#--evi_thresh 0.2 \
#--num_class 97
