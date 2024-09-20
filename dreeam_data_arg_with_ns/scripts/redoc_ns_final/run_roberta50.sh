TYPE=$1
LAMBDA=$2
SEED=$3
ns_rate=$4

for RATE in 50;
do
  NAME=${TYPE}_datarate${RATE}_value_ns${ns_rate}

  python run.py --do_train --add_recall --add_value \
  --data_dir /scratch/nlp/jiazixia/rl_enhanced/dreeam_pos_neg_rl/dataset/docred \
  --transformer_type roberta \
  --load_value_path /scratch/nlp/jiazixia/rl_enhanced/dreeam/redocred_output/sup_ns_final/ns_lambda0.05_ns10_rate${RATE}/2023-09-21_16:18:27.019116 \
  --model_name_or_path /scratch/nlp/jiazixia/dreeam/bert_models/roberta-large \
  --display_name  ${NAME} \
  --train_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/redocred/sample_data_for_dreeam/triple_sample_data0/sample_${RATE}_percent_data/train.json \
  --dev_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/redocred/sample_data_for_dreeam/triple_sample_data0/sample_${RATE}_percent_data/dev.json \
  --test_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/docred/test_revised.json \
  --save_path redocred_output/final/${NAME} \
  --train_batch_size 8 \
  --test_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --lr_transformer 5e-6 \
  --lr_added 3e-5 \
  --max_grad_norm 1.0 \
  --evi_thresh 0.2 \
  --rl_weight 10.0 \
  --sample_times 1 \
  --ns_rate_for_value ${ns_rate} \
  --evi_lambda ${LAMBDA} \
  --warmup_ratio 0.06 \
  --num_train_epochs 30.0 \
  --seed ${SEED} \
  --num_class 97

  python run.py --do_train --add_recall --add_value \
  --data_dir /scratch/nlp/jiazixia/rl_enhanced/dreeam_pos_neg_rl/dataset/docred \
  --transformer_type roberta \
  --load_value_path /scratch/nlp/jiazixia/rl_enhanced/dreeam/redocred_output/sup_ns_final/ns_lambda0.05_ns10_rate${RATE}/2023-09-21_18:39:14.991177 \
  --model_name_or_path /scratch/nlp/jiazixia/dreeam/bert_models/roberta-large \
  --display_name  ${NAME} \
  --train_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/redocred/sample_data_for_dreeam/triple_sample_data1/sample_${RATE}_percent_data/train.json \
  --dev_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/redocred/sample_data_for_dreeam/triple_sample_data1/sample_${RATE}_percent_data/dev.json \
  --test_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/docred/test_revised.json \
  --save_path redocred_output/final/${NAME} \
  --train_batch_size 8 \
  --test_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --lr_transformer 5e-6 \
  --lr_added 3e-5 \
  --max_grad_norm 1.0 \
  --evi_thresh 0.2 \
  --rl_weight 10.0 \
  --sample_times 1 \
  --ns_rate_for_value ${ns_rate} \
  --evi_lambda ${LAMBDA} \
  --warmup_ratio 0.06 \
  --num_train_epochs 30.0 \
  --seed ${SEED} \
  --num_class 97

  python run.py --do_train --add_recall --add_value \
  --data_dir /scratch/nlp/jiazixia/rl_enhanced/dreeam_pos_neg_rl/dataset/docred \
  --transformer_type roberta \
  --load_value_path /scratch/nlp/jiazixia/rl_enhanced/dreeam/redocred_output/sup_ns_final/ns_lambda0.05_ns10_rate${RATE}/2023-09-21_20:59:48.012870 \
  --model_name_or_path /scratch/nlp/jiazixia/dreeam/bert_models/roberta-large \
  --display_name  ${NAME} \
  --train_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/redocred/sample_data_for_dreeam/triple_sample_data2/sample_${RATE}_percent_data/train.json \
  --dev_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/redocred/sample_data_for_dreeam/triple_sample_data2/sample_${RATE}_percent_data/dev.json \
  --test_file /scratch/nlp/jiazixia/rl_enhanced/dreeam/dataset/docred/test_revised.json \
  --save_path redocred_output/final/${NAME} \
  --train_batch_size 8 \
  --test_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --lr_transformer 5e-6 \
  --lr_added 3e-5 \
  --max_grad_norm 1.0 \
  --evi_thresh 0.2 \
  --rl_weight 10.0 \
  --sample_times 1 \
  --ns_rate_for_value ${ns_rate} \
  --evi_lambda ${LAMBDA} \
  --warmup_ratio 0.06 \
  --num_train_epochs 30.0 \
  --seed ${SEED} \
  --num_class 97

done
