TYPE=$1
LAMBDA=$2
SEED=$3
ns_rate=$4

for RATE in 100;
do
  NAME=${TYPE}_datarate${RATE}_value_ns${ns_rate}
   
  python run.py --do_train --add_recall --add_value \
  --data_dir data/ \
  --transformer_type roberta \
  --load_value_path ckpt/sup_ns_final/docred_roberta_lambda0.05_ns10_rate100/2024-01-24_13:51:57.143670 \
  --model_name_or_path bert_models/roberta-large \
  --display_name  ${NAME} \
  --train_file docred/train_annotated.json \
  --dev_file docred/dev.json \
  --test_file docred/test_revised.json \
  --save_path ckpt/docred_output/final/${NAME} \
  --train_batch_size 8 \
  --test_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --lr_transformer 5e-6 \
  --lr_added 3e-5 \
  --max_grad_norm 1.0 \
  --evi_thresh 0.2 \
  --rl_weight 10.0 \
  --ns_rate_for_value ${ns_rate} \
  --evi_lambda ${LAMBDA} \
  --warmup_ratio 0.06 \
  --num_train_epochs 30.0 \
  --seed ${SEED} \
  --num_class 97

  
done
