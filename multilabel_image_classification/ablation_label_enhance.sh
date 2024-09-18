for ((RATE=9; RATE<10; RATE+=1))
do
        for ((times=1; times<2; times+=1))
        do
                python3 train.py \
                        --simulate_partial_type=rps \
                        --simulate_partial_param=0.${RATE} \
                        --path_dest=./outputs/neg/RL_rps${RATE}_time${times}_iter_ \
                        --wandb_id=zhangshichuan \
                        --wandb_proj=RL_ablation \
                        --best_epoch=14 \
                        --stage=3 \
                        --tunning_mode=label_enhance \
                        --ablation_mode=label_enhance
        done
done
