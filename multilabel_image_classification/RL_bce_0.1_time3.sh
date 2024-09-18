for ((RATE=1; RATE<2; RATE+=2))
do
        for ((times=3; times<4; times+=1))
        do
                python3 train.py \
                        --simulate_partial_type=rps \
                        --simulate_partial_param=0.${RATE} \
                        --path_dest=./outputs/neg/RL_rps${RATE}_time${times}_iter_ \
                        --wandb_id=zhangshichuan \
                        --wandb_proj=RL_bce \
                        --best_epoch=14 \
                        --stage=3 \
                        --tunning_mode=pseudo_0.8_0.5_
        done
done
