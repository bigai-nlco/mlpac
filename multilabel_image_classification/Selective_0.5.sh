for ((RATE=3; RATE<4; RATE+=1))
do
        for ((times=1; times<2; times+=1))
        do
                python train.py \
                        --gamma_pos=0 \
                        --gamma_neg=1 \
                        --gamma_unann=1 \
                        --simulate_partial_type=rps \
                        --simulate_partial_param=0.${RATE} \
                        --partial_loss_mode=selective \
                        --likelihood_topk=5 \
                        --prior_threshold=0.5 \
                        --prior_path=./outputs/priors/prior_rps_0.${RATE}_P.csv \
                        --path_dest=./outputs/neg/Sel_rps${RATE}_time${times} \
                        --wandb_proj=selective \
                        --wandb_id=zhangshichuan \
                        --epoch=30 \
                        --pct_start=0.2
        done
done
