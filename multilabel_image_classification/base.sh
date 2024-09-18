for ((RATE=9; RATE<10; RATE+=1))
do
        for ((times=1; times<2; times+=1))
        do
                python3 train.py \
                        --gamma_pos=0 \
                        --gamma_neg=1 \
                        --gamma_unann=4 \
                        --simulate_partial_type=rps \
                        --simulate_partial_param=0.${RATE} \
                        --partial_loss_mode=negative \
                        --likelihood_topk=5 \
                        --prior_threshold=0.5 \
                        --prior_path=./outputs/priors/prior_fpc_1000.csv \
                        --path_dest=./outputs/neg/rps${RATE}_posWeight_time${times} \
		        --wandb_id=zhangshichuan \
			--wandb_proj=base_bce_posWeight \
			--epoch=30 \
                        --pct_start=0.2 \
			--lr=2e-4 \
		        --weight_decay=1e-5
        done
done
