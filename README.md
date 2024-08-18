# mlpac
# Combining Supervised Learning and Reinforcement Learning for Multi-Label Classification Tasks with Partial Labels

## Abstract
Traditional supervised learning heavily relies on human-annotated datasets, especially in data-hungry neural approaches. However, various tasks, especially multi-label tasks like document-level relation extraction, pose challenges in fully manual annotation due to the specific domain knowledge and large class sets. Therefore, we address the multi-label positive unlabelled learning (MLPUL) problem, where only a subset of positive classes is annotated. We propose Mixture Learner for Partially Annotated Classification (MLPAC), an RL-based framework combining the exploration ability of reinforcement learning and the exploitation ability of supervised learning. Experimental results across various tasks, including document-level relation extraction, multi-label image classification, and binary PU learning, demonstrate the generalization and effectiveness of our framework.

![Images](./img/overview.jpg "Trainig process")


## For MS-COCO datatset

the data path as following:

./coco/train2014
./coco/val2014

## Training code
two-stage training, warm up the base model, load the warmed params and train by RL
change the 'data' to your data path, 'simulate_partial_param' is the ratio of unlabeled positive samples, please change the wandb info. 'best_epoch' specify the warm-up model to load.

Warm-up the base model:
python3 train.py \
        --simulate_partial_type=rps \
        --simulate_partial_param=0.5 \
        --partial_loss_mode=negative \
        --likelihood_topk=5 \
        --prior_threshold=0.5 \
        --prior_path=./outputs/priors/prior_fpc_1000.csv \
        --path_dest=./outputs/neg/rps0.5_posWeight_time1 \
        --wandb_id=zhangshichuan \
        --wandb_proj=base_bce_posWeight \
        --epoch=30 \
        --pct_start=0.2 \
        --lr=2e-4 \
        --weight_decay=1e-5

RL Train:
python3 train.py \
        --data='./coco/'
        --simulate_partial_type=rps \
        --simulate_partial_param=0.5 \
        --path_dest=./outputs/neg/RL_rps0.5_time1_iter_ \
        --wandb_id=zhangshichuan \
        --wandb_proj=RL_bce \
        --best_epoch=14 \
        --stage=3 \
        --tunning_mode=pseudo_0.8_0.5_

## Citation:
@inproceedings{jia2024combining,
  title={Combining Supervised Learning and Reinforcement Learning for Multi-Label Classification Tasks with Partial Labels},
  author={Jia, Zixia and Li, Junpeng and Zhang, Shichuan and Liu, Anji and Zheng, Zilong},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={13553--13569},
  year={2024}
}

## Acknowledgements
Some components of this code implementation are adapted from the repository https://github.com/Alibaba-MIIL/PartialLabelingCSL

