# Mixture Learner for Partially Annotated Classification (MLPAC)
This repo contains the code for the ACL 2024 paper "[Combining Supervised Learning and Reinforcement Learning for Multi-Label Classification Tasks with Partial Labels]()".

## Abstract
Traditional supervised learning heavily relies on human-annotated datasets, especially in data-hungry neural approaches. However, various tasks, especially multi-label tasks like document-level relation extraction, pose challenges in fully manual annotation due to the specific domain knowledge and large class sets. Therefore, we address the multi-label positive unlabelled learning (MLPUL) problem, where only a subset of positive classes is annotated. We propose Mixture Learner for Partially Annotated Classification (MLPAC), an RL-based framework combining the exploration ability of reinforcement learning and the exploitation ability of supervised learning. Experimental results across various tasks, including document-level relation extraction, multi-label image classification, and binary PU learning, demonstrate the generalization and effectiveness of our framework.


## Requirements
+ Python 3.8
+ Python packages
  + PyTorch 1.11+
  + transformers 4.14+
  + opt-einsum
  + tqdm
  + wandb
  + pandas
  + ujson
  

## Document-Level Relation Extraction Task

### Datasets
We have put [DocRED](https://github.com/thunlp/DocRED/tree/master/data) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED/tree/main/data) in [data/](https://github.com/bigai-nlco/mlpac/tree/main/dreeam_data_arg_with_ns/data).
Original datasets can be downloaded following the instructions at the corresponding links. 

### Training
Perform two-stage training processes: 

1. Pretraining an initiation model by supervised loss;
2. Load the initiation model then continual training by RL.

**STEP1**, supervised pretraining can refer to the codebase of [DREEAM](https://github.com/YoumiMa/dreeam). Note that we adopt a positive up-weight strategy for training the 10% ratio setting.

**STEP2**, we provide examples of run scripts for RL training, for example, run:
```
cd dreeam_data_arg_with_ns
bash scripts/redoc_ns_final/run_roberta50.sh  ${type} ${lambda} ${seed} ${ns_rate}
```
where ``${type}`` is the identifier of this run displayed in wandb, ``${lambda}`` is the scaler that controls the weight of evidence loss, ``${seed}`` is the value of the random seed, and ``${ns_rate}`` is the ratio of negative sampling.
We provide the relative path of datasets and pre-trained models in the config. You can change the data path and model path for your convenience.


### Evaluation
Make predictions on the enhanced test set with the commands below:
```
bash scripts/isf_bert.sh ${name} ${model_dir} ${test_file_path} # for BERT
bash scripts/isf_roberta.sh ${name} ${model_dir} ${test_file_path} # for RoBERTa
```
where ``${model_dir}`` is the directory that contains the checkpoint we are going to evaluate. 
The program will generate a test file ``result.json`` in the official evaluation format. 


## Multi-Label Image Classification Task

### Datasets
We leverage the MS-COCO dataset in this task. 

### Training Models
Perform two-stage training processes: 

1. Pretraining an initiation model by supervised loss;
2. Load the initiation model then continual training by RL.

**STEP1**, run:

```
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
```

**STEP2**, run:

```
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
```

Change the '--data' to your data path, '--simulate_partial_param' is the ratio of unlabeled positive samples, and 'best_epoch' specifies the initiation model to load.





## Citation
```bibtex
@inproceedings{jia-etal-2024-combining,
    title = "Combining Supervised Learning and Reinforcement Learning for Multi-Label Classification Tasks with Partial Labels",
    author = "Jia, Zixia  and
      Li, Junpeng  and
      Zhang, Shichuan  and
      Liu, Anji  and
      Zheng, Zilong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.731",
    doi = "10.18653/v1/2024.acl-long.731",
    pages = "13553--13569",
    abstract = "Traditional supervised learning heavily relies on human-annotated datasets, especially in data-hungry neural approaches. However, various tasks, especially multi-label tasks like document-level relation extraction, pose challenges in fully manual annotation due to the specific domain knowledge and large class sets. Therefore, we address the multi-label positive-unlabelled learning (MLPUL) problem, where only a subset of positive classes is annotated. We propose Mixture Learner for Partially Annotated Classification (MLPAC), an RL-based framework combining the exploration ability of reinforcement learning and the exploitation ability of supervised learning. Experimental results across various tasks, including document-level relation extraction, multi-label image classification, and binary PU learning, demonstrate the generalization and effectiveness of our framework.",
}
```



## Acknowledgements
The codebase of this repo is extended from [DREEAM](https://github.com/YoumiMa/dreeam) and [PartialLabelingCSL](https://github.com/Alibaba-MIIL/PartialLabelingCSL).
