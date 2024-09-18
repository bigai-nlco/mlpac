wandb: Currently logged in as: zhangshichuan. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.15.11 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.15.0
wandb: Run data is saved locally in /scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL-zsc/wandb/run-20230928_004051-z4clg5rt
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run train__lrsd_rpneg_base_bce_negSampl0.6
wandb: ⭐️ View project at https://wandb.ai/zhangshichuan/base_bce_negSampl
wandb: 🚀 View run at https://wandb.ai/zhangshichuan/base_bce_negSampl/runs/z4clg5rt
creating model...
done

loading annotations into memory...
Done (t=3.84s)
creating index...
index created!
loading annotations into memory...
Done (t=7.94s)
creating index...
index created!
len(val_dataset)):  40137
len(train_dataset)):  82081
Original stat: [0.55035879 0.02786272 0.10484765 0.0297511  0.02732667 0.034003
 0.03001913 0.05264312 0.02556012 0.03524567]
Simulate coco. Mode: rps. Param: 0.600000
Simulated stat: [0.21988036 0.01113534 0.042787   0.01193943 0.01130591]
Used parameters:
Image_size: 448
Learning_rate: 0.0002
Epochs: 30
Prior file was found in given path.
Prior file was loaded successfully. 
Epoch [0/30], Step [000/1283], LR 8.0e-06, Loss: 779.3
Epoch [0/30], Step [100/1283], LR 8.1e-06, Loss: 624.7
Epoch [0/30], Step [200/1283], LR 8.3e-06, Loss: 433.5
Epoch [0/30], Step [300/1283], LR 8.7e-06, Loss: 277.9
Epoch [0/30], Step [400/1283], LR 9.3e-06, Loss: 227.7
Epoch [0/30], Step [500/1283], LR 1.0e-05, Loss: 255.6
Epoch [0/30], Step [600/1283], LR 1.1e-05, Loss: 261.9
Epoch [0/30], Step [700/1283], LR 1.2e-05, Loss: 187.8
Epoch [0/30], Step [800/1283], LR 1.3e-05, Loss: 191.8
Epoch [0/30], Step [900/1283], LR 1.4e-05, Loss: 186.2
Epoch [0/30], Step [1000/1283], LR 1.6e-05, Loss: 150.0
Epoch [0/30], Step [1100/1283], LR 1.8e-05, Loss: 209.8
Epoch [0/30], Step [1200/1283], LR 1.9e-05, Loss: 159.9
starting validation
/scratch/jiazixia/.conda/envs/docre_clone/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:131: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
mAP score regular 45.71, mAP score EMA 9.09
F1 score regular 46.67, Precision regular 78.88, Recall regular 33.14, F1 score EMA 18.44, Precision EMA 13.00, Recall EMA31.70
best model Saving successfully.
current_F1 = 46.67, highest_F1 = 46.67

Prior (train), first 5 classes: tensor([0.5416, 0.1154, 0.2209, 0.1347, 0.1226], device='cuda:0')
Prior (train), first 10 classes: ['person' 'car' 'dining table' 'chair' 'cup' 'bottle' 'backpack' 'handbag'
 'truck' 'bowl']
Epoch [1/30], Step [000/1283], LR 2.1e-05, Loss: 202.8
Epoch [1/30], Step [100/1283], LR 2.3e-05, Loss: 149.8
Epoch [1/30], Step [200/1283], LR 2.5e-05, Loss: 150.5
Epoch [1/30], Step [300/1283], LR 2.7e-05, Loss: 180.3
Epoch [1/30], Step [400/1283], LR 3.0e-05, Loss: 158.7
Epoch [1/30], Step [500/1283], LR 3.2e-05, Loss: 131.0
Epoch [1/30], Step [600/1283], LR 3.5e-05, Loss: 155.3
Epoch [1/30], Step [700/1283], LR 3.8e-05, Loss: 132.4
Epoch [1/30], Step [800/1283], LR 4.1e-05, Loss: 146.5
Epoch [1/30], Step [900/1283], LR 4.4e-05, Loss: 129.8
Epoch [1/30], Step [1000/1283], LR 4.7e-05, Loss: 165.7
Epoch [1/30], Step [1100/1283], LR 5.0e-05, Loss: 112.2
Epoch [1/30], Step [1200/1283], LR 5.3e-05, Loss: 111.1
starting validation
mAP score regular 67.57, mAP score EMA 37.73
F1 score regular 67.34, Precision regular 80.34, Recall regular 57.97, F1 score EMA 46.81, Precision EMA 64.51, Recall EMA36.73
best model Saving successfully.
current_F1 = 67.34, highest_F1 = 67.34

Prior (train), first 5 classes: tensor([0.5192, 0.0794, 0.1779, 0.0870, 0.0773], device='cuda:0')
Prior (train), first 10 classes: ['person' 'car' 'chair' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [2/30], Step [000/1283], LR 5.6e-05, Loss: 153.0
Epoch [2/30], Step [100/1283], LR 5.9e-05, Loss: 127.8
Epoch [2/30], Step [200/1283], LR 6.3e-05, Loss: 137.1
Epoch [2/30], Step [300/1283], LR 6.7e-05, Loss: 105.0
Epoch [2/30], Step [400/1283], LR 7.0e-05, Loss: 136.1
Epoch [2/30], Step [500/1283], LR 7.4e-05, Loss: 107.4
Epoch [2/30], Step [600/1283], LR 7.8e-05, Loss: 141.4
Epoch [2/30], Step [700/1283], LR 8.1e-05, Loss: 133.0
Epoch [2/30], Step [800/1283], LR 8.5e-05, Loss: 130.4
Epoch [2/30], Step [900/1283], LR 8.9e-05, Loss: 110.7
Epoch [2/30], Step [1000/1283], LR 9.3e-05, Loss: 111.6
Epoch [2/30], Step [1100/1283], LR 9.7e-05, Loss: 93.7
Epoch [2/30], Step [1200/1283], LR 1.0e-04, Loss: 160.1
starting validation
mAP score regular 74.05, mAP score EMA 62.80
F1 score regular 71.00, Precision regular 82.44, Recall regular 62.35, F1 score EMA 62.40, Precision EMA 79.43, Recall EMA51.38
best model Saving successfully.
current_F1 = 71.00, highest_F1 = 71.00

Prior (train), first 5 classes: tensor([0.5058, 0.0664, 0.1599, 0.0693, 0.0603], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [3/30], Step [000/1283], LR 1.0e-04, Loss: 115.8
Epoch [3/30], Step [100/1283], LR 1.1e-04, Loss: 99.9
Epoch [3/30], Step [200/1283], LR 1.1e-04, Loss: 112.2
Epoch [3/30], Step [300/1283], LR 1.2e-04, Loss: 105.5
Epoch [3/30], Step [400/1283], LR 1.2e-04, Loss: 120.6
Epoch [3/30], Step [500/1283], LR 1.2e-04, Loss: 96.7
Epoch [3/30], Step [600/1283], LR 1.3e-04, Loss: 110.5
Epoch [3/30], Step [700/1283], LR 1.3e-04, Loss: 114.0
Epoch [3/30], Step [800/1283], LR 1.3e-04, Loss: 128.0
Epoch [3/30], Step [900/1283], LR 1.4e-04, Loss: 117.9
Epoch [3/30], Step [1000/1283], LR 1.4e-04, Loss: 136.1
Epoch [3/30], Step [1100/1283], LR 1.5e-04, Loss: 122.5
Epoch [3/30], Step [1200/1283], LR 1.5e-04, Loss: 114.5
starting validation
mAP score regular 76.04, mAP score EMA 71.79
F1 score regular 73.13, Precision regular 82.55, Recall regular 65.64, F1 score EMA 70.16, Precision EMA 81.80, Recall EMA61.41
best model Saving successfully.
current_F1 = 73.13, highest_F1 = 73.13

Prior (train), first 5 classes: tensor([0.4973, 0.0589, 0.1498, 0.0601, 0.0515], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [4/30], Step [000/1283], LR 1.5e-04, Loss: 82.2
Epoch [4/30], Step [100/1283], LR 1.6e-04, Loss: 115.3
Epoch [4/30], Step [200/1283], LR 1.6e-04, Loss: 134.6
Epoch [4/30], Step [300/1283], LR 1.6e-04, Loss: 98.0
Epoch [4/30], Step [400/1283], LR 1.6e-04, Loss: 105.5
Epoch [4/30], Step [500/1283], LR 1.7e-04, Loss: 111.1
Epoch [4/30], Step [600/1283], LR 1.7e-04, Loss: 114.0
Epoch [4/30], Step [700/1283], LR 1.7e-04, Loss: 104.7
Epoch [4/30], Step [800/1283], LR 1.8e-04, Loss: 99.7
Epoch [4/30], Step [900/1283], LR 1.8e-04, Loss: 133.8
Epoch [4/30], Step [1000/1283], LR 1.8e-04, Loss: 137.8
Epoch [4/30], Step [1100/1283], LR 1.8e-04, Loss: 111.1
Epoch [4/30], Step [1200/1283], LR 1.9e-04, Loss: 115.2
starting validation
mAP score regular 75.81, mAP score EMA 76.02
F1 score regular 71.70, Precision regular 83.94, Recall regular 62.58, F1 score EMA 73.98, Precision EMA 82.68, Recall EMA66.94
current_F1 = 71.70, highest_F1 = 73.13

Prior (train), first 5 classes: tensor([0.4913, 0.0543, 0.1433, 0.0544, 0.0461], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [5/30], Step [000/1283], LR 1.9e-04, Loss: 92.2
Epoch [5/30], Step [100/1283], LR 1.9e-04, Loss: 86.7
Epoch [5/30], Step [200/1283], LR 1.9e-04, Loss: 119.2
Epoch [5/30], Step [300/1283], LR 1.9e-04, Loss: 130.4
Epoch [5/30], Step [400/1283], LR 1.9e-04, Loss: 88.2
Epoch [5/30], Step [500/1283], LR 2.0e-04, Loss: 124.2
Epoch [5/30], Step [600/1283], LR 2.0e-04, Loss: 105.0
Epoch [5/30], Step [700/1283], LR 2.0e-04, Loss: 112.0
Epoch [5/30], Step [800/1283], LR 2.0e-04, Loss: 102.7
Epoch [5/30], Step [900/1283], LR 2.0e-04, Loss: 121.4
Epoch [5/30], Step [1000/1283], LR 2.0e-04, Loss: 100.9
Epoch [5/30], Step [1100/1283], LR 2.0e-04, Loss: 114.9
Epoch [5/30], Step [1200/1283], LR 2.0e-04, Loss: 79.8
starting validation
mAP score regular 75.51, mAP score EMA 78.26
F1 score regular 70.16, Precision regular 85.99, Recall regular 59.25, F1 score EMA 75.97, Precision EMA 83.14, Recall EMA69.94
current_F1 = 70.16, highest_F1 = 73.13

Prior (train), first 5 classes: tensor([0.4870, 0.0510, 0.1388, 0.0504, 0.0424], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [6/30], Step [000/1283], LR 2.0e-04, Loss: 75.9
Epoch [6/30], Step [100/1283], LR 2.0e-04, Loss: 113.6
Epoch [6/30], Step [200/1283], LR 2.0e-04, Loss: 96.2
Epoch [6/30], Step [300/1283], LR 2.0e-04, Loss: 120.7
Epoch [6/30], Step [400/1283], LR 2.0e-04, Loss: 83.7
Epoch [6/30], Step [500/1283], LR 2.0e-04, Loss: 107.1
Epoch [6/30], Step [600/1283], LR 2.0e-04, Loss: 98.5
Epoch [6/30], Step [700/1283], LR 2.0e-04, Loss: 108.7
Epoch [6/30], Step [800/1283], LR 2.0e-04, Loss: 120.7
Epoch [6/30], Step [900/1283], LR 2.0e-04, Loss: 103.7
Epoch [6/30], Step [1000/1283], LR 2.0e-04, Loss: 122.2
Epoch [6/30], Step [1100/1283], LR 2.0e-04, Loss: 122.7
Epoch [6/30], Step [1200/1283], LR 2.0e-04, Loss: 90.5
starting validation
mAP score regular 76.28, mAP score EMA 79.53
F1 score regular 72.21, Precision regular 85.97, Recall regular 62.24, F1 score EMA 77.24, Precision EMA 83.68, Recall EMA71.72
current_F1 = 72.21, highest_F1 = 73.13

Prior (train), first 5 classes: tensor([0.4836, 0.0485, 0.1350, 0.0475, 0.0398], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [7/30], Step [000/1283], LR 2.0e-04, Loss: 90.3
Epoch [7/30], Step [100/1283], LR 2.0e-04, Loss: 107.5
Epoch [7/30], Step [200/1283], LR 2.0e-04, Loss: 80.6
Epoch [7/30], Step [300/1283], LR 2.0e-04, Loss: 94.5
Epoch [7/30], Step [400/1283], LR 2.0e-04, Loss: 107.0
Epoch [7/30], Step [500/1283], LR 2.0e-04, Loss: 111.7
Epoch [7/30], Step [600/1283], LR 2.0e-04, Loss: 98.9
Epoch [7/30], Step [700/1283], LR 2.0e-04, Loss: 89.2
Epoch [7/30], Step [800/1283], LR 2.0e-04, Loss: 90.7
Epoch [7/30], Step [900/1283], LR 2.0e-04, Loss: 102.9
Epoch [7/30], Step [1000/1283], LR 2.0e-04, Loss: 89.0
Epoch [7/30], Step [1100/1283], LR 2.0e-04, Loss: 115.8
Epoch [7/30], Step [1200/1283], LR 2.0e-04, Loss: 96.7
starting validation
mAP score regular 76.32, mAP score EMA 80.20
F1 score regular 73.77, Precision regular 84.39, Recall regular 65.52, F1 score EMA 77.93, Precision EMA 83.93, Recall EMA72.73
best model Saving successfully.
current_F1 = 73.77, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4808, 0.0466, 0.1320, 0.0452, 0.0378], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [8/30], Step [000/1283], LR 2.0e-04, Loss: 94.8
Epoch [8/30], Step [100/1283], LR 2.0e-04, Loss: 92.7
Epoch [8/30], Step [200/1283], LR 2.0e-04, Loss: 81.0
Epoch [8/30], Step [300/1283], LR 2.0e-04, Loss: 94.5
Epoch [8/30], Step [400/1283], LR 2.0e-04, Loss: 109.0
Epoch [8/30], Step [500/1283], LR 2.0e-04, Loss: 86.9
Epoch [8/30], Step [600/1283], LR 1.9e-04, Loss: 97.8
Epoch [8/30], Step [700/1283], LR 1.9e-04, Loss: 100.3
Epoch [8/30], Step [800/1283], LR 1.9e-04, Loss: 104.1
Epoch [8/30], Step [900/1283], LR 1.9e-04, Loss: 125.3
Epoch [8/30], Step [1000/1283], LR 1.9e-04, Loss: 113.2
Epoch [8/30], Step [1100/1283], LR 1.9e-04, Loss: 108.8
Epoch [8/30], Step [1200/1283], LR 1.9e-04, Loss: 89.8
starting validation
mAP score regular 76.10, mAP score EMA 80.57
F1 score regular 71.33, Precision regular 83.59, Recall regular 62.21, F1 score EMA 78.27, Precision EMA 84.03, Recall EMA73.26
current_F1 = 71.33, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4781, 0.0448, 0.1294, 0.0434, 0.0361], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [9/30], Step [000/1283], LR 1.9e-04, Loss: 88.1
Epoch [9/30], Step [100/1283], LR 1.9e-04, Loss: 81.6
Epoch [9/30], Step [200/1283], LR 1.9e-04, Loss: 93.9
Epoch [9/30], Step [300/1283], LR 1.9e-04, Loss: 83.4
Epoch [9/30], Step [400/1283], LR 1.9e-04, Loss: 109.1
Epoch [9/30], Step [500/1283], LR 1.9e-04, Loss: 105.8
Epoch [9/30], Step [600/1283], LR 1.9e-04, Loss: 86.2
Epoch [9/30], Step [700/1283], LR 1.9e-04, Loss: 99.4
Epoch [9/30], Step [800/1283], LR 1.9e-04, Loss: 77.3
Epoch [9/30], Step [900/1283], LR 1.9e-04, Loss: 96.3
Epoch [9/30], Step [1000/1283], LR 1.9e-04, Loss: 85.1
Epoch [9/30], Step [1100/1283], LR 1.9e-04, Loss: 128.4
Epoch [9/30], Step [1200/1283], LR 1.9e-04, Loss: 89.7
starting validation
mAP score regular 75.78, mAP score EMA 80.67
F1 score regular 71.71, Precision regular 83.31, Recall regular 62.95, F1 score EMA 78.38, Precision EMA 84.10, Recall EMA73.39
current_F1 = 71.71, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4757, 0.0434, 0.1271, 0.0419, 0.0348], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [10/30], Step [000/1283], LR 1.9e-04, Loss: 82.3
Epoch [10/30], Step [100/1283], LR 1.9e-04, Loss: 83.6
Epoch [10/30], Step [200/1283], LR 1.9e-04, Loss: 86.6
Epoch [10/30], Step [300/1283], LR 1.9e-04, Loss: 71.4
Epoch [10/30], Step [400/1283], LR 1.8e-04, Loss: 82.3
Epoch [10/30], Step [500/1283], LR 1.8e-04, Loss: 100.2
Epoch [10/30], Step [600/1283], LR 1.8e-04, Loss: 101.6
Epoch [10/30], Step [700/1283], LR 1.8e-04, Loss: 84.6
Epoch [10/30], Step [800/1283], LR 1.8e-04, Loss: 101.8
Epoch [10/30], Step [900/1283], LR 1.8e-04, Loss: 95.0
Epoch [10/30], Step [1000/1283], LR 1.8e-04, Loss: 82.4
Epoch [10/30], Step [1100/1283], LR 1.8e-04, Loss: 93.8
Epoch [10/30], Step [1200/1283], LR 1.8e-04, Loss: 71.6
starting validation
mAP score regular 75.58, mAP score EMA 80.55
F1 score regular 69.46, Precision regular 88.54, Recall regular 57.14, F1 score EMA 78.35, Precision EMA 84.14, Recall EMA73.30
current_F1 = 69.46, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4735, 0.0422, 0.1250, 0.0405, 0.0337], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'bench']
Epoch [11/30], Step [000/1283], LR 1.8e-04, Loss: 79.4
Epoch [11/30], Step [100/1283], LR 1.8e-04, Loss: 87.8
Epoch [11/30], Step [200/1283], LR 1.8e-04, Loss: 88.2
Epoch [11/30], Step [300/1283], LR 1.8e-04, Loss: 103.9
Epoch [11/30], Step [400/1283], LR 1.8e-04, Loss: 85.8
Epoch [11/30], Step [500/1283], LR 1.8e-04, Loss: 94.0
Epoch [11/30], Step [600/1283], LR 1.8e-04, Loss: 86.2
Epoch [11/30], Step [700/1283], LR 1.7e-04, Loss: 82.6
Epoch [11/30], Step [800/1283], LR 1.7e-04, Loss: 92.0
Epoch [11/30], Step [900/1283], LR 1.7e-04, Loss: 60.7
Epoch [11/30], Step [1000/1283], LR 1.7e-04, Loss: 83.0
Epoch [11/30], Step [1100/1283], LR 1.7e-04, Loss: 74.5
Epoch [11/30], Step [1200/1283], LR 1.7e-04, Loss: 81.9
starting validation
mAP score regular 75.19, mAP score EMA 80.17
F1 score regular 71.85, Precision regular 83.53, Recall regular 63.04, F1 score EMA 78.15, Precision EMA 84.13, Recall EMA72.97
current_F1 = 71.85, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4713, 0.0411, 0.1231, 0.0394, 0.0327], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [12/30], Step [000/1283], LR 1.7e-04, Loss: 82.2
Epoch [12/30], Step [100/1283], LR 1.7e-04, Loss: 78.2
Epoch [12/30], Step [200/1283], LR 1.7e-04, Loss: 100.4
Epoch [12/30], Step [300/1283], LR 1.7e-04, Loss: 96.2
Epoch [12/30], Step [400/1283], LR 1.7e-04, Loss: 66.6
Epoch [12/30], Step [500/1283], LR 1.7e-04, Loss: 77.8
Epoch [12/30], Step [600/1283], LR 1.7e-04, Loss: 89.2
Epoch [12/30], Step [700/1283], LR 1.7e-04, Loss: 80.5
Epoch [12/30], Step [800/1283], LR 1.6e-04, Loss: 87.6
Epoch [12/30], Step [900/1283], LR 1.6e-04, Loss: 82.2
Epoch [12/30], Step [1000/1283], LR 1.6e-04, Loss: 69.9
Epoch [12/30], Step [1100/1283], LR 1.6e-04, Loss: 69.9
Epoch [12/30], Step [1200/1283], LR 1.6e-04, Loss: 94.0
starting validation
mAP score regular 73.21, mAP score EMA 79.75
F1 score regular 69.88, Precision regular 85.12, Recall regular 59.27, F1 score EMA 77.80, Precision EMA 84.12, Recall EMA72.36
current_F1 = 69.88, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4694, 0.0401, 0.1213, 0.0384, 0.0319], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [13/30], Step [000/1283], LR 1.6e-04, Loss: 93.2
Epoch [13/30], Step [100/1283], LR 1.6e-04, Loss: 70.3
Epoch [13/30], Step [200/1283], LR 1.6e-04, Loss: 54.7
Epoch [13/30], Step [300/1283], LR 1.6e-04, Loss: 79.8
Epoch [13/30], Step [400/1283], LR 1.6e-04, Loss: 73.6
Epoch [13/30], Step [500/1283], LR 1.6e-04, Loss: 71.3
Epoch [13/30], Step [600/1283], LR 1.6e-04, Loss: 58.8
Epoch [13/30], Step [700/1283], LR 1.6e-04, Loss: 75.8
Epoch [13/30], Step [800/1283], LR 1.5e-04, Loss: 85.3
Epoch [13/30], Step [900/1283], LR 1.5e-04, Loss: 88.8
Epoch [13/30], Step [1000/1283], LR 1.5e-04, Loss: 78.7
Epoch [13/30], Step [1100/1283], LR 1.5e-04, Loss: 69.2
Epoch [13/30], Step [1200/1283], LR 1.5e-04, Loss: 71.1
starting validation
mAP score regular 74.12, mAP score EMA 79.23
F1 score regular 70.62, Precision regular 83.38, Recall regular 61.24, F1 score EMA 77.30, Precision EMA 84.19, Recall EMA71.45
current_F1 = 70.62, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4675, 0.0391, 0.1196, 0.0374, 0.0311], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [14/30], Step [000/1283], LR 1.5e-04, Loss: 70.7
Epoch [14/30], Step [100/1283], LR 1.5e-04, Loss: 88.3
Epoch [14/30], Step [200/1283], LR 1.5e-04, Loss: 61.5
Epoch [14/30], Step [300/1283], LR 1.5e-04, Loss: 64.1
Epoch [14/30], Step [400/1283], LR 1.5e-04, Loss: 73.7
Epoch [14/30], Step [500/1283], LR 1.5e-04, Loss: 48.0
Epoch [14/30], Step [600/1283], LR 1.4e-04, Loss: 79.0
Epoch [14/30], Step [700/1283], LR 1.4e-04, Loss: 73.3
Epoch [14/30], Step [800/1283], LR 1.4e-04, Loss: 71.8
Epoch [14/30], Step [900/1283], LR 1.4e-04, Loss: 90.8
Epoch [14/30], Step [1000/1283], LR 1.4e-04, Loss: 90.0
Epoch [14/30], Step [1100/1283], LR 1.4e-04, Loss: 57.6
Epoch [14/30], Step [1200/1283], LR 1.4e-04, Loss: 49.9
starting validation
mAP score regular 73.11, mAP score EMA 78.65
F1 score regular 70.09, Precision regular 81.96, Recall regular 61.22, F1 score EMA 76.71, Precision EMA 84.31, Recall EMA70.37
current_F1 = 70.09, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4655, 0.0382, 0.1179, 0.0365, 0.0304], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [15/30], Step [000/1283], LR 1.4e-04, Loss: 76.3
Epoch [15/30], Step [100/1283], LR 1.4e-04, Loss: 79.0
Epoch [15/30], Step [200/1283], LR 1.4e-04, Loss: 55.7
Epoch [15/30], Step [300/1283], LR 1.4e-04, Loss: 55.3
Epoch [15/30], Step [400/1283], LR 1.3e-04, Loss: 83.4
Epoch [15/30], Step [500/1283], LR 1.3e-04, Loss: 65.9
Epoch [15/30], Step [600/1283], LR 1.3e-04, Loss: 54.1
Epoch [15/30], Step [700/1283], LR 1.3e-04, Loss: 78.0
Epoch [15/30], Step [800/1283], LR 1.3e-04, Loss: 68.2
Epoch [15/30], Step [900/1283], LR 1.3e-04, Loss: 56.6
Epoch [15/30], Step [1000/1283], LR 1.3e-04, Loss: 96.1
Epoch [15/30], Step [1100/1283], LR 1.3e-04, Loss: 76.4
Epoch [15/30], Step [1200/1283], LR 1.3e-04, Loss: 51.8
starting validation
mAP score regular 73.06, mAP score EMA 77.99
F1 score regular 69.54, Precision regular 83.38, Recall regular 59.64, F1 score EMA 75.94, Precision EMA 84.58, Recall EMA68.91
current_F1 = 69.54, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4635, 0.0374, 0.1163, 0.0357, 0.0298], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [16/30], Step [000/1283], LR 1.3e-04, Loss: 72.4
Epoch [16/30], Step [100/1283], LR 1.2e-04, Loss: 63.4
Epoch [16/30], Step [200/1283], LR 1.2e-04, Loss: 44.6
Epoch [16/30], Step [300/1283], LR 1.2e-04, Loss: 81.0
Epoch [16/30], Step [400/1283], LR 1.2e-04, Loss: 58.6
Epoch [16/30], Step [500/1283], LR 1.2e-04, Loss: 67.9
Epoch [16/30], Step [600/1283], LR 1.2e-04, Loss: 42.8
Epoch [16/30], Step [700/1283], LR 1.2e-04, Loss: 57.5
Epoch [16/30], Step [800/1283], LR 1.2e-04, Loss: 68.2
Epoch [16/30], Step [900/1283], LR 1.2e-04, Loss: 58.8
Epoch [16/30], Step [1000/1283], LR 1.2e-04, Loss: 57.1
Epoch [16/30], Step [1100/1283], LR 1.1e-04, Loss: 66.3
Epoch [16/30], Step [1200/1283], LR 1.1e-04, Loss: 52.6
starting validation
mAP score regular 72.48, mAP score EMA 77.35
F1 score regular 65.95, Precision regular 84.37, Recall regular 54.14, F1 score EMA 75.10, Precision EMA 84.83, Recall EMA67.37
current_F1 = 65.95, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4616, 0.0366, 0.1147, 0.0350, 0.0293], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [17/30], Step [000/1283], LR 1.1e-04, Loss: 59.6
Epoch [17/30], Step [100/1283], LR 1.1e-04, Loss: 64.2
Epoch [17/30], Step [200/1283], LR 1.1e-04, Loss: 47.0
Epoch [17/30], Step [300/1283], LR 1.1e-04, Loss: 47.0
Epoch [17/30], Step [400/1283], LR 1.1e-04, Loss: 66.4
Epoch [17/30], Step [500/1283], LR 1.1e-04, Loss: 56.6
Epoch [17/30], Step [600/1283], LR 1.1e-04, Loss: 66.4
Epoch [17/30], Step [700/1283], LR 1.1e-04, Loss: 55.1
Epoch [17/30], Step [800/1283], LR 1.0e-04, Loss: 64.2
Epoch [17/30], Step [900/1283], LR 1.0e-04, Loss: 84.8
Epoch [17/30], Step [1000/1283], LR 1.0e-04, Loss: 75.5
Epoch [17/30], Step [1100/1283], LR 1.0e-04, Loss: 66.3
Epoch [17/30], Step [1200/1283], LR 1.0e-04, Loss: 48.5
starting validation
mAP score regular 72.42, mAP score EMA 76.65
F1 score regular 66.13, Precision regular 85.86, Recall regular 53.77, F1 score EMA 74.08, Precision EMA 85.10, Recall EMA65.59
current_F1 = 66.13, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4596, 0.0359, 0.1131, 0.0343, 0.0288], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [18/30], Step [000/1283], LR 1.0e-04, Loss: 54.7
Epoch [18/30], Step [100/1283], LR 9.9e-05, Loss: 67.1
Epoch [18/30], Step [200/1283], LR 9.8e-05, Loss: 59.3
Epoch [18/30], Step [300/1283], LR 9.7e-05, Loss: 45.0
Epoch [18/30], Step [400/1283], LR 9.6e-05, Loss: 62.0
Epoch [18/30], Step [500/1283], LR 9.5e-05, Loss: 66.4
Epoch [18/30], Step [600/1283], LR 9.4e-05, Loss: 68.8
Epoch [18/30], Step [700/1283], LR 9.3e-05, Loss: 61.1
Epoch [18/30], Step [800/1283], LR 9.2e-05, Loss: 63.3
Epoch [18/30], Step [900/1283], LR 9.1e-05, Loss: 48.1
Epoch [18/30], Step [1000/1283], LR 9.0e-05, Loss: 65.9
Epoch [18/30], Step [1100/1283], LR 8.9e-05, Loss: 46.8
Epoch [18/30], Step [1200/1283], LR 8.8e-05, Loss: 43.8
starting validation
mAP score regular 71.18, mAP score EMA 75.93
F1 score regular 65.03, Precision regular 86.03, Recall regular 52.27, F1 score EMA 72.79, Precision EMA 85.53, Recall EMA63.36
current_F1 = 65.03, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4575, 0.0352, 0.1117, 0.0336, 0.0283], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [19/30], Step [000/1283], LR 8.7e-05, Loss: 65.0
Epoch [19/30], Step [100/1283], LR 8.6e-05, Loss: 29.3
Epoch [19/30], Step [200/1283], LR 8.5e-05, Loss: 57.5
Epoch [19/30], Step [300/1283], LR 8.4e-05, Loss: 52.5
Epoch [19/30], Step [400/1283], LR 8.3e-05, Loss: 64.2
Epoch [19/30], Step [500/1283], LR 8.2e-05, Loss: 40.7
Epoch [19/30], Step [600/1283], LR 8.1e-05, Loss: 48.5
Epoch [19/30], Step [700/1283], LR 8.0e-05, Loss: 45.0
Epoch [19/30], Step [800/1283], LR 7.9e-05, Loss: 53.5
Epoch [19/30], Step [900/1283], LR 7.8e-05, Loss: 46.8
Epoch [19/30], Step [1000/1283], LR 7.7e-05, Loss: 54.6
Epoch [19/30], Step [1100/1283], LR 7.6e-05, Loss: 55.5
Epoch [19/30], Step [1200/1283], LR 7.5e-05, Loss: 48.1
starting validation
mAP score regular 71.27, mAP score EMA 75.17
F1 score regular 65.26, Precision regular 85.54, Recall regular 52.75, F1 score EMA 71.36, Precision EMA 85.99, Recall EMA60.99
current_F1 = 65.26, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4554, 0.0345, 0.1102, 0.0330, 0.0278], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [20/30], Step [000/1283], LR 7.4e-05, Loss: 56.1
Epoch [20/30], Step [100/1283], LR 7.3e-05, Loss: 47.3
Epoch [20/30], Step [200/1283], LR 7.2e-05, Loss: 42.8
Epoch [20/30], Step [300/1283], LR 7.1e-05, Loss: 37.7
Epoch [20/30], Step [400/1283], LR 7.0e-05, Loss: 51.6
Epoch [20/30], Step [500/1283], LR 6.9e-05, Loss: 42.2
Epoch [20/30], Step [600/1283], LR 6.8e-05, Loss: 44.0
Epoch [20/30], Step [700/1283], LR 6.7e-05, Loss: 50.7
Epoch [20/30], Step [800/1283], LR 6.6e-05, Loss: 54.0
Epoch [20/30], Step [900/1283], LR 6.5e-05, Loss: 31.0
Epoch [20/30], Step [1000/1283], LR 6.4e-05, Loss: 45.5
Epoch [20/30], Step [1100/1283], LR 6.3e-05, Loss: 52.5
Epoch [20/30], Step [1200/1283], LR 6.2e-05, Loss: 48.9
starting validation
mAP score regular 71.46, mAP score EMA 74.48
F1 score regular 63.68, Precision regular 86.07, Recall regular 50.53, F1 score EMA 69.87, Precision EMA 86.29, Recall EMA58.70
current_F1 = 63.68, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4532, 0.0339, 0.1087, 0.0324, 0.0274], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [21/30], Step [000/1283], LR 6.2e-05, Loss: 36.2
Epoch [21/30], Step [100/1283], LR 6.1e-05, Loss: 50.1
Epoch [21/30], Step [200/1283], LR 6.0e-05, Loss: 51.0
Epoch [21/30], Step [300/1283], LR 5.9e-05, Loss: 25.8
Epoch [21/30], Step [400/1283], LR 5.8e-05, Loss: 51.6
Epoch [21/30], Step [500/1283], LR 5.7e-05, Loss: 37.9
Epoch [21/30], Step [600/1283], LR 5.6e-05, Loss: 43.4
Epoch [21/30], Step [700/1283], LR 5.5e-05, Loss: 57.9
Epoch [21/30], Step [800/1283], LR 5.4e-05, Loss: 39.0
Epoch [21/30], Step [900/1283], LR 5.3e-05, Loss: 58.5
Epoch [21/30], Step [1000/1283], LR 5.3e-05, Loss: 58.4
Epoch [21/30], Step [1100/1283], LR 5.2e-05, Loss: 43.6
Epoch [21/30], Step [1200/1283], LR 5.1e-05, Loss: 50.7
starting validation
mAP score regular 71.04, mAP score EMA 73.84
F1 score regular 60.50, Precision regular 88.23, Recall regular 46.04, F1 score EMA 68.23, Precision EMA 86.82, Recall EMA56.20
current_F1 = 60.50, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4509, 0.0333, 0.1073, 0.0318, 0.0270], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [22/30], Step [000/1283], LR 5.0e-05, Loss: 37.7
Epoch [22/30], Step [100/1283], LR 4.9e-05, Loss: 47.2
Epoch [22/30], Step [200/1283], LR 4.8e-05, Loss: 37.8
Epoch [22/30], Step [300/1283], LR 4.7e-05, Loss: 44.9
Epoch [22/30], Step [400/1283], LR 4.6e-05, Loss: 55.4
Epoch [22/30], Step [500/1283], LR 4.6e-05, Loss: 45.1
Epoch [22/30], Step [600/1283], LR 4.5e-05, Loss: 37.8
Epoch [22/30], Step [700/1283], LR 4.4e-05, Loss: 44.9
Epoch [22/30], Step [800/1283], LR 4.3e-05, Loss: 41.9
Epoch [22/30], Step [900/1283], LR 4.2e-05, Loss: 41.9
Epoch [22/30], Step [1000/1283], LR 4.1e-05, Loss: 40.8
Epoch [22/30], Step [1100/1283], LR 4.1e-05, Loss: 59.4
Epoch [22/30], Step [1200/1283], LR 4.0e-05, Loss: 37.9
starting validation
mAP score regular 70.70, mAP score EMA 73.20
F1 score regular 61.22, Precision regular 87.74, Recall regular 47.01, F1 score EMA 66.58, Precision EMA 87.31, Recall EMA53.80
current_F1 = 61.22, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4487, 0.0327, 0.1059, 0.0313, 0.0266], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [23/30], Step [000/1283], LR 3.9e-05, Loss: 39.5
Epoch [23/30], Step [100/1283], LR 3.8e-05, Loss: 56.9
Epoch [23/30], Step [200/1283], LR 3.8e-05, Loss: 37.4
Epoch [23/30], Step [300/1283], LR 3.7e-05, Loss: 51.2
Epoch [23/30], Step [400/1283], LR 3.6e-05, Loss: 42.5
Epoch [23/30], Step [500/1283], LR 3.5e-05, Loss: 42.2
Epoch [23/30], Step [600/1283], LR 3.4e-05, Loss: 55.4
Epoch [23/30], Step [700/1283], LR 3.4e-05, Loss: 37.6
Epoch [23/30], Step [800/1283], LR 3.3e-05, Loss: 40.8
Epoch [23/30], Step [900/1283], LR 3.2e-05, Loss: 33.4
Epoch [23/30], Step [1000/1283], LR 3.1e-05, Loss: 44.9
Epoch [23/30], Step [1100/1283], LR 3.1e-05, Loss: 31.8
Epoch [23/30], Step [1200/1283], LR 3.0e-05, Loss: 45.0
starting validation
mAP score regular 70.57, mAP score EMA 72.65
F1 score regular 58.76, Precision regular 89.10, Recall regular 43.84, F1 score EMA 65.00, Precision EMA 87.79, Recall EMA51.61
current_F1 = 58.76, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4464, 0.0321, 0.1045, 0.0308, 0.0263], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [24/30], Step [000/1283], LR 2.9e-05, Loss: 41.6
Epoch [24/30], Step [100/1283], LR 2.9e-05, Loss: 41.6
Epoch [24/30], Step [200/1283], LR 2.8e-05, Loss: 46.7
Epoch [24/30], Step [300/1283], LR 2.7e-05, Loss: 45.4
Epoch [24/30], Step [400/1283], LR 2.6e-05, Loss: 49.0
Epoch [24/30], Step [500/1283], LR 2.6e-05, Loss: 46.3
Epoch [24/30], Step [600/1283], LR 2.5e-05, Loss: 35.0
Epoch [24/30], Step [700/1283], LR 2.4e-05, Loss: 44.5
Epoch [24/30], Step [800/1283], LR 2.4e-05, Loss: 27.3
Epoch [24/30], Step [900/1283], LR 2.3e-05, Loss: 44.8
Epoch [24/30], Step [1000/1283], LR 2.2e-05, Loss: 48.9
Epoch [24/30], Step [1100/1283], LR 2.2e-05, Loss: 44.4
Epoch [24/30], Step [1200/1283], LR 2.1e-05, Loss: 37.1
starting validation
mAP score regular 70.45, mAP score EMA 72.13
F1 score regular 58.03, Precision regular 88.42, Recall regular 43.19, F1 score EMA 63.38, Precision EMA 88.06, Recall EMA49.50
current_F1 = 58.03, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4440, 0.0316, 0.1032, 0.0303, 0.0259], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [25/30], Step [000/1283], LR 2.1e-05, Loss: 46.1
Epoch [25/30], Step [100/1283], LR 2.0e-05, Loss: 43.9
Epoch [25/30], Step [200/1283], LR 1.9e-05, Loss: 49.2
Epoch [25/30], Step [300/1283], LR 1.9e-05, Loss: 38.3
Epoch [25/30], Step [400/1283], LR 1.8e-05, Loss: 46.8
Epoch [25/30], Step [500/1283], LR 1.8e-05, Loss: 50.5
Epoch [25/30], Step [600/1283], LR 1.7e-05, Loss: 38.6
Epoch [25/30], Step [700/1283], LR 1.7e-05, Loss: 35.1
Epoch [25/30], Step [800/1283], LR 1.6e-05, Loss: 47.0
Epoch [25/30], Step [900/1283], LR 1.5e-05, Loss: 32.8
Epoch [25/30], Step [1000/1283], LR 1.5e-05, Loss: 42.4
Epoch [25/30], Step [1100/1283], LR 1.4e-05, Loss: 38.6
Epoch [25/30], Step [1200/1283], LR 1.4e-05, Loss: 27.1
starting validation
mAP score regular 70.17, mAP score EMA 71.69
F1 score regular 56.39, Precision regular 88.63, Recall regular 41.34, F1 score EMA 61.88, Precision EMA 88.37, Recall EMA47.61
current_F1 = 56.39, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4417, 0.0311, 0.1020, 0.0298, 0.0256], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [26/30], Step [000/1283], LR 1.3e-05, Loss: 47.4
Epoch [26/30], Step [100/1283], LR 1.3e-05, Loss: 35.9
Epoch [26/30], Step [200/1283], LR 1.2e-05, Loss: 25.6
Epoch [26/30], Step [300/1283], LR 1.2e-05, Loss: 35.1
Epoch [26/30], Step [400/1283], LR 1.1e-05, Loss: 28.1
Epoch [26/30], Step [500/1283], LR 1.1e-05, Loss: 33.7
Epoch [26/30], Step [600/1283], LR 1.0e-05, Loss: 26.8
Epoch [26/30], Step [700/1283], LR 1.0e-05, Loss: 39.6
Epoch [26/30], Step [800/1283], LR 9.6e-06, Loss: 35.5
Epoch [26/30], Step [900/1283], LR 9.2e-06, Loss: 46.0
Epoch [26/30], Step [1000/1283], LR 8.7e-06, Loss: 37.6
Epoch [26/30], Step [1100/1283], LR 8.3e-06, Loss: 38.1
Epoch [26/30], Step [1200/1283], LR 7.9e-06, Loss: 26.7
starting validation
mAP score regular 70.33, mAP score EMA 71.32
F1 score regular 56.19, Precision regular 89.47, Recall regular 40.96, F1 score EMA 60.57, Precision EMA 88.58, Recall EMA46.01
current_F1 = 56.19, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4395, 0.0306, 0.1008, 0.0294, 0.0253], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [27/30], Step [000/1283], LR 7.6e-06, Loss: 27.0
Epoch [27/30], Step [100/1283], LR 7.2e-06, Loss: 52.6
Epoch [27/30], Step [200/1283], LR 6.8e-06, Loss: 39.6
Epoch [27/30], Step [300/1283], LR 6.5e-06, Loss: 39.5
Epoch [27/30], Step [400/1283], LR 6.1e-06, Loss: 36.6
Epoch [27/30], Step [500/1283], LR 5.8e-06, Loss: 41.2
Epoch [27/30], Step [600/1283], LR 5.4e-06, Loss: 41.5
Epoch [27/30], Step [700/1283], LR 5.1e-06, Loss: 31.6
Epoch [27/30], Step [800/1283], LR 4.8e-06, Loss: 41.8
Epoch [27/30], Step [900/1283], LR 4.5e-06, Loss: 35.1
Epoch [27/30], Step [1000/1283], LR 4.2e-06, Loss: 32.6
Epoch [27/30], Step [1100/1283], LR 3.9e-06, Loss: 26.4
Epoch [27/30], Step [1200/1283], LR 3.6e-06, Loss: 44.7
starting validation
mAP score regular 70.32, mAP score EMA 71.05
F1 score regular 56.16, Precision regular 89.43, Recall regular 40.93, F1 score EMA 59.37, Precision EMA 88.78, Recall EMA44.59
current_F1 = 56.16, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4373, 0.0302, 0.0997, 0.0290, 0.0251], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [28/30], Step [000/1283], LR 3.4e-06, Loss: 33.2
Epoch [28/30], Step [100/1283], LR 3.1e-06, Loss: 22.1
Epoch [28/30], Step [200/1283], LR 2.9e-06, Loss: 39.5
Epoch [28/30], Step [300/1283], LR 2.7e-06, Loss: 40.8
Epoch [28/30], Step [400/1283], LR 2.4e-06, Loss: 45.3
Epoch [28/30], Step [500/1283], LR 2.2e-06, Loss: 30.0
Epoch [28/30], Step [600/1283], LR 2.0e-06, Loss: 37.0
Epoch [28/30], Step [700/1283], LR 1.8e-06, Loss: 33.6
Epoch [28/30], Step [800/1283], LR 1.6e-06, Loss: 44.8
Epoch [28/30], Step [900/1283], LR 1.4e-06, Loss: 24.6
Epoch [28/30], Step [1000/1283], LR 1.3e-06, Loss: 36.6
Epoch [28/30], Step [1100/1283], LR 1.1e-06, Loss: 42.0
Epoch [28/30], Step [1200/1283], LR 9.7e-07, Loss: 43.0
starting validation
mAP score regular 70.17, mAP score EMA 70.84
F1 score regular 55.87, Precision regular 89.25, Recall regular 40.66, F1 score EMA 58.45, Precision EMA 88.93, Recall EMA43.53
current_F1 = 55.87, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4352, 0.0298, 0.0987, 0.0286, 0.0248], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [29/30], Step [000/1283], LR 8.5e-07, Loss: 37.2
Epoch [29/30], Step [100/1283], LR 7.3e-07, Loss: 35.3
Epoch [29/30], Step [200/1283], LR 6.1e-07, Loss: 24.4
Epoch [29/30], Step [300/1283], LR 5.0e-07, Loss: 17.3
Epoch [29/30], Step [400/1283], LR 4.0e-07, Loss: 34.6
Epoch [29/30], Step [500/1283], LR 3.2e-07, Loss: 36.2
Epoch [29/30], Step [600/1283], LR 2.4e-07, Loss: 26.3
Epoch [29/30], Step [700/1283], LR 1.8e-07, Loss: 45.6
Epoch [29/30], Step [800/1283], LR 1.2e-07, Loss: 41.0
Epoch [29/30], Step [900/1283], LR 7.6e-08, Loss: 31.6
Epoch [29/30], Step [1000/1283], LR 4.2e-08, Loss: 42.1
Epoch [29/30], Step [1100/1283], LR 1.8e-08, Loss: 55.1
Epoch [29/30], Step [1200/1283], LR 4.2e-09, Loss: 31.3
starting validation
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:  BCE_loss █▇▅▄▄▅▃▄▃▃▃▄▃▃▃▄▄▃▃▂▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▂▁▁
wandb:  F1 score ▁▆▇█▇▇██▇▇▇█▇▇▇▇▆▆▆▆▅▅▅▄▄▄▃▃▃▃
wandb: Precision ▁▂▃▃▄▆▆▅▄▄▇▄▅▄▃▄▅▆▆▅▆▇▇█▇▇████
wandb:    Recall ▁▆▇█▇▇▇█▇▇▆▇▇▇▇▇▆▅▅▅▅▄▄▃▃▃▃▃▃▃
wandb:       mAP ▁▆▇█████████▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
wandb: 
wandb: Run summary:
wandb:  BCE_loss 17.0379
wandb:  F1 score 56.20072
wandb: Precision 89.08952
wandb:    Recall 41.04741
wandb:       mAP 70.11117
wandb: 
wandb: 🚀 View run train__lrsd_rpneg_base_bce_negSampl0.6 at: https://wandb.ai/zhangshichuan/base_bce_negSampl/runs/z4clg5rt
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230928_004051-z4clg5rt/logs
mAP score regular 70.11, mAP score EMA 70.67
F1 score regular 56.20, Precision regular 89.09, Recall regular 41.05, F1 score EMA 57.79, Precision EMA 89.06, Recall EMA42.78
current_F1 = 56.20, highest_F1 = 73.77

Prior (train), first 5 classes: tensor([0.4334, 0.0294, 0.0977, 0.0283, 0.0246], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
wandb: Currently logged in as: zhangshichuan. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.15.11 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.15.0
wandb: Run data is saved locally in /scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL-zsc/wandb/run-20230928_043745-0uttvykq
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run train__lrsd_rpneg_base_bce_negSampl0.8
wandb: ⭐️ View project at https://wandb.ai/zhangshichuan/base_bce_negSampl
wandb: 🚀 View run at https://wandb.ai/zhangshichuan/base_bce_negSampl/runs/0uttvykq
creating model...
done

loading annotations into memory...
Done (t=4.08s)
creating index...
index created!
loading annotations into memory...
Done (t=8.76s)
creating index...
index created!
len(val_dataset)):  40137
len(train_dataset)):  82081
Original stat: [0.55035879 0.02786272 0.10484765 0.0297511  0.02732667 0.034003
 0.03001913 0.05264312 0.02556012 0.03524567]
Simulate coco. Mode: rps. Param: 0.800000
Simulated stat: [0.11142652 0.00557985 0.02160061 0.00626211 0.00517781]
Used parameters:
Image_size: 448
Learning_rate: 0.0002
Epochs: 30
Prior file was found in given path.
Prior file was loaded successfully. 
Epoch [0/30], Step [000/1283], LR 8.0e-06, Loss: 735.2
Epoch [0/30], Step [100/1283], LR 8.1e-06, Loss: 606.0
Epoch [0/30], Step [200/1283], LR 8.3e-06, Loss: 349.3
Epoch [0/30], Step [300/1283], LR 8.7e-06, Loss: 191.9
Epoch [0/30], Step [400/1283], LR 9.3e-06, Loss: 128.9
Epoch [0/30], Step [500/1283], LR 1.0e-05, Loss: 149.4
Epoch [0/30], Step [600/1283], LR 1.1e-05, Loss: 121.7
Epoch [0/30], Step [700/1283], LR 1.2e-05, Loss: 144.7
Epoch [0/30], Step [800/1283], LR 1.3e-05, Loss: 146.7
Epoch [0/30], Step [900/1283], LR 1.4e-05, Loss: 139.4
Epoch [0/30], Step [1000/1283], LR 1.6e-05, Loss: 124.3
Epoch [0/30], Step [1100/1283], LR 1.8e-05, Loss: 151.7
Epoch [0/30], Step [1200/1283], LR 1.9e-05, Loss: 134.4
starting validation
/scratch/jiazixia/.conda/envs/docre_clone/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:131: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
mAP score regular 27.98, mAP score EMA 4.98
F1 score regular 16.79, Precision regular 95.78, Recall regular 9.20, F1 score EMA 10.23, Precision EMA 7.78, Recall EMA14.96
best model Saving successfully.
current_F1 = 16.79, highest_F1 = 16.79

Prior (train), first 5 classes: tensor([0.3802, 0.1085, 0.1460, 0.1101, 0.1060], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'dining table' 'car' 'sports ball' 'cup' 'bottle' 'bowl'
 'truck' 'wine glass']
Epoch [1/30], Step [000/1283], LR 2.1e-05, Loss: 96.7
Epoch [1/30], Step [100/1283], LR 2.3e-05, Loss: 121.8
Epoch [1/30], Step [200/1283], LR 2.5e-05, Loss: 93.3
Epoch [1/30], Step [300/1283], LR 2.7e-05, Loss: 107.6
Epoch [1/30], Step [400/1283], LR 3.0e-05, Loss: 103.8
Epoch [1/30], Step [500/1283], LR 3.2e-05, Loss: 106.1
Epoch [1/30], Step [600/1283], LR 3.5e-05, Loss: 87.9
Epoch [1/30], Step [700/1283], LR 3.8e-05, Loss: 93.3
Epoch [1/30], Step [800/1283], LR 4.1e-05, Loss: 96.4
Epoch [1/30], Step [900/1283], LR 4.4e-05, Loss: 88.9
Epoch [1/30], Step [1000/1283], LR 4.7e-05, Loss: 85.5
Epoch [1/30], Step [1100/1283], LR 5.0e-05, Loss: 61.7
Epoch [1/30], Step [1200/1283], LR 5.3e-05, Loss: 77.9
starting validation
mAP score regular 60.78, mAP score EMA 22.41
F1 score regular 31.30, Precision regular 92.91, Recall regular 18.82, F1 score EMA 19.64, Precision EMA 73.71, Recall EMA11.33
best model Saving successfully.
current_F1 = 31.30, highest_F1 = 31.30

Prior (train), first 5 classes: tensor([0.3623, 0.0672, 0.1156, 0.0683, 0.0637], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'bowl' 'handbag'
 'truck' 'sports ball']
Epoch [2/30], Step [000/1283], LR 5.6e-05, Loss: 94.5
Epoch [2/30], Step [100/1283], LR 5.9e-05, Loss: 75.5
Epoch [2/30], Step [200/1283], LR 6.3e-05, Loss: 97.1
Epoch [2/30], Step [300/1283], LR 6.7e-05, Loss: 96.8
Epoch [2/30], Step [400/1283], LR 7.0e-05, Loss: 77.7
Epoch [2/30], Step [500/1283], LR 7.4e-05, Loss: 67.2
Epoch [2/30], Step [600/1283], LR 7.8e-05, Loss: 78.7
Epoch [2/30], Step [700/1283], LR 8.1e-05, Loss: 89.9
Epoch [2/30], Step [800/1283], LR 8.5e-05, Loss: 78.8
Epoch [2/30], Step [900/1283], LR 8.9e-05, Loss: 89.3
Epoch [2/30], Step [1000/1283], LR 9.3e-05, Loss: 110.0
Epoch [2/30], Step [1100/1283], LR 9.7e-05, Loss: 98.0
Epoch [2/30], Step [1200/1283], LR 1.0e-04, Loss: 84.5
starting validation
mAP score regular 69.01, mAP score EMA 53.60
F1 score regular 41.87, Precision regular 89.64, Recall regular 27.31, F1 score EMA 30.00, Precision EMA 92.43, Recall EMA17.91
best model Saving successfully.
current_F1 = 41.87, highest_F1 = 41.87

Prior (train), first 5 classes: tensor([0.3536, 0.0525, 0.1039, 0.0533, 0.0481], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'bowl' 'handbag'
 'truck' 'bench']
Epoch [3/30], Step [000/1283], LR 1.0e-04, Loss: 94.3
Epoch [3/30], Step [100/1283], LR 1.1e-04, Loss: 55.9
Epoch [3/30], Step [200/1283], LR 1.1e-04, Loss: 74.4
Epoch [3/30], Step [300/1283], LR 1.2e-04, Loss: 64.8
Epoch [3/30], Step [400/1283], LR 1.2e-04, Loss: 82.5
Epoch [3/30], Step [500/1283], LR 1.2e-04, Loss: 82.9
Epoch [3/30], Step [600/1283], LR 1.3e-04, Loss: 86.0
Epoch [3/30], Step [700/1283], LR 1.3e-04, Loss: 96.0
Epoch [3/30], Step [800/1283], LR 1.3e-04, Loss: 94.2
Epoch [3/30], Step [900/1283], LR 1.4e-04, Loss: 69.9
Epoch [3/30], Step [1000/1283], LR 1.4e-04, Loss: 68.8
Epoch [3/30], Step [1100/1283], LR 1.5e-04, Loss: 94.7
Epoch [3/30], Step [1200/1283], LR 1.5e-04, Loss: 68.3
starting validation
mAP score regular 71.49, mAP score EMA 66.59
F1 score regular 51.72, Precision regular 92.36, Recall regular 35.91, F1 score EMA 44.77, Precision EMA 94.84, Recall EMA29.30
best model Saving successfully.
current_F1 = 51.72, highest_F1 = 51.72

Prior (train), first 5 classes: tensor([0.3480, 0.0448, 0.0970, 0.0452, 0.0401], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'bowl' 'handbag'
 'truck' 'bench']
Epoch [4/30], Step [000/1283], LR 1.5e-04, Loss: 75.6
Epoch [4/30], Step [100/1283], LR 1.6e-04, Loss: 77.7
Epoch [4/30], Step [200/1283], LR 1.6e-04, Loss: 72.9
Epoch [4/30], Step [300/1283], LR 1.6e-04, Loss: 110.5
Epoch [4/30], Step [400/1283], LR 1.6e-04, Loss: 65.2
Epoch [4/30], Step [500/1283], LR 1.7e-04, Loss: 73.0
Epoch [4/30], Step [600/1283], LR 1.7e-04, Loss: 95.2
Epoch [4/30], Step [700/1283], LR 1.7e-04, Loss: 74.0
Epoch [4/30], Step [800/1283], LR 1.8e-04, Loss: 87.6
Epoch [4/30], Step [900/1283], LR 1.8e-04, Loss: 61.1
Epoch [4/30], Step [1000/1283], LR 1.8e-04, Loss: 88.4
Epoch [4/30], Step [1100/1283], LR 1.8e-04, Loss: 95.6
Epoch [4/30], Step [1200/1283], LR 1.9e-04, Loss: 102.7
starting validation
mAP score regular 71.65, mAP score EMA 72.20
F1 score regular 52.12, Precision regular 93.33, Recall regular 36.16, F1 score EMA 54.22, Precision EMA 95.38, Recall EMA37.88
best model Saving successfully.
current_F1 = 52.12, highest_F1 = 52.12

Prior (train), first 5 classes: tensor([0.3440, 0.0402, 0.0928, 0.0403, 0.0353], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'bench']
Epoch [5/30], Step [000/1283], LR 1.9e-04, Loss: 80.4
Epoch [5/30], Step [100/1283], LR 1.9e-04, Loss: 74.7
Epoch [5/30], Step [200/1283], LR 1.9e-04, Loss: 73.5
Epoch [5/30], Step [300/1283], LR 1.9e-04, Loss: 72.2
Epoch [5/30], Step [400/1283], LR 1.9e-04, Loss: 77.4
Epoch [5/30], Step [500/1283], LR 2.0e-04, Loss: 72.2
Epoch [5/30], Step [600/1283], LR 2.0e-04, Loss: 64.2
Epoch [5/30], Step [700/1283], LR 2.0e-04, Loss: 70.9
Epoch [5/30], Step [800/1283], LR 2.0e-04, Loss: 66.3
Epoch [5/30], Step [900/1283], LR 2.0e-04, Loss: 72.4
Epoch [5/30], Step [1000/1283], LR 2.0e-04, Loss: 83.7
Epoch [5/30], Step [1100/1283], LR 2.0e-04, Loss: 75.8
Epoch [5/30], Step [1200/1283], LR 2.0e-04, Loss: 79.4
starting validation
mAP score regular 71.90, mAP score EMA 74.85
F1 score regular 52.61, Precision regular 92.47, Recall regular 36.76, F1 score EMA 59.14, Precision EMA 95.31, Recall EMA42.87
best model Saving successfully.
current_F1 = 52.61, highest_F1 = 52.61

Prior (train), first 5 classes: tensor([0.3408, 0.0370, 0.0896, 0.0371, 0.0321], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [6/30], Step [000/1283], LR 2.0e-04, Loss: 85.3
Epoch [6/30], Step [100/1283], LR 2.0e-04, Loss: 68.0
Epoch [6/30], Step [200/1283], LR 2.0e-04, Loss: 57.0
Epoch [6/30], Step [300/1283], LR 2.0e-04, Loss: 77.2
Epoch [6/30], Step [400/1283], LR 2.0e-04, Loss: 68.7
Epoch [6/30], Step [500/1283], LR 2.0e-04, Loss: 82.2
Epoch [6/30], Step [600/1283], LR 2.0e-04, Loss: 70.5
Epoch [6/30], Step [700/1283], LR 2.0e-04, Loss: 64.0
Epoch [6/30], Step [800/1283], LR 2.0e-04, Loss: 56.7
Epoch [6/30], Step [900/1283], LR 2.0e-04, Loss: 95.8
Epoch [6/30], Step [1000/1283], LR 2.0e-04, Loss: 89.3
Epoch [6/30], Step [1100/1283], LR 2.0e-04, Loss: 77.5
Epoch [6/30], Step [1200/1283], LR 2.0e-04, Loss: 78.2
starting validation
mAP score regular 71.89, mAP score EMA 76.17
F1 score regular 54.87, Precision regular 89.72, Recall regular 39.51, F1 score EMA 62.00, Precision EMA 94.84, Recall EMA46.06
best model Saving successfully.
current_F1 = 54.87, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3379, 0.0347, 0.0870, 0.0346, 0.0297], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
Epoch [7/30], Step [000/1283], LR 2.0e-04, Loss: 74.6
Epoch [7/30], Step [100/1283], LR 2.0e-04, Loss: 76.6
Epoch [7/30], Step [200/1283], LR 2.0e-04, Loss: 54.4
Epoch [7/30], Step [300/1283], LR 2.0e-04, Loss: 64.9
Epoch [7/30], Step [400/1283], LR 2.0e-04, Loss: 66.1
Epoch [7/30], Step [500/1283], LR 2.0e-04, Loss: 66.3
Epoch [7/30], Step [600/1283], LR 2.0e-04, Loss: 66.8
Epoch [7/30], Step [700/1283], LR 2.0e-04, Loss: 73.9
Epoch [7/30], Step [800/1283], LR 2.0e-04, Loss: 67.5
Epoch [7/30], Step [900/1283], LR 2.0e-04, Loss: 58.1
Epoch [7/30], Step [1000/1283], LR 2.0e-04, Loss: 59.7
Epoch [7/30], Step [1100/1283], LR 2.0e-04, Loss: 59.5
Epoch [7/30], Step [1200/1283], LR 2.0e-04, Loss: 58.2
starting validation
mAP score regular 71.31, mAP score EMA 76.76
F1 score regular 52.92, Precision regular 91.90, Recall regular 37.16, F1 score EMA 63.37, Precision EMA 94.10, Recall EMA47.77
current_F1 = 52.92, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3355, 0.0328, 0.0849, 0.0327, 0.0279], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [8/30], Step [000/1283], LR 2.0e-04, Loss: 58.9
Epoch [8/30], Step [100/1283], LR 2.0e-04, Loss: 94.0
Epoch [8/30], Step [200/1283], LR 2.0e-04, Loss: 58.7
Epoch [8/30], Step [300/1283], LR 2.0e-04, Loss: 62.0
Epoch [8/30], Step [400/1283], LR 2.0e-04, Loss: 58.8
Epoch [8/30], Step [500/1283], LR 2.0e-04, Loss: 62.3
Epoch [8/30], Step [600/1283], LR 1.9e-04, Loss: 48.7
Epoch [8/30], Step [700/1283], LR 1.9e-04, Loss: 71.7
Epoch [8/30], Step [800/1283], LR 1.9e-04, Loss: 75.2
Epoch [8/30], Step [900/1283], LR 1.9e-04, Loss: 53.8
Epoch [8/30], Step [1000/1283], LR 1.9e-04, Loss: 69.2
Epoch [8/30], Step [1100/1283], LR 1.9e-04, Loss: 76.9
Epoch [8/30], Step [1200/1283], LR 1.9e-04, Loss: 44.9
starting validation
mAP score regular 70.33, mAP score EMA 76.83
F1 score regular 53.62, Precision regular 89.99, Recall regular 38.19, F1 score EMA 63.82, Precision EMA 93.57, Recall EMA48.43
current_F1 = 53.62, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3335, 0.0313, 0.0831, 0.0311, 0.0264], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [9/30], Step [000/1283], LR 1.9e-04, Loss: 64.4
Epoch [9/30], Step [100/1283], LR 1.9e-04, Loss: 56.8
Epoch [9/30], Step [200/1283], LR 1.9e-04, Loss: 56.7
Epoch [9/30], Step [300/1283], LR 1.9e-04, Loss: 53.2
Epoch [9/30], Step [400/1283], LR 1.9e-04, Loss: 75.5
Epoch [9/30], Step [500/1283], LR 1.9e-04, Loss: 51.9
Epoch [9/30], Step [600/1283], LR 1.9e-04, Loss: 63.8
Epoch [9/30], Step [700/1283], LR 1.9e-04, Loss: 64.2
Epoch [9/30], Step [800/1283], LR 1.9e-04, Loss: 65.0
Epoch [9/30], Step [900/1283], LR 1.9e-04, Loss: 79.6
Epoch [9/30], Step [1000/1283], LR 1.9e-04, Loss: 106.0
Epoch [9/30], Step [1100/1283], LR 1.9e-04, Loss: 68.3
Epoch [9/30], Step [1200/1283], LR 1.9e-04, Loss: 68.0
starting validation
mAP score regular 68.71, mAP score EMA 76.56
F1 score regular 50.21, Precision regular 88.49, Recall regular 35.05, F1 score EMA 63.63, Precision EMA 92.73, Recall EMA48.43
current_F1 = 50.21, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3315, 0.0300, 0.0815, 0.0298, 0.0251], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [10/30], Step [000/1283], LR 1.9e-04, Loss: 67.4
Epoch [10/30], Step [100/1283], LR 1.9e-04, Loss: 55.5
Epoch [10/30], Step [200/1283], LR 1.9e-04, Loss: 75.2
Epoch [10/30], Step [300/1283], LR 1.9e-04, Loss: 55.6
Epoch [10/30], Step [400/1283], LR 1.8e-04, Loss: 77.4
Epoch [10/30], Step [500/1283], LR 1.8e-04, Loss: 69.0
Epoch [10/30], Step [600/1283], LR 1.8e-04, Loss: 54.1
Epoch [10/30], Step [700/1283], LR 1.8e-04, Loss: 67.3
Epoch [10/30], Step [800/1283], LR 1.8e-04, Loss: 63.4
Epoch [10/30], Step [900/1283], LR 1.8e-04, Loss: 66.5
Epoch [10/30], Step [1000/1283], LR 1.8e-04, Loss: 60.0
Epoch [10/30], Step [1100/1283], LR 1.8e-04, Loss: 56.2
Epoch [10/30], Step [1200/1283], LR 1.8e-04, Loss: 53.1
starting validation
mAP score regular 68.71, mAP score EMA 76.06
F1 score regular 50.87, Precision regular 90.50, Recall regular 35.38, F1 score EMA 63.24, Precision EMA 91.96, Recall EMA48.19
current_F1 = 50.87, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3300, 0.0288, 0.0800, 0.0287, 0.0241], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [11/30], Step [000/1283], LR 1.8e-04, Loss: 54.4
Epoch [11/30], Step [100/1283], LR 1.8e-04, Loss: 62.6
Epoch [11/30], Step [200/1283], LR 1.8e-04, Loss: 65.3
Epoch [11/30], Step [300/1283], LR 1.8e-04, Loss: 60.8
Epoch [11/30], Step [400/1283], LR 1.8e-04, Loss: 57.9
Epoch [11/30], Step [500/1283], LR 1.8e-04, Loss: 43.1
Epoch [11/30], Step [600/1283], LR 1.8e-04, Loss: 49.3
Epoch [11/30], Step [700/1283], LR 1.7e-04, Loss: 57.1
Epoch [11/30], Step [800/1283], LR 1.7e-04, Loss: 56.7
Epoch [11/30], Step [900/1283], LR 1.7e-04, Loss: 55.8
Epoch [11/30], Step [1000/1283], LR 1.7e-04, Loss: 61.6
Epoch [11/30], Step [1100/1283], LR 1.7e-04, Loss: 42.4
Epoch [11/30], Step [1200/1283], LR 1.7e-04, Loss: 52.8
starting validation
mAP score regular 67.58, mAP score EMA 75.38
F1 score regular 46.65, Precision regular 91.67, Recall regular 31.29, F1 score EMA 62.22, Precision EMA 91.26, Recall EMA47.21
current_F1 = 46.65, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3283, 0.0278, 0.0786, 0.0277, 0.0232], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [12/30], Step [000/1283], LR 1.7e-04, Loss: 44.1
Epoch [12/30], Step [100/1283], LR 1.7e-04, Loss: 64.9
Epoch [12/30], Step [200/1283], LR 1.7e-04, Loss: 31.7
Epoch [12/30], Step [300/1283], LR 1.7e-04, Loss: 56.0
Epoch [12/30], Step [400/1283], LR 1.7e-04, Loss: 67.3
Epoch [12/30], Step [500/1283], LR 1.7e-04, Loss: 63.9
Epoch [12/30], Step [600/1283], LR 1.7e-04, Loss: 51.8
Epoch [12/30], Step [700/1283], LR 1.7e-04, Loss: 59.8
Epoch [12/30], Step [800/1283], LR 1.6e-04, Loss: 73.4
Epoch [12/30], Step [900/1283], LR 1.6e-04, Loss: 57.8
Epoch [12/30], Step [1000/1283], LR 1.6e-04, Loss: 61.6
Epoch [12/30], Step [1100/1283], LR 1.6e-04, Loss: 70.9
Epoch [12/30], Step [1200/1283], LR 1.6e-04, Loss: 56.0
starting validation
mAP score regular 66.44, mAP score EMA 74.50
F1 score regular 48.76, Precision regular 86.90, Recall regular 33.88, F1 score EMA 60.90, Precision EMA 90.51, Recall EMA45.88
current_F1 = 48.76, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3265, 0.0269, 0.0772, 0.0268, 0.0224], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [13/30], Step [000/1283], LR 1.6e-04, Loss: 67.0
Epoch [13/30], Step [100/1283], LR 1.6e-04, Loss: 63.2
Epoch [13/30], Step [200/1283], LR 1.6e-04, Loss: 47.7
Epoch [13/30], Step [300/1283], LR 1.6e-04, Loss: 53.3
Epoch [13/30], Step [400/1283], LR 1.6e-04, Loss: 54.8
Epoch [13/30], Step [500/1283], LR 1.6e-04, Loss: 43.3
Epoch [13/30], Step [600/1283], LR 1.6e-04, Loss: 46.2
Epoch [13/30], Step [700/1283], LR 1.6e-04, Loss: 49.1
Epoch [13/30], Step [800/1283], LR 1.5e-04, Loss: 48.0
Epoch [13/30], Step [900/1283], LR 1.5e-04, Loss: 43.7
Epoch [13/30], Step [1000/1283], LR 1.5e-04, Loss: 48.6
Epoch [13/30], Step [1100/1283], LR 1.5e-04, Loss: 56.2
Epoch [13/30], Step [1200/1283], LR 1.5e-04, Loss: 64.7
starting validation
mAP score regular 66.03, mAP score EMA 73.46
F1 score regular 49.16, Precision regular 85.57, Recall regular 34.48, F1 score EMA 59.25, Precision EMA 89.90, Recall EMA44.19
current_F1 = 49.16, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3246, 0.0260, 0.0759, 0.0260, 0.0217], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
wandb: Network error (ReadTimeout), entering retry loop.
Epoch [14/30], Step [000/1283], LR 1.5e-04, Loss: 31.0
Epoch [14/30], Step [100/1283], LR 1.5e-04, Loss: 37.5
Epoch [14/30], Step [200/1283], LR 1.5e-04, Loss: 34.9
Epoch [14/30], Step [300/1283], LR 1.5e-04, Loss: 35.9
Epoch [14/30], Step [400/1283], LR 1.5e-04, Loss: 39.0
Epoch [14/30], Step [500/1283], LR 1.5e-04, Loss: 50.8
Epoch [14/30], Step [600/1283], LR 1.4e-04, Loss: 56.6
Epoch [14/30], Step [700/1283], LR 1.4e-04, Loss: 49.9
Epoch [14/30], Step [800/1283], LR 1.4e-04, Loss: 40.5
Epoch [14/30], Step [900/1283], LR 1.4e-04, Loss: 44.3
Epoch [14/30], Step [1000/1283], LR 1.4e-04, Loss: 51.1
Epoch [14/30], Step [1100/1283], LR 1.4e-04, Loss: 39.7
Epoch [14/30], Step [1200/1283], LR 1.4e-04, Loss: 46.5
starting validation
mAP score regular 64.91, mAP score EMA 72.38
F1 score regular 45.39, Precision regular 88.22, Recall regular 30.56, F1 score EMA 57.15, Precision EMA 89.53, Recall EMA41.97
current_F1 = 45.39, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3224, 0.0252, 0.0746, 0.0252, 0.0210], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [15/30], Step [000/1283], LR 1.4e-04, Loss: 34.3
Epoch [15/30], Step [100/1283], LR 1.4e-04, Loss: 43.3
Epoch [15/30], Step [200/1283], LR 1.4e-04, Loss: 36.6
Epoch [15/30], Step [300/1283], LR 1.4e-04, Loss: 34.2
Epoch [15/30], Step [400/1283], LR 1.3e-04, Loss: 42.3
Epoch [15/30], Step [500/1283], LR 1.3e-04, Loss: 41.4
Epoch [15/30], Step [600/1283], LR 1.3e-04, Loss: 43.5
Epoch [15/30], Step [700/1283], LR 1.3e-04, Loss: 42.4
Epoch [15/30], Step [800/1283], LR 1.3e-04, Loss: 44.3
Epoch [15/30], Step [900/1283], LR 1.3e-04, Loss: 62.6
Epoch [15/30], Step [1000/1283], LR 1.3e-04, Loss: 33.7
Epoch [15/30], Step [1100/1283], LR 1.3e-04, Loss: 44.5
Epoch [15/30], Step [1200/1283], LR 1.3e-04, Loss: 40.0
starting validation
mAP score regular 63.99, mAP score EMA 71.17
F1 score regular 42.78, Precision regular 89.12, Recall regular 28.15, F1 score EMA 54.79, Precision EMA 89.19, Recall EMA39.54
current_F1 = 42.78, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3200, 0.0244, 0.0731, 0.0245, 0.0204], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [16/30], Step [000/1283], LR 1.3e-04, Loss: 37.3
Epoch [16/30], Step [100/1283], LR 1.2e-04, Loss: 39.7
Epoch [16/30], Step [200/1283], LR 1.2e-04, Loss: 33.1
Epoch [16/30], Step [300/1283], LR 1.2e-04, Loss: 32.4
Epoch [16/30], Step [400/1283], LR 1.2e-04, Loss: 33.6
Epoch [16/30], Step [500/1283], LR 1.2e-04, Loss: 27.8
Epoch [16/30], Step [600/1283], LR 1.2e-04, Loss: 41.2
Epoch [16/30], Step [700/1283], LR 1.2e-04, Loss: 35.3
Epoch [16/30], Step [800/1283], LR 1.2e-04, Loss: 34.8
Epoch [16/30], Step [900/1283], LR 1.2e-04, Loss: 54.5
Epoch [16/30], Step [1000/1283], LR 1.2e-04, Loss: 46.8
Epoch [16/30], Step [1100/1283], LR 1.1e-04, Loss: 27.6
Epoch [16/30], Step [1200/1283], LR 1.1e-04, Loss: 42.1
starting validation
mAP score regular 62.53, mAP score EMA 69.97
F1 score regular 39.87, Precision regular 89.02, Recall regular 25.69, F1 score EMA 52.07, Precision EMA 89.05, Recall EMA36.79
current_F1 = 39.87, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3174, 0.0237, 0.0718, 0.0238, 0.0199], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [17/30], Step [000/1283], LR 1.1e-04, Loss: 37.2
Epoch [17/30], Step [100/1283], LR 1.1e-04, Loss: 43.1
Epoch [17/30], Step [200/1283], LR 1.1e-04, Loss: 34.8
Epoch [17/30], Step [300/1283], LR 1.1e-04, Loss: 44.7
Epoch [17/30], Step [400/1283], LR 1.1e-04, Loss: 44.9
Epoch [17/30], Step [500/1283], LR 1.1e-04, Loss: 23.7
Epoch [17/30], Step [600/1283], LR 1.1e-04, Loss: 56.1
Epoch [17/30], Step [700/1283], LR 1.1e-04, Loss: 31.1
Epoch [17/30], Step [800/1283], LR 1.0e-04, Loss: 28.4
Epoch [17/30], Step [900/1283], LR 1.0e-04, Loss: 29.7
Epoch [17/30], Step [1000/1283], LR 1.0e-04, Loss: 34.7
Epoch [17/30], Step [1100/1283], LR 1.0e-04, Loss: 36.2
Epoch [17/30], Step [1200/1283], LR 1.0e-04, Loss: 41.6
starting validation
mAP score regular 61.97, mAP score EMA 68.72
F1 score regular 37.41, Precision regular 88.07, Recall regular 23.75, F1 score EMA 49.04, Precision EMA 89.08, Recall EMA33.84
current_F1 = 37.41, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3146, 0.0230, 0.0704, 0.0232, 0.0194], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [18/30], Step [000/1283], LR 1.0e-04, Loss: 31.1
Epoch [18/30], Step [100/1283], LR 9.9e-05, Loss: 39.6
Epoch [18/30], Step [200/1283], LR 9.8e-05, Loss: 43.6
Epoch [18/30], Step [300/1283], LR 9.7e-05, Loss: 32.8
Epoch [18/30], Step [400/1283], LR 9.6e-05, Loss: 30.6
Epoch [18/30], Step [500/1283], LR 9.5e-05, Loss: 29.9
Epoch [18/30], Step [600/1283], LR 9.4e-05, Loss: 31.2
Epoch [18/30], Step [700/1283], LR 9.3e-05, Loss: 20.4
Epoch [18/30], Step [800/1283], LR 9.2e-05, Loss: 32.8
Epoch [18/30], Step [900/1283], LR 9.1e-05, Loss: 41.9
Epoch [18/30], Step [1000/1283], LR 9.0e-05, Loss: 29.8
Epoch [18/30], Step [1100/1283], LR 8.9e-05, Loss: 33.7
Epoch [18/30], Step [1200/1283], LR 8.8e-05, Loss: 25.5
starting validation
mAP score regular 61.77, mAP score EMA 67.52
F1 score regular 35.59, Precision regular 87.26, Recall regular 22.36, F1 score EMA 46.11, Precision EMA 89.01, Recall EMA31.11
current_F1 = 35.59, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3113, 0.0224, 0.0690, 0.0226, 0.0189], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [19/30], Step [000/1283], LR 8.7e-05, Loss: 25.7
Epoch [19/30], Step [100/1283], LR 8.6e-05, Loss: 37.7
Epoch [19/30], Step [200/1283], LR 8.5e-05, Loss: 29.9
Epoch [19/30], Step [300/1283], LR 8.4e-05, Loss: 36.8
Epoch [19/30], Step [400/1283], LR 8.3e-05, Loss: 35.5
Epoch [19/30], Step [500/1283], LR 8.2e-05, Loss: 19.9
Epoch [19/30], Step [600/1283], LR 8.1e-05, Loss: 43.2
Epoch [19/30], Step [700/1283], LR 8.0e-05, Loss: 25.1
Epoch [19/30], Step [800/1283], LR 7.9e-05, Loss: 40.3
Epoch [19/30], Step [900/1283], LR 7.8e-05, Loss: 27.4
Epoch [19/30], Step [1000/1283], LR 7.7e-05, Loss: 28.9
Epoch [19/30], Step [1100/1283], LR 7.6e-05, Loss: 23.7
Epoch [19/30], Step [1200/1283], LR 7.5e-05, Loss: 35.7
starting validation
mAP score regular 61.84, mAP score EMA 66.42
F1 score regular 34.07, Precision regular 88.15, Recall regular 21.11, F1 score EMA 42.95, Precision EMA 89.11, Recall EMA28.29
current_F1 = 34.07, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3078, 0.0219, 0.0677, 0.0221, 0.0185], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [20/30], Step [000/1283], LR 7.4e-05, Loss: 28.5
Epoch [20/30], Step [100/1283], LR 7.3e-05, Loss: 21.8
Epoch [20/30], Step [200/1283], LR 7.2e-05, Loss: 27.0
Epoch [20/30], Step [300/1283], LR 7.1e-05, Loss: 29.2
Epoch [20/30], Step [400/1283], LR 7.0e-05, Loss: 32.6
Epoch [20/30], Step [500/1283], LR 6.9e-05, Loss: 23.9
Epoch [20/30], Step [600/1283], LR 6.8e-05, Loss: 26.3
Epoch [20/30], Step [700/1283], LR 6.7e-05, Loss: 34.9
Epoch [20/30], Step [800/1283], LR 6.6e-05, Loss: 27.3
Epoch [20/30], Step [900/1283], LR 6.5e-05, Loss: 25.2
Epoch [20/30], Step [1000/1283], LR 6.4e-05, Loss: 25.7
Epoch [20/30], Step [1100/1283], LR 6.3e-05, Loss: 21.8
Epoch [20/30], Step [1200/1283], LR 6.2e-05, Loss: 17.4
starting validation
mAP score regular 60.09, mAP score EMA 65.38
F1 score regular 25.27, Precision regular 89.02, Recall regular 14.73, F1 score EMA 39.61, Precision EMA 89.28, Recall EMA25.45
current_F1 = 25.27, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3042, 0.0213, 0.0664, 0.0215, 0.0180], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [21/30], Step [000/1283], LR 6.2e-05, Loss: 16.2
Epoch [21/30], Step [100/1283], LR 6.1e-05, Loss: 17.3
Epoch [21/30], Step [200/1283], LR 6.0e-05, Loss: 19.6
Epoch [21/30], Step [300/1283], LR 5.9e-05, Loss: 32.5
Epoch [21/30], Step [400/1283], LR 5.8e-05, Loss: 17.7
Epoch [21/30], Step [500/1283], LR 5.7e-05, Loss: 14.9
Epoch [21/30], Step [600/1283], LR 5.6e-05, Loss: 18.1
Epoch [21/30], Step [700/1283], LR 5.5e-05, Loss: 25.6
Epoch [21/30], Step [800/1283], LR 5.4e-05, Loss: 20.8
Epoch [21/30], Step [900/1283], LR 5.3e-05, Loss: 13.2
Epoch [21/30], Step [1000/1283], LR 5.3e-05, Loss: 28.9
Epoch [21/30], Step [1100/1283], LR 5.2e-05, Loss: 22.9
Epoch [21/30], Step [1200/1283], LR 5.1e-05, Loss: 25.1
starting validation
mAP score regular 59.93, mAP score EMA 64.29
F1 score regular 26.22, Precision regular 89.60, Recall regular 15.36, F1 score EMA 36.49, Precision EMA 89.28, Recall EMA22.93
current_F1 = 26.22, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.3004, 0.0208, 0.0651, 0.0210, 0.0176], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [22/30], Step [000/1283], LR 5.0e-05, Loss: 13.9
Epoch [22/30], Step [100/1283], LR 4.9e-05, Loss: 28.8
Epoch [22/30], Step [200/1283], LR 4.8e-05, Loss: 24.6
Epoch [22/30], Step [300/1283], LR 4.7e-05, Loss: 16.8
Epoch [22/30], Step [400/1283], LR 4.6e-05, Loss: 23.0
Epoch [22/30], Step [500/1283], LR 4.6e-05, Loss: 19.6
Epoch [22/30], Step [600/1283], LR 4.5e-05, Loss: 14.1
Epoch [22/30], Step [700/1283], LR 4.4e-05, Loss: 18.4
Epoch [22/30], Step [800/1283], LR 4.3e-05, Loss: 29.2
Epoch [22/30], Step [900/1283], LR 4.2e-05, Loss: 22.8
Epoch [22/30], Step [1000/1283], LR 4.1e-05, Loss: 15.1
Epoch [22/30], Step [1100/1283], LR 4.1e-05, Loss: 15.2
Epoch [22/30], Step [1200/1283], LR 4.0e-05, Loss: 17.1
starting validation
mAP score regular 59.64, mAP score EMA 63.28
F1 score regular 27.16, Precision regular 88.04, Recall regular 16.06, F1 score EMA 33.62, Precision EMA 89.31, Recall EMA20.71
current_F1 = 27.16, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.2964, 0.0203, 0.0639, 0.0205, 0.0173], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [23/30], Step [000/1283], LR 3.9e-05, Loss: 12.6
Epoch [23/30], Step [100/1283], LR 3.8e-05, Loss: 12.3
Epoch [23/30], Step [200/1283], LR 3.8e-05, Loss: 14.0
Epoch [23/30], Step [300/1283], LR 3.7e-05, Loss: 16.6
Epoch [23/30], Step [400/1283], LR 3.6e-05, Loss: 35.2
Epoch [23/30], Step [500/1283], LR 3.5e-05, Loss: 10.8
Epoch [23/30], Step [600/1283], LR 3.4e-05, Loss: 24.5
Epoch [23/30], Step [700/1283], LR 3.4e-05, Loss: 21.4
Epoch [23/30], Step [800/1283], LR 3.3e-05, Loss: 18.8
Epoch [23/30], Step [900/1283], LR 3.2e-05, Loss: 22.3
Epoch [23/30], Step [1000/1283], LR 3.1e-05, Loss: 37.6
Epoch [23/30], Step [1100/1283], LR 3.1e-05, Loss: 27.7
Epoch [23/30], Step [1200/1283], LR 3.0e-05, Loss: 25.9
starting validation
mAP score regular 59.57, mAP score EMA 62.43
F1 score regular 25.03, Precision regular 89.20, Recall regular 14.56, F1 score EMA 31.02, Precision EMA 89.58, Recall EMA18.76
current_F1 = 25.03, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.2925, 0.0198, 0.0627, 0.0201, 0.0169], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [24/30], Step [000/1283], LR 2.9e-05, Loss: 13.3
Epoch [24/30], Step [100/1283], LR 2.9e-05, Loss: 19.5
Epoch [24/30], Step [200/1283], LR 2.8e-05, Loss: 20.8
Epoch [24/30], Step [300/1283], LR 2.7e-05, Loss: 15.2
Epoch [24/30], Step [400/1283], LR 2.6e-05, Loss: 15.5
Epoch [24/30], Step [500/1283], LR 2.6e-05, Loss: 11.8
Epoch [24/30], Step [600/1283], LR 2.5e-05, Loss: 15.6
Epoch [24/30], Step [700/1283], LR 2.4e-05, Loss: 16.7
Epoch [24/30], Step [800/1283], LR 2.4e-05, Loss: 12.6
Epoch [24/30], Step [900/1283], LR 2.3e-05, Loss: 11.9
Epoch [24/30], Step [1000/1283], LR 2.2e-05, Loss: 27.7
Epoch [24/30], Step [1100/1283], LR 2.2e-05, Loss: 19.1
Epoch [24/30], Step [1200/1283], LR 2.1e-05, Loss: 18.6
starting validation
mAP score regular 58.94, mAP score EMA 61.67
F1 score regular 22.21, Precision regular 89.45, Recall regular 12.68, F1 score EMA 28.69, Precision EMA 89.66, Recall EMA17.08
current_F1 = 22.21, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.2887, 0.0194, 0.0616, 0.0197, 0.0166], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [25/30], Step [000/1283], LR 2.1e-05, Loss: 16.3
Epoch [25/30], Step [100/1283], LR 2.0e-05, Loss: 17.9
Epoch [25/30], Step [200/1283], LR 1.9e-05, Loss: 11.9
Epoch [25/30], Step [300/1283], LR 1.9e-05, Loss: 17.5
Epoch [25/30], Step [400/1283], LR 1.8e-05, Loss: 16.2
Epoch [25/30], Step [500/1283], LR 1.8e-05, Loss: 13.7
Epoch [25/30], Step [600/1283], LR 1.7e-05, Loss: 12.1
Epoch [25/30], Step [700/1283], LR 1.7e-05, Loss: 19.4
Epoch [25/30], Step [800/1283], LR 1.6e-05, Loss: 15.3
Epoch [25/30], Step [900/1283], LR 1.5e-05, Loss: 12.4
Epoch [25/30], Step [1000/1283], LR 1.5e-05, Loss: 29.8
Epoch [25/30], Step [1100/1283], LR 1.4e-05, Loss: 16.1
Epoch [25/30], Step [1200/1283], LR 1.4e-05, Loss: 20.2
starting validation
mAP score regular 59.12, mAP score EMA 61.03
F1 score regular 22.91, Precision regular 88.92, Recall regular 13.15, F1 score EMA 26.79, Precision EMA 89.84, Recall EMA15.74
current_F1 = 22.91, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.2849, 0.0190, 0.0605, 0.0193, 0.0162], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [26/30], Step [000/1283], LR 1.3e-05, Loss: 15.7
Epoch [26/30], Step [100/1283], LR 1.3e-05, Loss: 20.0
Epoch [26/30], Step [200/1283], LR 1.2e-05, Loss: 13.6
Epoch [26/30], Step [300/1283], LR 1.2e-05, Loss: 23.1
Epoch [26/30], Step [400/1283], LR 1.1e-05, Loss: 15.1
Epoch [26/30], Step [500/1283], LR 1.1e-05, Loss: 22.9
Epoch [26/30], Step [600/1283], LR 1.0e-05, Loss: 19.4
Epoch [26/30], Step [700/1283], LR 1.0e-05, Loss: 11.3
Epoch [26/30], Step [800/1283], LR 9.6e-06, Loss: 17.3
Epoch [26/30], Step [900/1283], LR 9.2e-06, Loss: 20.3
Epoch [26/30], Step [1000/1283], LR 8.7e-06, Loss: 19.8
Epoch [26/30], Step [1100/1283], LR 8.3e-06, Loss: 15.6
Epoch [26/30], Step [1200/1283], LR 7.9e-06, Loss: 18.2
starting validation
mAP score regular 59.00, mAP score EMA 60.52
F1 score regular 21.85, Precision regular 89.70, Recall regular 12.44, F1 score EMA 25.28, Precision EMA 89.90, Recall EMA14.71
current_F1 = 21.85, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.2812, 0.0186, 0.0595, 0.0189, 0.0159], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [27/30], Step [000/1283], LR 7.6e-06, Loss: 23.2
Epoch [27/30], Step [100/1283], LR 7.2e-06, Loss: 27.6
Epoch [27/30], Step [200/1283], LR 6.8e-06, Loss: 25.0
Epoch [27/30], Step [300/1283], LR 6.5e-06, Loss: 24.3
Epoch [27/30], Step [400/1283], LR 6.1e-06, Loss: 14.8
Epoch [27/30], Step [500/1283], LR 5.8e-06, Loss: 9.8
Epoch [27/30], Step [600/1283], LR 5.4e-06, Loss: 24.1
Epoch [27/30], Step [700/1283], LR 5.1e-06, Loss: 18.8
Epoch [27/30], Step [800/1283], LR 4.8e-06, Loss: 14.3
Epoch [27/30], Step [900/1283], LR 4.5e-06, Loss: 16.2
Epoch [27/30], Step [1000/1283], LR 4.2e-06, Loss: 13.5
Epoch [27/30], Step [1100/1283], LR 3.9e-06, Loss: 8.1
Epoch [27/30], Step [1200/1283], LR 3.6e-06, Loss: 16.5
starting validation
mAP score regular 58.75, mAP score EMA 60.13
F1 score regular 21.07, Precision regular 89.82, Recall regular 11.94, F1 score EMA 23.99, Precision EMA 89.84, Recall EMA13.84
current_F1 = 21.07, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.2777, 0.0182, 0.0585, 0.0186, 0.0157], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'car' 'dining table' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [28/30], Step [000/1283], LR 3.4e-06, Loss: 27.6
Epoch [28/30], Step [100/1283], LR 3.1e-06, Loss: 22.1
Epoch [28/30], Step [200/1283], LR 2.9e-06, Loss: 12.5
Epoch [28/30], Step [300/1283], LR 2.7e-06, Loss: 20.5
Epoch [28/30], Step [400/1283], LR 2.4e-06, Loss: 9.5
Epoch [28/30], Step [500/1283], LR 2.2e-06, Loss: 18.7
Epoch [28/30], Step [600/1283], LR 2.0e-06, Loss: 17.6
Epoch [28/30], Step [700/1283], LR 1.8e-06, Loss: 11.7
Epoch [28/30], Step [800/1283], LR 1.6e-06, Loss: 21.1
Epoch [28/30], Step [900/1283], LR 1.4e-06, Loss: 11.9
Epoch [28/30], Step [1000/1283], LR 1.3e-06, Loss: 22.4
Epoch [28/30], Step [1100/1283], LR 1.1e-06, Loss: 7.8
Epoch [28/30], Step [1200/1283], LR 9.7e-07, Loss: 14.3
starting validation
mAP score regular 58.86, mAP score EMA 59.76
F1 score regular 21.31, Precision regular 90.27, Recall regular 12.08, F1 score EMA 23.07, Precision EMA 89.79, Recall EMA13.23
current_F1 = 21.31, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.2744, 0.0179, 0.0576, 0.0182, 0.0154], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'dining table' 'car' 'cup' 'bottle' 'handbag' 'bowl'
 'backpack' 'truck']
Epoch [29/30], Step [000/1283], LR 8.5e-07, Loss: 21.1
Epoch [29/30], Step [100/1283], LR 7.3e-07, Loss: 23.0
Epoch [29/30], Step [200/1283], LR 6.1e-07, Loss: 16.1
Epoch [29/30], Step [300/1283], LR 5.0e-07, Loss: 15.2
Epoch [29/30], Step [400/1283], LR 4.0e-07, Loss: 6.9
Epoch [29/30], Step [500/1283], LR 3.2e-07, Loss: 27.6
Epoch [29/30], Step [600/1283], LR 2.4e-07, Loss: 16.5
Epoch [29/30], Step [700/1283], LR 1.8e-07, Loss: 25.1
Epoch [29/30], Step [800/1283], LR 1.2e-07, Loss: 13.9
Epoch [29/30], Step [900/1283], LR 7.6e-08, Loss: 10.1
Epoch [29/30], Step [1000/1283], LR 4.2e-08, Loss: 10.4
Epoch [29/30], Step [1100/1283], LR 1.8e-08, Loss: 20.2
Epoch [29/30], Step [1200/1283], LR 4.2e-09, Loss: 17.4
starting validation
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:  BCE_loss █▇▅▄▅▅▅▄▄▄▅▄▄▃▃▃▃▃▃▂▃▂▂▂▃▂▂▂▂▁▁▁▁▂▁▁▂▂▁▁
wandb:  F1 score ▁▄▆▇▇████▇▇▆▇▇▆▆▅▅▄▄▃▃▃▃▂▂▂▂▂▂
wandb: Precision █▆▄▆▆▆▄▅▄▃▄▅▂▁▃▃▃▃▂▃▃▄▃▃▄▃▄▄▄▄
wandb:    Recall ▁▃▅▇▇▇█▇█▇▇▆▇▇▆▅▅▄▄▄▂▂▃▂▂▂▂▂▂▂
wandb:       mAP ▁▆███████▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆
wandb: 
wandb: Run summary:
wandb:  BCE_loss 5.74204
wandb:  F1 score 21.70159
wandb: Precision 89.7972
wandb:    Recall 12.34219
wandb:       mAP 59.03669
wandb: 
wandb: 🚀 View run train__lrsd_rpneg_base_bce_negSampl0.8 at: https://wandb.ai/zhangshichuan/base_bce_negSampl/runs/0uttvykq
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230928_043745-0uttvykq/logs
mAP score regular 59.04, mAP score EMA 59.50
F1 score regular 21.70, Precision regular 89.80, Recall regular 12.34, F1 score EMA 22.34, Precision EMA 89.86, Recall EMA12.75
current_F1 = 21.70, highest_F1 = 54.87

Prior (train), first 5 classes: tensor([0.2713, 0.0176, 0.0568, 0.0179, 0.0152], device='cuda:0')
Prior (train), first 10 classes: ['person' 'chair' 'dining table' 'car' 'cup' 'bottle' 'handbag' 'bowl'
 'truck' 'backpack']
