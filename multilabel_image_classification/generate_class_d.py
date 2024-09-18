import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, asses
from src.models import create_model
# from src.loss_functions.losses import AsymmetricLoss
from src.loss_functions.partial_asymmetric_loss import PartialSelectiveLoss, ComputePrior, RL_loss

from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from src.helper_functions.coco_simulation import simulate_coco
from src.helper_functions.get_data import get_data
import torch.nn.functional as F
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL/data/coco')
parser.add_argument('--metadata', type=str, default='./data/COCO_2014')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--stop_epoch', default=None, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--model-path', default='/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL/models/pretrain/tresnet_m_miil_21k.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--simulate_partial_type', type=str, default=None, help="options are fpc or rps")
parser.add_argument('--simulate_partial_param', type=float, default=1000)
parser.add_argument('--partial_loss_mode', type=str, default="negative")
parser.add_argument('--clip', type=float, default=0)
parser.add_argument('--gamma_pos', type=float, default=0)
parser.add_argument('--gamma_neg', type=float, default=1)
parser.add_argument('--gamma_unann', type=float, default=2)
parser.add_argument('--alpha_pos', type=float, default=1)
parser.add_argument('--alpha_neg', type=float, default=1)
parser.add_argument('--alpha_unann', type=float, default=1)
parser.add_argument('--likelihood_topk', type=int, default=5)
parser.add_argument('--prior_path', type=str, default=None)
parser.add_argument('--prior_threshold', type=float, default=0.05)
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--debug_mode', type=str, default="on_farm")
parser.add_argument('--stage', type=int, default=1)

label2class = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def main():

    # ---------------------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()
    args.do_bottleneck_head = False

    # ---------------------------------------------------------------------------------

    # Setup model
    print('creating model...')
    model = create_model(args).cuda().eval()
    model.load_state_dict(torch.load(args.model_path))

    print('done\n')

    # ---------------------------------------------------------------------------------
    train_loader, val_loader = get_data(args)

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    model.eval()
    validate_multi(args, train_loader, model, ema)

def validate_multi(args, train_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []

    for i, (input, target) in enumerate(train_loader):
        target = target
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    save_class_d(args, torch.cat(targets), torch.cat(preds_regular))
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    F1, P, R = asses(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    F1_ema, P_ema, R_ema = asses(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print('F1 regular: {}, Precision regular: {}, Recall regular: {}, F1 ema: {}, P ema: {}, R ema: {}'.format(F1, P, R, F1_ema, P_ema, R_ema))

def save_class_d(args, targets, preds):
    preds = preds.to(dtype=torch.float32)
    tar = targets.clone()
    tar[targets<1]=0
    S = tar.sum(dim=0) #(80,)
    S = S/tar.shape[0]
    #breakpoint()
    preds_mask = torch.zeros_like(preds)
    preds_mask[targets<1] = 1
    S_with_pred = (preds * preds_mask + tar).sum(dim=0)
    S_with_pred = S_with_pred/tar.shape[0]

    names = label2class.values()
    pandas_S = pd.DataFrame(columns=['Classes','avg_pred'])
    pandas_S_with_pred = pd.DataFrame(columns=['Classes', 'avg_pred'])
    pandas_S_with_pred['Classes'] = names
    pandas_S['Classes'] = names
    pandas_S_with_pred['avg_pred'] = S_with_pred
    pandas_S['avg_pred'] = S

    pandas_S.to_csv(r"outputs/priors/prior_rps_{}.csv".format(args.simulate_partial_param), sep=',', index=False)
    pandas_S_with_pred.to_csv(r"outputs/priors/prior_rps_{}_P.csv".format(args.simulate_partial_param), sep=',', index=False)

if __name__ == '__main__':
    main()
