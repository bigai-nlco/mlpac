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



parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL/data/coco')
parser.add_argument('--metadata', type=str, default='./data/COCO_2014')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--stop_epoch', default=None, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--model-path', default='/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL-zsc/outputs/neg/rps0.5_RL/model-2-1283.ckpt', type=str)
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


def main():

    # ---------------------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()
    args.do_bottleneck_head = False

    # ---------------------------------------------------------------------------------

    # Setup model
    print('creating model...')
    model = create_model(args).cuda()
    model.load_state_dict(torch.load(args.model_path))

    print('done\n')

    # ---------------------------------------------------------------------------------
    train_loader, val_loader = get_data(args)

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    model.eval()
    validate_multi(args, val_loader, model, ema)

def validate_multi(args, val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    p_num = 0
    u_num = 0
    dic_filename2targets = {}
    for i, (input, target, filename) in enumerate(val_loader):
        target = target
        # target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())
        if filename[1] not in dic_filename2targets.keys():
            dic_filename2targets[filename[1]]=[output_regular.cpu().numpy()[1], target[1].numpy()]
    import pickle
    f_p = open('base_RL.pickle','wb')
    pickle.dump(dic_filename2targets, f_p)
    #pickle.close()
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    F1, P, R = asses(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    F1_ema, P_ema, R_ema = asses(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print('F1 regular: {}, Precision regular: {}, Recall regular: {}, F1 ema: {}, P ema: {}, R ema: {}'.format(F1, P, R, F1_ema, P_ema, R_ema))

if __name__ == '__main__':
    main()
