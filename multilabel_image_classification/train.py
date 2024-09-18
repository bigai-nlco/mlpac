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
from src.models.utils.factory import WarmupCosineSchedule

from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from src.helper_functions.coco_simulation import simulate_coco
from src.helper_functions.get_data import get_data
import wandb


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL/data/coco')
parser.add_argument('--metadata', type=str, default='./data/COCO_2014')
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--stop_epoch', default=None, type=int)
parser.add_argument('--weight_decay', default=1e-5, type=float)
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
parser.add_argument('--path_dest', type=str, default="./outputs")
parser.add_argument('--debug_mode', type=str, default="on_farm")
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--wandb_proj', type=str, default="base_bec")
parser.add_argument('--wandb_id', type=str, default="zhangshichuan")
parser.add_argument('--results_dir', type=str, default="0.5")
parser.add_argument('--best_epoch', type=int, default=10)
parser.add_argument('--tunning_mode', type=str, default="_lrsd_rpneg_")
parser.add_argument('--m', type=float, default=0.999)
parser.add_argument('--pct_start', type=float, default=0.1)
parser.add_argument('--running_mode', type=str, default="not_both")
parser.add_argument('--ablation_mode', type=str, default="no_ablation")

def main():

    # ---------------------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()
    args.do_bottleneck_head = False
    if not os.path.exists(args.path_dest):
        os.makedirs(args.path_dest)

    # ---------------------------------------------------------------------------------
    #args.prior_threshold = (1 - args.simulate_partial_param)/2

    # Setup model
    print('creating model...')
    model = create_model(args).cuda()
    if args.stage == 2:
        model_v = create_model(args).cuda()
        model_v.load_state_dict(torch.load('/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL-zsc/outputs/neg/rps{}_time1/model-{}-1283.ckpt'.format(str(args.simulate_partial_param)[-1], args.best_epoch)))
        model.load_state_dict(torch.load('/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL-zsc/outputs/neg/rps{}_time1/model-{}-1283.ckpt'.format(str(args.simulate_partial_param)[-1], args.best_epoch)))
    elif args.stage == 1:
        model_t = create_model(args).cuda()
        model_v = create_model(args).cuda()
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['state_dict'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
    elif args.stage == 3:
        model_t = create_model(args).cuda()
        model_v = create_model(args).cuda()
        #state = torch.load(args.model_path, map_location='cpu')
        #filtered_dict = {k: v for k, v in state['state_dict'].items() if
        #                 (k in model.state_dict() and 'head.fc' not in k)}
        #model.load_state_dict(filtered_dict, strict=False)
        model_v.load_state_dict(torch.load('/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL-zsc/outputs/neg/Sel_onlyPos_rps{}_time1/model-highest.ckpt'.format(str(args.simulate_partial_param)[-1])))
        model.load_state_dict(torch.load('/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL-zsc/outputs/neg/Sel_onlyPos_rps{}_time1/model-highest.ckpt'.format(str(args.simulate_partial_param)[-1])))
        model_t.load_state_dict(torch.load('/scratch/nlp/jiazixia/rl_enhanced/PartialLabelingCSL-zsc/outputs/neg/Sel_onlyPos_rps{}_time1/model-highest.ckpt'.format(str(args.simulate_partial_param)[-1])))
    print('done\n')

    # ---------------------------------------------------------------------------------
    train_loader, val_loader = get_data(args)

    args.results_dir = args.wandb_proj + str(args.simulate_partial_param)
    if not os.path.exists(args.wandb_id):
        os.mkdir(args.wandb_id)
    if not os.path.exists(f'{args.wandb_id}/{args.results_dir}'):
        os.mkdir(f'{args.wandb_id}/{args.results_dir}')

    # Actuall Training
    train_multi_label_coco(model, model_v, model_t, train_loader, val_loader, args)


def train_multi_label_coco(model, model_v, model_t, train_loader, val_loader, args):

    print("Used parameters:")
    print("Image_size:", args.image_size)
    print("Learning_rate:", args.lr)
    print("Epochs:", args.epochs)

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    prior = ComputePrior(train_loader.dataset.classes)

    # set optimizer
    Epochs = args.epochs
    if args.stop_epoch is not None:
        Stop_epoch = args.stop_epoch
    else:
        Stop_epoch = args.epochs
    weight_decay = args.weight_decay
    lr = args.lr

    criterion = PartialSelectiveLoss(args)
    # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=False)

    parameters = add_weight_decay(model, weight_decay)
    parameters_v = add_weight_decay(model_v, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    optimizer_v = torch.optim.Adam(params=parameters_v, lr=lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=args.pct_start)
    scheduler_v = lr_scheduler.OneCycleLR(optimizer_v, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=0.1)
    #scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1283, t_total=12830)

    highest_mAP, highest_p, highest_r = 0, 0, 0
    trainInfoList = []
    scaler = GradScaler()
    wandb.init(project=args.wandb_proj, entity=args.wandb_id, name='train_'+args.tunning_mode+str(args.results_dir), group=f'train_coco_{args.wandb_proj}')
    wandb.config.update(args)
    if os.path.exists(f'{args.wandb_id}/{args.results_dir}/test_logs.txt'):
        os.remove(f'{args.wandb_id}/{args.results_dir}/test_logs.txt')
    test_logs = open(f'{args.wandb_id}/{args.results_dir}/test_logs.txt', 'a')
    
    num_steps = 0
    r_step, r_w = 0, 12 #5 #7 #15 #10 #12
    r_list = [11.5, 11, 10.5, 10] #[4.5, 4, 3.5] #[6, 5.5, 5, 4.5] #[14,13.5,13,12.5] #[9, 8.5, 8, 7.5] #[11.5,11,10.5,10]
    thresh_v, thresh_p = 0.25, 0.5
    for epoch in range(Epochs):

        if epoch > Stop_epoch:
            break
        #breakpoint()
        for i, (inputData, target, filename) in enumerate(train_loader):
            num_steps += 1
            inputData = inputData.cuda()
            # target = target.max(dim=1)[0]
            target = target.cuda()

            if args.stage < 3:
                model.train()
                if args.stage == 2:
                    with autocast():  # mixed precision
                        output = model(inputData).float()
                    with torch.no_grad():
                        output_v = model_v(inputData).float()
                    loss_RL,recall_reward,prob_reward,R_p_pos,R_p_neg,log_comb_prob = RL_loss(output_v, output, target)
                    loss = loss_RL
                    wandb.log({'loss_RL': loss_RL, 'recall_reward': recall_reward, 'prob_reward': prob_reward,'R_p_pos':R_p_pos,'R_p_neg':R_p_neg,'log_comb_prob':log_comb_prob}, step=num_steps)
                elif args.stage == 1:
                    with autocast():  # mixed precision
                        output = model(inputData).float()
                    loss = criterion(output, target)
                    wandb.log({'BCE_loss': loss}, step=num_steps)

                model.zero_grad()
                scaler.scale(loss).backward()
                # loss.backward()
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                scheduler.step()
                ema.update(model)
                prior.update(output)
            else:
                # model_v training
                if epoch < 5:
                    model_v.train()
                    with autocast():  # mixed precision
                        output = model_v(inputData).float()
                    with torch.no_grad():
                        output_t = model_t(inputData).float()
                        output_t = torch.sigmoid(output_t)
                        target_sd_p = torch.zeros_like(target)
                        target_sd_v = torch.zeros_like(target)
                        target_new = torch.zeros_like(target)
                        target_new[target>0.5]=1
                        target_sd_p[output_t>0.8]=1
                        target_sd_v[torch.sigmoid(output)>0.5]=1
                        target_sd = target_sd_v * target_sd_p
                        target_new = target_new + target_sd
                        new_added = target_sd.sum()-(target_new==2).float().sum()
                        target_new[target_new>0]=1
                    if args.ablation_mode == 'label_enhance':
                        loss_v = criterion(output, target)
                    else:
                        loss_v = criterion(output, target_new)#torch.nn.BCEWithLogitsLoss()(output, target_new)
                    wandb.log({'BCE_loss_model_v_1': loss_v, 'pos_added':new_added}, step=num_steps)
                    model_v.zero_grad()
                    scaler.scale(loss_v).backward()
                    torch.nn.utils.clip_grad_norm_(parameters=model_v.parameters(), max_norm=10, norm_type=2)
                    scaler.step(optimizer_v)
                    scaler.update()
                scheduler_v.step()

                # model_p training based on model_v
                model.train()
                with autocast():  # mixed precision
                    output = model(inputData).float()
                with torch.no_grad():
                    output_v = model_v(inputData).float()
                loss_RL, recall_reward, prob_reward, R_p_pos, R_p_neg, log_comb_prob = RL_loss(args, output_v, output, target, r_w)
                if args.ablation_mode == 'self_training':
                    with torch.no_grad():
                        output_v = model_v(inputData).float()
                        output_v = torch.sigmoid(output_v)
                        target_sd_p = torch.zeros_like(target)
                        target_sd_v = torch.zeros_like(target)
                        target_new = torch.zeros_like(target)
                        target_new[target>0]=1
                        target_sd_p[output_v>0.5]=1
                        target_sd_v[torch.sigmoid(output)>0.5]=1
                        target_sd = target_sd_v * target_sd_p
                        target_new = target_new + target_sd
                        new_added = target_sd.sum()-(target_new==2).float().sum()
                        target_new[target_new>0]=1
                    loss = criterion(output, target_new) 
                else:
                    loss = loss_RL
                wandb.log({'loss_RL': loss_RL, 'recall_reward': recall_reward, 'prob_reward': prob_reward, 'R_p_pos': R_p_pos,
                     'R_p_neg': R_p_neg, 'log_comb_prob': log_comb_prob}, step=num_steps)
                model.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                ema.update(model)
                prior.update(output)

                # model_v update based on model_p
                #if epoch == 0:
                #    with torch.no_grad():
                #        for param_q, param_k in zip(model.parameters(), model_v.parameters()):
                #            param_k.data = param_k.data * args.m + param_q.data * (1. - args.m)

            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

            if args.stage == 3: 
                if (i%500 == 0 and i > 0) or i == len(train_loader)-1:
                    model.eval()
                    F1, P, R, mAP = validate_multi(val_loader, model, ema)
                    # wandb.log({'F1 score': F1, 'Precision': P, 'Recall': R, 'mAP': mAP}, step=epoch)
                    wandb.log({'F1 score': F1, 'Precision': P, 'Recall': R, 'mAP': mAP}, step=num_steps)
                    model_v.eval()
                    F1_v, P_v, R_v, mAP_v = validate_multi(val_loader, model_v, ema)
                    # wandb.log({'F1 score': F1, 'Precision': P, 'Recall': R, 'mAP': mAP}, step=epoch)
                    wandb.log({'F1 score v': F1_v, 'Precision v': P_v, 'Recall v': R_v, 'mAP v': mAP_v}, step=num_steps)
                    r_w = r_list[r_step%len(r_list)]
                    r_step = r_step + 1
                    wandb.log({'rec_weight':r_w}, step = num_steps)
                    # model.train()
                    if F1 > highest_mAP:
                        highest_mAP = F1
                        model_t.load_state_dict(model.state_dict())
                        try:
                            torch.save(model.state_dict(), os.path.join(args.path_dest, 'model-highest.ckpt'))
                            print("best model Saving successfully.")
                        except:
                            print("best model Saving failed.")
                    test_logs.write(f'Epoch [{epoch}/{Epochs}]: F1 score: {F1:.4f} Precision: {P:.4f}, Recall: {R:.4f}, mAP: {mAP:.4f}, current_best_F1: {highest_mAP:.4f} \n')
                    print('current_F1 = {:.2f}, highest_F1 = {:.2f}\n'.format(F1, highest_mAP))
        if args.stage == 1:
            model.eval()
            F1, P, R, mAP = validate_multi(val_loader, model, ema)
            wandb.log({'F1 score': F1, 'Precision': P, 'Recall': R, 'mAP': mAP}, step=num_steps)
            if F1 > highest_mAP:
                highest_mAP = F1
                highest_p, highest_r =  P, R
                try:
                    torch.save(model.state_dict(), os.path.join(args.path_dest, 'model-highest.ckpt'))
                    print("best model Saving successfully.")
                except:
                    print("best model Saving failed.")
            test_logs.write(f'Epoch [{epoch}/{Epochs}]: F1 score: {F1:.4f} Precision: {P:.4f}, Recall: {R:.4f}, mAP: {mAP:.4f}, current_best_F1: {highest_mAP:.4f} \n')
            print('current_F1 = {:.2f}, highest_F1 = {:.2f}\n'.format(F1, highest_mAP))

        #model_t.load_state_dict(model.state_dict()) 
        
        # Report prior
        prior.save_prior()
        prior.get_top_freq_classes()
        
        # Save ckpt
        #try:
        #    torch.save(model.state_dict(), os.path.join(args.path_dest, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        #    print("Model saved successfully.")
        #except:
        #    print("Saving model failed.")
    
    wandb.finish()
    with open("results_{}.txt".format(args.wandb_proj), "a") as f:
        f.write('mask_ratio-{}: F1-{}, P-{}, R-{}'.format(args.simulate_partial_param, highest_mAP, highest_p, highest_r))
        f.write('\r\n')

def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
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
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))

    F1_score_regular, P_regular, R_regular = asses(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    F1_score_ema, P_ema, R_ema = asses(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("F1 score regular {:.2f}, Precision regular {:.2f}, Recall regular {:.2f}, F1 score EMA {:.2f}, Precision EMA {:.2f}, Recall EMA{:.2f}".format(F1_score_regular, P_regular, R_regular, F1_score_ema, P_ema, R_ema))
    return F1_score_regular, P_regular, R_regular, mAP_score_regular


if __name__ == '__main__':
    main()
