import torch
from torch import nn as nn, Tensor
import os
import pandas as pd
import numpy as np
from torch.distributions import Categorical

def RL_loss(args, v_prob, p_prob, ground_truth, r_w, thresh=0.5, sampling_num = 100):
    #with torch.no_grad():
    if args.ablation_mode == 'action_sampling':
        r_w = 20
        sampling_num = 1
    v_prob = torch.clamp(torch.sigmoid(v_prob),1e-7,1-(1e-7))
    p_prob = torch.clamp(torch.sigmoid(p_prob),1e-7,1-(1e-7))

    #v_prob = torch.sigmoid(v_prob)
    #p_prob = torch.sigmoid(p_prob)
    #breakpoint()
    ground_truth[ground_truth == -1] = 0
    v_prob = v_prob.detach()
    loss_thresh, loss_R_recall_thresh, loss_R_p_thresh, R_p_pos_thresh, R_p_neg_thresh, log_comb_prob_thresh = RL_loss_sampling1(args, v_prob, p_prob, ground_truth, r_w, thresh)
    loss_sampling, loss_R_recall, loss_R_p = 0,0,0
    loss_R_p_pos, loss_R_p_neg, loss_log_comb_prob = 0,0,0
    for s in range(sampling_num):
        neg_p_prob = 1 - p_prob
        if torch.any(torch.isnan(p_prob)):
            continue
        p = torch.cat([neg_p_prob[:, :, None], p_prob[:, :, None]], dim=2)
        dist = Categorical(p)
        dist.sample()
        loss_s, R_recall, R_p, R_p_pos, R_p_neg, log_comb_prob = RL_loss_sampling1(args, v_prob, p_prob, ground_truth, r_w, sampler=dist)
        loss_sampling = loss_sampling + loss_s
        loss_R_recall = loss_R_recall + R_recall
        loss_R_p = loss_R_p + R_p
        loss_R_p_pos = loss_R_p_pos + R_p_pos
        loss_R_p_neg = loss_R_p_neg + R_p_neg
        loss_log_comb_prob = loss_log_comb_prob + log_comb_prob
    #if args.ablation_mode == 'action_sampling':
    #    return loss_thresh, loss_R_recall_thresh, loss_R_p_thresh, R_p_pos_thresh, R_p_neg_thresh, log_comb_prob_thresh
    #else:
    return 0.1*loss_thresh+0.9*loss_sampling/sampling_num, loss_R_recall/sampling_num, loss_R_p/sampling_num, loss_R_p_pos/sampling_num, loss_R_p_neg/sampling_num, loss_log_comb_prob/sampling_num
    #return loss_thresh, loss_R_recall_thresh, loss_R_p_thresh, R_p_pos_thresh, R_p_neg_thresh

def RL_loss_sampling1(args, v_prob, p_prob, ground_truth, r_w, thresh=None, sampler=None): # all are B x C
    #with torch.no_grad():
    if thresh is not None and sampler is None:
        action = (p_prob>thresh).float()
    elif thresh is None and sampler is not None:
        action = torch.from_numpy(sampler.sample().cpu().numpy()).cuda()
    else:
        raise ValueError('illegal assignment for thresh and sampler......')

    neg_p_prob = 1 - p_prob
    #with torch.no_grad():
    neg_action = 1 - action
    combined_prob = action * p_prob + neg_action * neg_p_prob
    log_comb_prob = torch.log(combined_prob).sum(dim=1) # (B, )
    #with torch.no_grad():
    correct_num = ((action+ground_truth)==2).float().sum(dim=1) #(B, )
    total_num = (ground_truth==1).float().sum(dim=1) #(B, )
    pred_num = (action==1).float().sum(dim=1)

    rec = correct_num/(total_num+1e-9)
    prec = correct_num/(pred_num+1e-9)
    f1_s = 2*(rec*prec)/(rec+prec+1e-9)

    gt_clone = ground_truth.clone()
    gt_clone[gt_clone==-1]=-10
    corect_num_neg = ((action+gt_clone)==0).float().sum(dim=1)
    total_num_neg = (gt_clone==0).float().sum(dim=1)
    rec_neg = corect_num_neg/(total_num_neg+1e-9)

    if args.ablation_mode == 'F1':
        R_recall = f1_s
        r_w = 20
    elif args.ablation_mode == 'Precision':
        R_recall = prec
        r_w = 20
    else:
        R_recall = rec+0.2*rec_neg #(B, )

    neg_v_prob = 1 - v_prob
    idx_random = torch.from_numpy(np.random.random((neg_action.shape)) < 0.8).cuda()
    neg_action_ = neg_action.clone()
    neg_action_[idx_random] = 0
    R_p_neg = torch.clamp(torch.log(neg_v_prob/v_prob)*neg_action_,-1,1).sum(dim=1)/(1e-9+neg_action_.sum(dim=1))#(B,)
    R_p_pos = torch.clamp(torch.log(v_prob/neg_v_prob)*action,-1,1).sum(dim=1)/(1e-9+action.sum(dim=1))#(B,)
    R_p = R_p_pos + R_p_neg #(B, )

    # w = R_recall.mean()/R_p.mean()
    if args.ablation_mode == 'r_reward':
        loss = -1*((log_comb_prob * R_p.detach()).mean())
    elif args.ablation_mode == 'p_reward':
        loss = -1*((log_comb_prob * (r_w*R_recall.detach())).mean())
    else:
        w = R_p / (R_recall+1e-9)
        loss = -1*((log_comb_prob * (5*w*R_recall.detach()+R_p.detach())).mean())

    #import pdb
    #pdb.set_trace()

    return loss, R_recall.mean(), R_p.mean(), R_p_pos.mean(), R_p_neg.mean(), -log_comb_prob.mean()


class PartialSelectiveLoss(nn.Module):

    def __init__(self, args):
        super(PartialSelectiveLoss, self).__init__()
        self.args = args
        self.clip = args.clip
        self.gamma_pos = args.gamma_pos
        self.gamma_neg = args.gamma_neg
        self.gamma_unann = args.gamma_unann
        self.alpha_pos = args.alpha_pos
        self.alpha_neg = args.alpha_neg
        self.alpha_unann = args.alpha_unann

        self.targets_weights = None

        if args.prior_path is not None:
            print("Prior file was found in given path.")
            df = pd.read_csv(args.prior_path)
            self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
            print("Prior file was loaded successfully. ")

    def forward(self, logits, targets):

        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (targets == -1).float()

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        prior_classes = None
        if hasattr(self, "prior_classes"):
            prior_classes = torch.tensor(list(self.prior_classes.values())).cuda()
        
        targets_weights = self.targets_weights
        targets_weights, xs_neg = edit_targets_parital_labels(self.args, targets, targets_weights, xs_neg,
                                                              prior_classes=prior_classes)

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unann = self.alpha_unann * targets_unann * torch.log(torch.clamp(xs_neg, min=1e-8))

        if self.args.running_mode == 'posWeight':
            w_pos = 0 if targets_pos.sum()==0 else targets.shape[-1]*targets.shape[-2]/targets_pos.sum()
            BCE_loss = w_pos*BCE_pos + BCE_neg + BCE_unann
        elif self.args.running_mode == 'negSampl':
            pos_num = targets_pos.sum()
            total_num = targets.shape[-1]*targets.shape[-2]
            mask_num = total_num - 10*pos_num
            mask_ratio = mask_num/(targets_unann.sum()+targets_neg.sum())
            if mask_ratio > 0:
                idx_random = torch.from_numpy(np.random.random((targets.shape)) < 0.8).cuda()
                targets_neg_unann = targets_neg + targets_unann
                targets_neg_unann[idx_random] = 0
                BCE_neg_unann = BCE_unann + BCE_neg
                BCE_neg_unann = BCE_neg_unann * targets_neg_unann
                BCE_loss = BCE_pos + BCE_neg_unann
            else:
                BCE_loss = BCE_pos + BCE_neg + BCE_unann
        else:
            BCE_loss = BCE_pos + BCE_neg + BCE_unann

        # Adding asymmetric gamma weights
        # with torch.no_grad():
        #     asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unann),
        #                              self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
        #                              self.gamma_unann * targets_unann)
        # BCE_loss *= asymmetric_w
        
        # partial labels weights
        # BCE_loss *= targets_weights

        return -BCE_loss.sum()


def edit_targets_parital_labels(args, targets, targets_weights, xs_neg, prior_classes=None):
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if args.partial_loss_mode is None:
        targets_weights = 1.0

    elif args.partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets_weights = 1.0

    elif args.partial_loss_mode == 'ignore':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        targets_weights[targets == -1] = 0

    elif args.partial_loss_mode == 'ignore_normalize_classes':
        # remove all unsure targets and normalize by Durand et al. https://arxiv.org/pdf/1902.09720.pdfs
        alpha_norm, beta_norm = 1, 1
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        n_annotated = 1 + torch.sum(targets != -1, axis=1)    # Add 1 to avoid dividing by zero

        g_norm = alpha_norm * (1 / n_annotated) + beta_norm
        n_classes = targets_weights.shape[1]
        targets_weights *= g_norm.repeat([n_classes, 1]).T
        targets_weights[targets == -1] = 0

    elif args.partial_loss_mode == 'selective':
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        else:
            targets_weights[:] = 1.0
        num_top_k = args.likelihood_topk * targets_weights.shape[0]

        xs_neg_prob = xs_neg
        if prior_classes is not None:
            if args.prior_threshold:
                idx_ignore = torch.where(prior_classes > args.prior_threshold)[0]
                targets_weights[:, idx_ignore] = 0
                #targets_weights += (targets == 1).float()
                targets_weights += (targets != -1).float()
                targets_weights = targets_weights.bool()

        negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)

        # set all unsure targets as negative
        # targets[targets == -1] = 0

    return targets_weights, xs_neg


# @torch.jit.script
def negative_backprop_fun_jit(targets: Tensor, xs_neg_prob: Tensor, targets_weights: Tensor, num_top_k: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        #cond_flatten = torch.where(targets_flatten < 1)[0]
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_prob_flatten = xs_neg_prob.flatten()
        ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_k]]] = 0


class ComputePrior:
    def __init__(self, classes):
        self.classes = classes
        n_classes = len(self.classes)
        self.sum_pred_train = torch.zeros(n_classes).cuda()
        self.sum_pred_val = torch.zeros(n_classes).cuda()
        self.cnt_samples_train,  self.cnt_samples_val = .0, .0
        self.avg_pred_train, self.avg_pred_val = None, None
        self.path_dest = "./outputs"
        self.path_local = "/class_prior/"

    def update(self, logits, training=True):
        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            if training:
                self.sum_pred_train += torch.sum(preds, axis=0)
                self.cnt_samples_train += preds.shape[0]
                self.avg_pred_train = self.sum_pred_train / self.cnt_samples_train

            else:
                self.sum_pred_val += torch.sum(preds, axis=0)
                self.cnt_samples_val += preds.shape[0]
                self.avg_pred_val = self.sum_pred_val / self.cnt_samples_val

    def save_prior(self):

        print('Prior (train), first 5 classes: {}'.format(self.avg_pred_train[:5]))

        # Save data frames as csv files
        if not os.path.exists(self.path_dest):
            os.makedirs(self.path_dest)

        df_train = pd.DataFrame({"Classes": list(self.classes.values()),
                                 "avg_pred": self.avg_pred_train.cpu()})
        df_train.to_csv(path_or_buf=os.path.join(self.path_dest, "train_avg_preds.csv"),
                        sep=',', header=True, index=False, encoding='utf-8')

        if self.avg_pred_val is not None:
            df_val = pd.DataFrame({"Classes": list(self.classes.values()),
                                   "avg_pred": self.avg_pred_val.cpu()})
            df_val.to_csv(path_or_buf=os.path.join(self.path_dest, "val_avg_preds.csv"),
                          sep=',', header=True, index=False, encoding='utf-8')

    def get_top_freq_classes(self):
        n_top = 10
        top_idx = torch.argsort(-self.avg_pred_train.cpu())[:n_top]
        top_classes = np.array(list(self.classes.values()))[top_idx]
        print('Prior (train), first {} classes: {}'.format(n_top, top_classes))
