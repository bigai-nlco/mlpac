import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO


def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def ap_per_class(conf,target_cls):
    """ 通过召回率与精确度曲线计算mAP
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # 参数说明
        tp: True positives (list).
        conf: 置信度[0,1] (list).
        pred_cls: 预测的目标类别 (list).
        target_cls: 真正的目标类别 (list).
    # 返回
          [precision,recall,average precision,f1, classes_num]
    """

    # 按照预测的置信度做降序排列, 得到排序的索引
    pred_cls = (conf>0.5).astype(np.float)
    tp = ((pred_cls+target_cls)==2).astype(np.float)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 类别c的人工标注目标的数量
    n_gt = target_cls.sum()
    # 类别c的预测目标数量
    n_p = pred_cls.sum()

    if n_p == 0 or n_gt == 0:
        ap=0
    else:
        # 累加计算FPs与TPs
        fpc = (1 - tp).cumsum()
        tpc = (tp).cumsum()

        # Recall
        recall_curve = tpc / (n_gt + 1e-16)  # TP/(TP + FN) (TP+FN)为当前类人工标注目标数量

        # Precision
        precision_curve = tpc / (tpc + fpc)  # TP/(TP + FP) (TP+FP)为预测框的数量

        # 从召回率-精确率曲线计算AP
        ap=compute_ap(recall_curve, precision_curve)

    return ap


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
        #ap[k] = ap_per_class(scores, targets)
    return 100 * ap.mean()

def asses(targs, preds):
    targs, preds = torch.from_numpy(targs).cuda(), torch.from_numpy(preds).cuda()
    re = torch.zeros_like(preds).cuda()
    re[preds>0.5]=1

    #v,ix = preds.topk(10,1,True,True)
    #top4_mask = torch.scatter(torch.zeros_like(preds).cuda(), 1, ix, 1)

    #re = re * top4_mask
    correct_re = re + targs
    correct_num = (correct_re==2).float().sum()
    pred_num = (re==1).float().sum()
    total_num = (targs==1).float().sum()
    
    R = 100.*correct_num/total_num
    P = 0. if pred_num==0 else 100.*correct_num/pred_num
    F = 0. if R==0 or P==0 else 2*R*P/(R+P)
    return F, P, R

def AP_partial(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    cnt_class_with_no_neg = 0
    cnt_class_with_no_pos = 0
    cnt_class_with_no_labels = 0

    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]

        # Filter out samples without label
        idx = (targets != -1)
        scores = scores[idx]
        targets = targets[idx]
        if len(targets) == 0:
            cnt_class_with_no_labels += 1
            ap[k] = -1
            continue
        elif sum(targets) == 0:
            cnt_class_with_no_pos += 1
            ap[k] = -1
            continue
        if sum(targets == 0) == 0:
            cnt_class_with_no_neg += 1
            ap[k] = -1
            continue

        # compute average precision
        ap[k] = average_precision(scores, targets)

    idx_valid_classes = np.where(ap != -1)[0]
    ap_valid = ap[idx_valid_classes]
    map = 100 * np.mean(ap_valid)

    # Compute macro-map
    targs_macro_valid = targs[:, idx_valid_classes].copy()
    targs_macro_valid[targs_macro_valid <= 0] = 0  # set partial labels as negative
    n_per_class = targs_macro_valid.sum(0)  # get number of targets for each class
    n_total = np.sum(n_per_class)
    map_macro = 100 * np.sum(ap_valid * n_per_class / n_total)

    return ap, map, map_macro, cnt_class_with_no_labels, cnt_class_with_no_neg, cnt_class_with_no_pos


def mAP_partial(targs, preds):
    """ mean Average precision for partial annotated validatiion set"""

    if np.size(preds) == 0:
        return 0
    results = AP_partial(targs, preds)
    mAP = results[1]
    return mAP

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, spread_targets=True):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

        self.spread_targets = spread_targets
        if self.spread_targets:
            self.targets_all = self.get_targets()

    def get_targets(self):

        targets_all = {}
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)

            output = torch.zeros((3, 80), dtype=torch.long)
            for obj in target:
                if obj['area'] < 32 * 32:
                    output[0][self.cat2cat[obj['category_id']]] = 1
                elif obj['area'] < 96 * 96:
                    output[1][self.cat2cat[obj['category_id']]] = 1
                else:
                    output[2][self.cat2cat[obj['category_id']]] = 1
            target = output
            target = target.max(dim=0)[0]
            targets_all[img_id] = target
        return targets_all

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        if self.spread_targets:
            target = self.targets_all[img_id]
        else:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)

            output = torch.zeros((3, 80), dtype=torch.long)
            for obj in target:
                if obj['area'] < 32 * 32:
                    output[0][self.cat2cat[obj['category_id']]] = 1
                elif obj['area'] < 96 * 96:
                    output[1][self.cat2cat[obj['category_id']]] = 1
                else:
                    output[2][self.cat2cat[obj['category_id']]] = 1
            target = output
            target = target.max(dim=0)[0]

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
