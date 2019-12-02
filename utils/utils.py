import glob
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import shapely
from shapely.geometry import Polygon,MultiPoint  
from shapely.geometry import Polygon
import re

from utils.nms.r_nms import r_nms
from . import torch_utils  # , google_utils

matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def hyp_parse(hyp_path):
    hyp = {}
    keys = [] #用来存储读取的顺序
    with open(hyp_path,'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip())==0 : continue
            v = line.strip().split(':')
            try:
                hyp[v[0]] = float(v[1].strip().split(' ')[0])
            except:
                hyp[v[0]] = eval(v[1].strip().split(' ')[0])
            keys.append(v[0])
        f.close()
        print(hyp)
    return hyp


def floatn(x, n=3):  # format floats to n decimals
    return float(format(x, '.%gf' % n))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

# 打印模型的参数,如layers,parameters等
def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report is 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    ni = len(labels)  # number of images
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    weights = np.hstack([gpi * ni - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco_class_weights():  # frequency of each class in coco train2014
    n = [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380]
    weights = 1 / torch.Tensor(n)
    weights /= weights.sum()
    # with open('data/coco.names', 'r') as f:
    #     for k, v in zip(f.read().splitlines(), n):
    #         print('%20s: %g' % (k, v))
    return weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def xywha2coors(x):
    # 带旋转角度，顺时针正，+-0.5pi;返回四个点坐标
    coors = []  ## 一张图的所有box
    for obj in x:
        cx = obj[0]; cy = obj[1]; w = obj[2]; h = obj[3]; a = obj[4]
        xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
        t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
        x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
        y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
        x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
        y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
        x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
        y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
        x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
        y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 
        coors.append(np.array([[float(x0),float(y0)],[float(x1),float(y1)],[float(x2),float(y2)],[float(x3),float(y3)]]))
    return coors



def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xywha) from img1_shape to img0_shape
    gain = max(img1_shape) / max(img0_shape)  # gain  = old / new  < 1
    # 裁去灰色边
    coords[:, 0] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
    coords[:, 1] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    # clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xywha bounding boxes to image shape (height, width)
    coors = torch.stack([get_rotated_coors(box[:5]) for box in boxes])
    clipx = (coors[:,::2] <img_shape[1]).all(1) # clip x
    clipy = (coors[:,1::2]<img_shape[0]).all(1) # clip y
    clip = clipx*clipy
    boxes = boxes[clip]

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r.append(recall[-1])

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p.append(precision[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall, precision))

            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.concatenate(([0.], recall)), np.concatenate(([0.], precision)))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim(0, 1)
            fig.tight_layout()
            fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under PR curve, looking for points where x axis (recall) changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# 输入的是xywha
# 支持输入单个的box和多box tensor的iou计算，其中默认box1为单个的
def skew_bbox_iou(box1, box2, GIoU=False):
    ft = torch.cuda.FloatTensor 
    if isinstance(box1,list):  box1 = ft(box1)
    if isinstance(box2,list):  box2 = ft(box2)
    if len(box1.shape) < len(box2.shape):   # 输入的单box维度不匹配时，unsqueeze一下
        box1 = box1.unsqueeze(0)
    if not box1.shape == box2.shape:
        box1 = box1.repeat(len(box2),1)

    box1 = box1[:,:5]
    box2 = box2[:,:5]

    if GIoU: 
        mode = 'giou'
    else: 
        mode = 'iou'
    
    ious = []
    for i in range(len(box2)):
        r_b1 = get_rotated_coors(box1[i])
        r_b2 = get_rotated_coors(box2[i])
        
        ious.append(skewiou(r_b1, r_b2, mode=mode))

    # if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
    #     c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
    #     c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
    #     c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
    #     return iou - (c_area - union_area) / c_area  # GIoU

    return ft(ious)

def bbox_iou(box1, box2, GIoU=False):
    # Returns the IoU of box1 to box2.x, y, w, h = box1
    b1_x1, b1_x2 = box1[:,0] - box1[:,2] / 2, box1[:,0] + box1[:,2] / 2
    b1_y1, b1_y2 = box1[:,1] - box1[:,3] / 2, box1[:,1] + box1[:,3] / 2
    b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2
    b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
        c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
        return iou - (c_area - union_area) / c_area  # GIoU

    return iou

    
def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    if box1.shape != box2.shape:
        box2 = box2.t()
        # w, h = box1
        w1, h1 = box1[0], box1[1]
        w2, h2 = box2[0], box2[1]
    else:
        w1, h1 = box1[:,0], box1[:,1]
        w2, h2 = box2[:,0], box2[:,1]
    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area  # iou


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# targets 是[num_boxes,7](当前batch的)的gt张量
def compute_loss(p, targets, model, hyp):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    # lerg的定位损失包括giou和角度回归两部分
    lcls, liou, lobj, lreg = ft([0]), ft([0]), ft([0]), ft([0])
    # build_targets 完成了两件事：
    # - 将target标注缩放放到三个yolo层上
    # - 选出各个yolo层上和label box iou较大的anchor将其输出（通过indices），用于回归/计算loss
    tcls, tbox, indices, anchor_vec = build_targets(model, targets, hyp) # 参数含义参见build_targets的说明,参数长度均为3(yolo层的个数)
    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures

    # lx, ly, lw, lh, la = ft([0]), ft([0]), ft([0]), ft([0]), ft([0])    # 坐标回归的loss项

    # Define criteria
    # pos_weight是解决不平衡问题参数,在loss中作为正目标的损失函数系数,>1会减小错误负样本的权重
    # 该参数的len个和类别相同,这里是cls和obj_conf的BCE所以一维参数就行
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]))
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()      # weight=model.class_weights
    SM = nn.SmoothL1Loss(reduction='mean')


    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g), FocalLoss(SM, g)

    # Compute losses
    for i, pi in enumerate(p):  # layer index, layer predictions每次遍历一个yolo层的特征  eg. torch.Size([2, 36, 13, 13, 7])
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx 取出当前特征图尺度的所有batch gt的信息
        tobj = torch.zeros_like(pi[..., 0])  # target obj 尺度和yolo层特征图一致,只是去掉channel  eg. torch.Size([2, 36, 13, 13])

        # Compute losses
        nb = len(b)
        num_anchors = len(a)
        if nb:  # number of targets
            # [b, a, gj, gi]是build_target的结果,是经过筛选合适回归(iou大)的anchor位置(如果有筛选)]
            # ps是对前向pi特征图取出物体中心点负责的grid cell，len和预选留下的anchor一致
            # pi: [bs, na, f_w, f_h, 8]   ps: [num_box , 7]  7 = xywha + obj +classses
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj 因为在这些位置一定有物体,所以gt的obj conf =1,取余gt为0
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pbox = torch.cat((pxy, ps[:, 2:5]) , 1)       


        # 回归定位：
            ## iou_loss 都是特征图尺寸下的计算
            ng, anchor_vec = model.module_list[model.yolo_layers[i]].ng, model.module_list[model.yolo_layers[i]].anchor_vec
            rect_p = pbox.clone()
            rect_p[:,2:4]  = torch.exp(rect_p[:,2:4])*anchor_vec[a][:,:-1]
            rect_gt = tbox[i].clone()
            rect_gt[:,2:4] = torch.exp(rect_gt[:,2:4])*anchor_vec[a][:,:-1]
            if rect_p.dtype == torch.float16:   # apex适配gt的类型
                rect_gt = rect_gt.half()
            iou = wh_iou(rect_gt[:,2:4], rect_p[:,2:4])

            liou += (1.0 - iou).mean()  # giou loss
            lreg = lreg+ SM(pbox[:,[0,1]],tbox[i][:,[0,1]]) + 2*SM(pbox[:,4],tbox[i][:,4]) + liou * h['giou']  
            # lreg = lreg +  SM(pbox,tbox[i])
                


        # 分类loss(针对多类别检测)
            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                # t是类别的gt,下面根据target设置分类得分的gt mask 然后和forward的对应结果计算cls loss
                t = torch.zeros_like(ps[:, 6:])  # targets classes得分 [num_box,cls]
                t[range(nb), tcls[i]] = 1.0          # gt的位置得分是1
                lcls += BCEcls(ps[:, 5:], t)     # BCE
                # lcls += CE(ps[:, 5:], tcls[i]) # CE

                # Instance-class weighting (use with reduction='none')
                # nt = t.sum(0) + 1  # number of targets per class
                # lcls += (BCEcls(ps[:, 5:], t) / nt).mean() * nt.mean()  # v1
                # lcls += (BCEcls(ps[:, 5:], t) / nt[tcls[i]].view(-1,1)).mean() * nt.mean()  # v2

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        # 置信度obj_conf loss
        if 'default' in arc:  # seperate obj and cls
            lobj += BCEobj(pi[..., 5], tobj)  # obj loss

        elif 'BCE' in arc:  # unified BCE (80 classes)
            t = torch.zeros_like(pi[..., 6:])  # targets
            if nb:
                t[b, a, gj, gi, tcls[i]] = 1.0
            lobj += BCE(pi[..., 6:], t)

        elif 'CE' in arc:  # unified CE (1 background + 80 classes)
            t = torch.zeros_like(pi[..., 0], dtype=torch.long)  # targets
            if nb:
                t[b, a, gj, gi] = tcls[i] + 1
            lcls += CE(pi[..., 5:].view(-1, model.nc + 1), t.view(-1))

    lobj *= h['obj']
    lcls *= h['cls']
    lreg *= h['reg']
    loss =  lobj + lcls + lreg

    return loss, torch.cat(( lobj, lcls, lreg, loss)).detach()

# 输入: targets 是[num_boxes,6](当前batch的)的gt张量
# 输出: (在当前batch的所有gt)
#     - tcls      类别id list
#     - tbox      编码gt的xywh (xy是一个cell内的浮点小数)
#     - indices   提供索引信息,包含: b-图片index ; a-anchor索引(与gt iou过小的会被筛掉) ; gj,gi-grid cell的索引
#     - av        anchor的选择,对哪个gt用哪几个的anchor
# 以上输出都是list,len是yolo层的个数
# build的用处：选取yolo层上对应的anchor和gxy位置，这些位置计算gt的缩放真值并返回；在计算loss时只用这几个gt，所以相当于指定这些位置的anchor向gt回归
def build_targets(model, targets, hyp):
    # targets = [image, class, x, y, w, h, a]
    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    ft = torch.cuda.FloatTensor if targets.is_cuda else torch.Tensor

    for id,i in enumerate(model.yolo_layers):    # yolo层的index: [82, 94, 106]
        # get number of grid points and anchor vec for this yolo layer
        # 这里的 anchor_vec 是缩放到降采样步长的
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # iou of targets-anchors
        targets[:,4] += targets[:,5]*(hyp['context_factor']-1)
        targets[:,5] *= hyp['context_factor']

        t, a = targets, []
        gwha = t[:, 4:7].clone() # 注意深拷贝！
        gwha[:,:-1] *= ng        # 缩放到当前yolo层尺寸上
        if nt:
            # shape: [num_anchor, num_boxes],所有gt和当前yolo层上anchor的iou

            # Method1:计算anchor正框的iou
            all_ious = torch.stack([wh_iou(x, gwha[:,:-1]) for x in anchor_vec[:,:-1]], 0) # (num_anchors,gts)-->(72,15)

            # Method2:计算斜框的iou
            # _anchor_vec = torch.cat((ft((0,0)).repeat(len(anchor_vec),1),anchor_vec),1) # torch.Size([18, 8]) 8=xyxyxyxy
            # _tar_anchors = torch.cat((ft((0,0)).repeat(len(gwha),1),gwha),1)   #torch.Size([7, 8]) 7=len(t) in a bs

            # _ious = torch.stack([skew_bbox_iou(x,_tar_anchors) for x in _anchor_vec],0)  # ious.shape: (num_anchors, num_tragets) num_anchors = 36
            # anchor和gt的可视化，观察iou是否有误，及时调整anchor超参数
            # _anchors  = [get_rotated_coors(x) for x in _anchor_vec]
            # _tars = [get_rotated_coors(x) for x in _tar_anchors]
            # strides = [32,16,8]*2
            # stride = strides[id]
            # img = np.zeros((416*2,416*2,3), np.uint8)
            # img.fill(255)
            # for ans in _anchors:
            #     ans *= stride*2  # 额外放大一倍便于观察
            #     ans += 416  # 移到中间
            #     ans = ans.cpu().numpy().reshape(4,2).astype(np.int32)
            #     img = cv2.polylines(img,[ans],True,(0,0,255),1)
            #     for tars in _tars:
            #         tars *= stride*2
            #         tars += 416
            #         tars = tars.cpu().numpy().reshape(4,2).astype(np.int32)
            #         img = cv2.polylines(img,[tars],True,(255,0,0),1)
            #     cv2.imshow('anchor_show', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            use_best_anchor = False
            if use_best_anchor:
                ious, a = all_ious.max(0)  # best iou and anchor  注意：是返回每一列的最大值
            else:  # use all anchors
                na = len(anchor_vec)  # number of anchors
                # a是长度 num_box * na 的行向量；每num_box元素是na的index；如3anchor,num_box=2,则a=[0,0,1,1,2,2]
                # 用于anchor对box的mask
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)  # (72,15)拉成squence,每15个为一组
                # t就是将target在num_box方向扩展na倍 [num_boxes*na，7]，7中单独每个元素取出来如t[:,-1]也是72*15拉成squence,每15个为一组
                t = targets.repeat([na, 1]) 
                gwha = gwha.repeat([na, 1])   # 同理将box wh也扩展na倍 [num_boxes*na,2],单个元素同上
                ious = all_ious.view(-1)  # use all ious  展开成[num_boxes*na]的行向量


            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            reject = True   # 训练时筛掉和gt的iou过小, 以及角度差太大（至少阈值0.5pi，确保回归范围无错）的anchor
            if reject:
                angle_offset = gwha[:,-1] - anchor_vec[:,-1].view((-1, 1)).repeat([1,nt]).view(-1)
                j1 = ious > model.hyp['iou_t']  # iou threshold hyperparameter
                j2 = abs(angle_offset) <  model.hyp['ang_t']  # angle threshold hyperparameter
                j = j1 * j2
                # if not j.reshape(all_ious.shape).max(0)[0].all():   # 存在某个gt在当前了layer没有合适的anchor，保留最大iou的anchor
                #     # import ipdb; ipdb.set_trace()
                #     gt_no_anchor_index = (j.reshape(all_ious.shape).max(0)[0]==False).nonzero().squeeze(-1)
                #     max_iou_index = all_ious[:,gt_no_anchor_index].max(0)[1]  
                #     j[gt_no_anchor_index+na * max_iou_index] = True
                # import ipdb; ipdb.set_trace()
                t, a, gwha = t[j], a[j], gwha[j]

        
        # Indices 这个变量只提供索引信息: b-图片index ; a-anchor索引 ; gj,gi grid cell的索引
        b, c = t[:, :2].long().t()  # target image, class 分别是图像index和类别id分离出来,便于索引
        gxy = t[:, 2:4] * ng  # grid x, y  将选出的gt box的xy缩放到当前遍历yolo层的特征图尺寸上去
        gi, gj = gxy.long().t()  # grid x, y indices gi gj就是gxy分别取整得到的grid cell编号
        indices.append((b, a, gj, gi))


        # GIoU   
        gxy -= gxy.floor()  # xy  # 在一个grid cell内的坐标(即原坐标减去grid cell的位置) 
        gwha[:,:2] = torch.log(gwha[:,:2] / anchor_vec[a][:,:2]) 
        gwha[:, 2] = torch.tan(gwha[:, 2] - anchor_vec[a][:, 2])
        tbox.append(torch.cat((gxy, gwha), 1))  # xywha (grids)  tbox编码xywa (xy是一个cell内的偏移)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() <= model.nc, 'Target classes exceed model classes'

    return tcls, tbox, indices, av


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x, y, w, h, a, object_conf, class_conf, class)
    """
    # prediction: torch.Size([1, 8190, 8]) 第一维bs是图片数,第二维是所有的proposal,第三维是xywh + conf + classes(这里是三类)
    min_wh = 2  # (pixels) minimum box width and height
    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Multiply conf by class conf to get combined confidence
        # max(1)是按照1维搜索,对每个proposal取出多分类分数,得到最大的那个值
        # 返回值class_conf和索引class_pred,索引就是类别所属
        class_conf, class_pred = pred[:, 6:].max(1)     # max(1) 是每行找最大的，即当前proposal最可能是哪个类
        pred[:, 5] *= class_conf            # 乘以conf才是真正的得分,赋值到conf的位置

        # Select only suitable predictions
        # 先创造一个满足要求的索引bool矩阵,然后据此第二步进行索引
        # 条件为:1.最大类的conf大于预设值   2.该anchor的预测wh大于2像素   3.非nan或无穷
        i = (pred[:, 5] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]  # bool向量筛掉False的conf
        class_pred = class_pred[i].unsqueeze(1).float() # torch.Size([num_of_proposal]) --> torch.Size([num_of_proposal,1])便于后面的concat

        use_cuda_nms = True
        
        if  use_cuda_nms:
            det_max = []
            pred = torch.cat((pred[:, :6], class_conf.unsqueeze(1), class_pred), 1)
            pred = pred[(-pred[:, 5]).argsort()]
            for c in pred[:, -1].unique():
                dc = pred[pred[:, -1] == c]
                dc = dc[(-dc[:, 5]).argsort()]
                if len(dc)>100:
                    dc = dc[:100]
                # Non-maximum suppression
                inds = r_nms(dc[:,:6], nms_thres)
                det_max.append(dc[inds])
            if len(det_max):
                det_max = torch.cat(det_max)  # concatenate
                output[image_i] = det_max[(-det_max[:, 5]).argsort()]  # sort

        else:
            # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
            pred = torch.cat((pred[:, :6], class_conf.unsqueeze(1), class_pred), 1)

            # Get detections sorted by decreasing confidence scores
            pred = pred[(-pred[:, 5]).argsort()]

            det_max = []
            nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)

            for c in pred[:, -1].unique():
                dc = pred[pred[:, -1] == c]  # select class c #  shape [num,7]  7 = (x1, y1, x2, y2, object_conf, class_conf)
                n = len(dc)
                if n == 1:
                    det_max.append(dc)  # No NMS required if only 1 prediction
                    continue
                elif n > 100:
                    dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

                # Non-maximum suppression
                if nms_style == 'OR':  # default
                    # METHOD1
                    # ind = list(range(len(dc)))
                    # while len(ind):
                    # j = ind[0]
                    # det_max.append(dc[j:j + 1])  # save highest conf detection
                    # reject = (skew_bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                    # [ind.pop(i) for i in reversed(reject)]

                    # METHOD2
                    while dc.shape[0]:
                        det_max.append(dc[:1])  # save highest conf detection
                        if len(dc) == 1:  # Stop if we're at the last detection
                            break
                        iou = skew_bbox_iou(dc[0], dc[1:])  # iou with other boxes
                        dc = dc[1:][iou < nms_thres]  # remove ious > threshold

                elif nms_style == 'AND':  # requires overlap, single boxes erased
                    while len(dc) > 1:
                        iou = skew_bbox_iou(dc[0], dc[1:])  # iou with other boxes
                        if iou.max() > 0.5:
                            det_max.append(dc[:1])
                        dc = dc[1:][iou < nms_thres]  # remove ious > threshold

                elif nms_style == 'MERGE':  # weighted mixture box
                    while len(dc):
                        if len(dc) == 1:
                            det_max.append(dc)
                            break
                        # 有个bug:如果当前一批box中和最高conf(排序后是第一个也就是dc[0])的iou都小于nms_thres,
                        # 那么i全为False,导致weights=[],从而weights.sum()=0导致dc[0]变成nan!
                        i = skew_bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes, 返回的也是boolean,便于后面矩阵索引和筛选
                        weights = dc[i, 5:6]    # 大于nms阈值的重复较多的proposal,取出conf
                        assert len(weights)>0, 'Bugs on MERGE NMS!!'
                        dc[0, :5] = (weights * dc[i, :5]).sum(0) / weights.sum()    # 将最高conf的bbox代之为大于阈值的所有bbox加权结果(conf不变,变了也没意义)
                        det_max.append(dc[:1])
                        dc = dc[i == 0]         # bool的false等价于0,这一步将dc中的已经计算过的predbox剔除掉

                elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                    sigma = 0.5  # soft-nms sigma parameter
                    while len(dc):
                        if len(dc) == 1:
                            det_max.append(dc)
                            break
                        det_max.append(dc[:1])
                        iou = skew_bbox_iou(dc[0], dc[1:])  # iou with other boxes
                        dc = dc[1:]
                        dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                        # dc = dc[dc[:, 4] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

            if len(det_max):
                det_max = torch.cat(det_max)  # concatenate
                import ipdb; ipdb.set_trace()
                output[image_i] = det_max[(-det_max[:, 5]).argsort()]  # sort
            

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary (per output layer):')
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for l in model.yolo_layers:  # print pretrained biases
        if multi_gpu:
            b = model.module.module_list[l - 1][0].bias.view(3, -1)  # bias 3x85
        else:
            b = model.module_list[l - 1][0].bias.view(3, -1)  # bias 3x85
        print('regression: %5.2f+/-%-5.2f ' % (b[:, :4].mean(), b[:, :4].std()),
              'objectness: %5.2f+/-%-5.2f ' % (b[:, 4].mean(), b[:, 4].std()),
              'classification: %5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std()))


def strip_optimizer(f='weights/last.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    x = torch.load(f)
    x['optimizer'] = None
    torch.save(x, f)


def create_backbone(f='weights/last.pt'):  # from utils.utils import *; create_backbone()
    # create a backbone from a *.pt file
    x = torch.load(f)
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    for p in x['model'].values():
        try:
            p.requires_grad = True
        except:
            pass
    torch.save(x, 'weights/backbone.pt')


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/val2014/'):
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def select_best_evolve(path='evolve*.txt'):  # from utils.utils import *; select_best_evolve()
    # Find best evolved mutation
    for file in sorted(glob.glob(path)):
        x = np.loadtxt(file, dtype=np.float32, ndmin=2)
        print(file, x[fitness(x).argmax()])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images



def print_mutation(hyp, results, bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    np.savetxt('evolve.txt', x[np.argsort(-fitness(x))], '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    return x[:, 2] * 0.8 + x[:, 3] * 0.2  # weighted mAP and F1 combination


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    coor = torch.cuda.FloatTensor(get_rotated_coors(x)).reshape(4,2)
    img = cv2.polylines(img, [coor.cpu().numpy().astype(np.int32)], True, color, tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        label_coor = get_rotated_coors(torch.FloatTensor([coor[0,0], coor[0,1], t_size[0], t_size[1], x[-1]]))
        # cv2.polylines(img,[label_coor.reshape(4,2).cpu().numpy().astype(np.int32)],True, color, tl)
        cv2.putText(img, label, tuple(coor[0].cpu().numpy()) , 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


# 绘制斜框gt，看看增强是否有效，label是否加载正确
def plot_images(imgs, targets, paths=None, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        img = imgs[i]
        img *= 255.0  # 从归一化的浮点数映射回原图便于opencv画多边形box（matplotlib本身可以用浮点画图）
        img = np.ascontiguousarray(img, dtype=np.uint8)  
        img = img.transpose(1,2,0)  # BGR to RGB, to 3x416x416
        
        boxes = xywha2coors(targets[targets[:, 0] == i, 2:7])
        if len(boxes)>0:   # 不一定都有gt(增强后可能没了)
            for box in boxes:
                box[:,0]*=w; box[:,1]*=h
                box = box.astype(np.int32)
                r,g,b = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

                img = cv2.polylines(img,[box],True,(r,g,b),2)
        else:
            # cv2.imshow('p',img)
            # cv2.waitKey(0)
            continue
        img = img.get().astype('i')
        plt.subplot(ns, ns, i + 1).imshow(img)
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()




def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig('hist2d.jpg', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    fig.tight_layout()
    plt.savefig('hist1d.jpg', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot test.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32)
    x = x.T

    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    fig.tight_layout()
    plt.savefig('targets.jpg', dpi=200)


def plot_evolution_results(hyp):  # from utils.utils import *; plot_evolution_results(hyp)
    # Plot hyperparameter evolution results in evolve.txt
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    weights = (f - f.min()) ** 2  # for weighted results
    fig = plt.figure(figsize=(12, 10))
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 5]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(4, 5, i + 1)
        plt.plot(mu, f.max(), 'o', markersize=10)
        plt.plot(y, f, '.')
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        print('%15s: %.3g' % (k, mu))
    fig.tight_layout()
    plt.savefig('evolve.png', dpi=200)


def plot_results(start=0, stop=0):  # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    fig, ax = plt.subplots(2, 5, figsize=(14, 7))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP', 'F1']
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
            ax[i].plot(x, y, marker='.', label=f.replace('.txt', ''))
            ax[i].set_title(s[i])
            if i in [5, 6, 7]:  # share train and val loss y axes
                ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

    fig.tight_layout()
    ax[1].legend()
    fig.savefig('results.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training results files 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP', 'val', 'val', 'val', 'Recall', 'F1']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5))
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                if i in [0, 1, 2]:
                    y[y == 0] = np.nan  # dont show zero loss values
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.tight_layout()
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def version_to_tuple(version):
    # Used to compare versions of library
    return tuple(map(int, (version.split("."))))

# anchor对齐阶段计算iou
def skewiou(box1, box2,mode='iou',return_coor = False):
    a=box1.reshape(4, 2)   
    b=box2.reshape(4, 2)
    # 所有点的最小凸的表示形式，四边形对象，会自动计算四个点，最后顺序为：左上 左下  右下 右上 左上
    poly1 = Polygon(a).convex_hull  
    poly2 = Polygon(b).convex_hull
    if not poly1.is_valid or not poly2.is_valid:
        print('formatting errors for boxes!!!! ')
        return 0
    if  poly1.area == 0 or  poly2.area  == 0 :
        return 0

    inter = Polygon(poly1).intersection(Polygon(poly2)).area
    if   mode == 'iou':
        union = poly1.area + poly2.area - inter
    elif mode =='tiou':
        union_poly = np.concatenate((a,b))   #合并两个box坐标，变为8*2
        union = MultiPoint(union_poly).convex_hull.area
        coors = MultiPoint(union_poly).convex_hull.wkt
    elif mode == 'giou':
        union_poly = np.concatenate((a,b))   
        union = MultiPoint(union_poly).envelope.area
        coors = MultiPoint(union_poly).envelope.wkt
    elif mode== 'r_giou':
        union_poly = np.concatenate((a,b))   
        union = MultiPoint(union_poly).minimum_rotated_rectangle.area
        coors = MultiPoint(union_poly).minimum_rotated_rectangle.wkt
    else:
        print('incorrect mode!')

    if union == 0:
        return 0
    else:
        if return_coor:
            return inter/union,coors
        else:
            return inter/union


def get_rotated_coors(box):
    assert len(box) > 0 , 'Input valid box!'
    cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 

    if isinstance(x0,torch.Tensor):
        r_box=torch.cat([x0.unsqueeze(0),y0.unsqueeze(0),
                         x1.unsqueeze(0),y1.unsqueeze(0),
                         x2.unsqueeze(0),y2.unsqueeze(0),
                         x3.unsqueeze(0),y3.unsqueeze(0)], 0)
    else:
        r_box = np.array([x0,y0,x1,y1,x2,y2,x3,y3])
    return r_box
