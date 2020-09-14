import glob
import os
import random
import shutil
from pathlib import Path
import copy
import time

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
import torch.nn.functional as F

from utils.nms.r_nms import r_nms
from utils.utils import *



class NoSampler(object):
    def __init__(self):
        pass
    def __call__(self,target, pos_indices):
        return target==0


class PropSampler(object):
    def __init__(self, neg_ratio = 3):
        self.ratio = neg_ratio

    def __call__(self, target, pos_indices):
        target_mask = target.clone()
        target_mask.fill_(-1)
        b, a, gj, gi = pos_indices
        dim_b, dim_a, dim_gj, dim_gi = target_mask.shape
        pos_num = len(b)
        neg_num = self.ratio * pos_num
        total_num = target_mask.numel()
        if len(b)!=0:
            pos_index = set((b * dim_a  * dim_gj * dim_gi + a *  dim_gj * dim_gi + gj * dim_gi + gi).tolist())
            neg_index = random.sample(set([i for i in range(total_num)]) - pos_index, neg_num)  # type list
            ngb = [x // (dim_a * dim_gj * dim_gi) for x in neg_index]
            nga = [x % (dim_a * dim_gj * dim_gi) // (dim_gj * dim_gi) for x in neg_index]
            nggj = [x % (dim_a * dim_gj * dim_gi) % (dim_gj * dim_gi) // (dim_gi) for x in neg_index]
            nggi = [x % (dim_a * dim_gj * dim_gi) % (dim_gj * dim_gi) % (dim_gi) for x in neg_index]
            target_mask[b, a, gj, gi] = 1
            target_mask[ngb, nga, nggj, nggi] = 0

        mask = target_mask!=-1
        return mask


class GradualSampler(object):
    def __init__(self, init_ratio = 3, max_epoches = 100):
        self.init_ratio = init_ratio
        self.max_epoches = max_epoches

    def __call__(self, target, pos_indices, epoch):
        target_mask = target.clone()
        target_mask.fill_(-1)
        b, a, gj, gi = pos_indices
        dim_b, dim_a, dim_gj, dim_gi = target_mask.shape
        pos_num = len(b)
        total_num = target_mask.numel()
        
        if pos_num != 0:
            self.final_ratio = (total_num - pos_num)/pos_num
            self.ratio = (0.5 - 0.5 * math.cos(math.pi * epoch / self.max_epoches)) * \
                        (self.final_ratio - self.init_ratio) + self.init_ratio

            neg_num = int(np.floor(self.ratio * pos_num))
            assert neg_num + pos_num <= total_num, 'neg_num overflow, minus 1 pls.'

            # pos_index = set((b * dim_a  * dim_gj * dim_gi + a *  dim_gj * dim_gi + gj * dim_gi + gi).tolist())
            # # all_index = torch.cuda.LongTensor([i for i in range(total_num)])
            # all_index = random.sample(range(total_num),total_num)
            # judge = np.ones_like(all_index)==1
            # for elem in pos_index:
            #     judge = judge & (all_index != elem)  
            # subtract = np.array(all_index)[judge]
            # # subtract = list(set([i for i in range(total_num)]) - pos_index)
            # neg_index = subtract[:neg_num]

            pos_index = set((b * dim_a  * dim_gj * dim_gi + a *  dim_gj * dim_gi + gj * dim_gi + gi).tolist())
            neg_index = random.sample(set([i for i in range(total_num)]) - pos_index, neg_num)  # type list

            # pos_index = set((b * dim_a  * dim_gj * dim_gi + a *  dim_gj * dim_gi + gj * dim_gi + gi).tolist())
            # neg_index = random.sample(set([i for i in range(total_num)]) - pos_index, neg_num)  # type list
            ngb = [x // (dim_a * dim_gj * dim_gi) for x in neg_index]
            nga = [x % (dim_a * dim_gj * dim_gi) // (dim_gj * dim_gi) for x in neg_index]
            nggj = [x % (dim_a * dim_gj * dim_gi) % (dim_gj * dim_gi) // (dim_gi) for x in neg_index]
            nggi = [x % (dim_a * dim_gj * dim_gi) % (dim_gj * dim_gi) % (dim_gi) for x in neg_index]
            target_mask[b, a, gj, gi] = 1
            target_mask[ngb, nga, nggj, nggi] = 0
        mask = target_mask!=-1
        return mask
    
    def vis_sampler_curve():
        ratios = np.array([(0.5 - 0.5 * math.cos(math.pi * x / self.max_epoches)) * \
                                (self.final_ratio-self.init_ratio)+self.init_ratio for x in range(self.max_epoches)])
        epoches = np.array([x for x in range(self.max_epoches)])
        plt.plot(epoches,ratios,color='red')
        plt.title('GradualSampler')
        plt.xlabel('epoch')
        plt.ylabel('ratio')
        plt.savefig('sampler.png')
        # plt.show()




def h_iou_loss(input,target):
    iou = wh_iou(input, target)
    loss = 1.0 - iou
    return loss # 返回的是矩阵，以防别的地方会用



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
    tcls, tbox, indices, av, square_ious = [], [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    ft = torch.cuda.FloatTensor if targets.is_cuda else torch.Tensor

    for id,i in enumerate(model.yolo_layers):    # yolo层的index: [82, 94, 106]
        # get number of grid points and anchor vec for this yolo layer
        # 这里的 anchor_vec 是缩放到降采样步长的
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # context setting
        targets[:,4] += targets[:,5]*(hyp['context_factor']-1)
        targets[:,5] *= hyp['context_factor']
        
        # iou of targets-anchors
        t, a = targets, []
        gwha = t[:, 4:7].clone() # 注意深拷贝！
        gwha[:,:-1] *= ng        # 缩放到当前yolo层尺寸上
        if nt:
            # shape: [num_anchor, num_boxes],所有gt和当前yolo层上anchor的iou

            # Method1:计算anchor正框的iou
            all_ious = torch.stack([wh_iou(x, gwha[:,:-1]) for x in anchor_vec[:,:-1]], 0) # (num_anchors,gts)-->(72,15)

            use_best_anchor = False
            if use_best_anchor:
                ious, a = all_ious.max(0)  # best iou and anchor  注意：是返回每一列的最大值
            else:  # use all anchors
                na = len(anchor_vec)  # number of anchors
                # a是长度 num_box * na 的行向量；每num_box元素是na的index；如3anchor,num_box=2,则a=[0,0,1,1,2,2]
                # 用于anchor对box的mask
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1).cuda()  # (72,15)拉成squence,每15个为一组
                # t就是将target在num_box方向扩展na倍 [num_boxes*na，7]，7中单独每个元素取出来如t[:,-1]也是72*15拉成squence,每15个为一组
                t = targets.repeat([na, 1]) 
                gwha = gwha.repeat([na, 1])   # 同理将box wh也扩展na倍 [num_boxes*na,2],单个元素同上
                ious = all_ious.view(-1)  # use all ious  展开成[num_boxes*na]的行向量
                square_ious.append(ious)

        # Indices 这个变量只提供索引信息: b-图片index ; a-anchor索引 ; gj,gi grid cell的索引
        b, c = t[:, :2].long().t()  # target image, class 分别是图像index和类别id分离出来,便于索引
        gxy = t[:, 2:4] * ng  # grid x, y  将选出的gt box的xy缩放到当前遍历yolo层的特征图尺寸上去
        gi, gj = gxy.long().t()  # grid x, y indices gi gj就是gxy分别取整得到的grid cell编号
        indices.append([b, a, gj, gi])

        # gt_convert   
        gxy -= gxy.floor()  # xy  # 在一个grid cell内的坐标(即原坐标减去grid cell的位置) 
        t_gwha = gwha.clone()   # 留个备份后面reject用的到
        # gwha[:,:2] = torch.log(gwha[:,:2] / anchor_vec[a][:,:2]) 
        # gwha[:, 2] = torch.tan(gwha[:, 2] - anchor_vec[a][:, 2])
        tbox.append(torch.cat((gxy, gwha), 1))  # xywha (grids)  tbox编码xywa (xy是一个cell内的偏移)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() <= model.nc, 'Target classes exceed model classes'

    # reject anchors below iou_thres 
    # 实现方法是mask矩阵j，shape为(num_layers, nt * 72); 正样本1 负样本0 ign -1
    reject = True   # 训练时筛掉和gt的iou过小, 以及角度差太大（至少阈值0.5pi，确保回归范围无错）的anchor
    if reject and nt :
        angle_offset = abs(t_gwha[:,-1] - anchor_vec[:,-1].view((-1, 1)).repeat([1,nt]).view(-1))   # 角度mask没必要区分layer，不同layer的angle一样，只是尺度大小不同
        angle_offset[angle_offset > 0.5*math.pi] = math.pi - angle_offset[angle_offset > 0.5*math.pi]
        j_iou = [sq_iou >model.hyp['iou_t'] for sq_iou in square_ious] # iou要区分layer计算
        j_a = angle_offset <  model.hyp['ang_t']  # 角度不分layer都是一样的
        j = [ju * j_a for ju in j_iou] 
        # anchor补漏:没有anchor分配的gt选择最大iou的anchor
        gt_j = torch.stack([juu.reshape(all_ious.shape).max(0)[0] for juu in j],0).t() # (nt,yolo_layer_num)
        num_layers = len(model.yolo_layers)
        for gt_id, gt_ in enumerate(gt_j):
            if not any(gt_):    # 遍历每个多个层都没有anchor的gt
                gt_ious = torch.cat([sq_iou[gt_id::nt]  for sq_iou in square_ious],0)   # 取出所有layer iou展成行进行比较(na*num_layers)
                best_iou_indexes = torch.where(gt_ious==gt_ious.max(0)[0])[0]  # gt_iou内的索引:max 216 = na * num_layer
                layer_id = (best_iou_indexes/na)[0]
                best_ang_indexes = angle_offset[gt_id::nt].repeat(num_layers)[best_iou_indexes].min(0)[1]
                best_iou_indexes = best_iou_indexes[best_ang_indexes]
                j[layer_id][(best_iou_indexes % na) * nt + gt_id] = True

        # mask = torch.stack(j,0)  # (layer_num, nt*72)
        # mask = torch.where(mask,torch.cuda.LongTensor([1]),torch.cuda.LongTensor([-1]))   # 采用-1 0 1 mask

        # anchor masking 只挑出正样本
        assert j[0].sum()+j[1].sum()+j[2].sum() >= nt, 'something wrong at target building'
        for lid, mask_layer in enumerate(j):
            tbox[lid] = tbox[lid][mask_layer]
            tcls[lid] = tcls[lid][mask_layer]
            av[lid]   = av[lid][mask_layer]
            indices[lid][0]  = indices[lid][0][mask_layer]
            indices[lid][1]  = indices[lid][1][mask_layer]
            indices[lid][2]  = indices[lid][2][mask_layer]
            indices[lid][3]  = indices[lid][3][mask_layer]
    # print(len(tbox))  # 分配的正样本个数
    return tcls, tbox, indices, av



# build_targets 完成了两件事：
# - 将target标注缩放放到三个yolo层上
# - 选出各个yolo层上和label box iou较大的anchor将其输出（通过indices），用于回归/计算loss
# targets 是[num_boxes,7](当前batch的)的gt张量
def compute_loss(p, targets, model, hyp):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    # lerg的定位损失包括giou和角度回归两部分
    lcls, liou, lobj, lreg = ft([0]), ft([0]), ft([0]), ft([0])
    
    tcls, tbox, indices, anchor_vec   = build_targets(model, targets, hyp) # 参数含义参见build_targets的说明,参数长度均为3(yolo层的个数)
    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures

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
        BCEcls, BCEobj, BCE, CE, SM,  = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g), FocalLoss(SM, g), 

    ##  类别不平衡采样
    sampling = False
    if sampling:
        sampler = PropSampler(neg_ratio = 3)
        # sampler = GradualSampler(init_ratio = 3, max_epoches = hyp['epochs'])
    else: 
        sampler = NoSampler()
    # Compute losses
    for i, pi in enumerate(p):  # layer index, layer predictions每次遍历一个yolo层的特征  eg. torch.Size([2, 36, 13, 13, 7])
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx 取出当前特征图尺度的所有batch gt的信息
        tobj = torch.zeros_like(pi[..., 0])  # target obj 尺度和yolo层特征图一致,只是去掉channel  eg. torch.Size([2, 36, 13, 13])
        mask = sampler(tobj, indices[i])
        # mask = sampler(tobj, indices[i], model.epoch)
        # Compute losses
        nb = len(b)
        num_anchors = len(a)
        if nb:  # number of targets
            # [b, a, gj, gi]是build_target的结果,是经过筛选合适回归(iou大)的anchor位置(如果有筛选)]
            # ps是对前向pi特征图取出物体中心点负责的grid cell，len和预选留下的anchor一致
            # pi: [bs, na, f_w, f_h, 8]   ps: [num_box , 7]  7 = xywha + obj +classses
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj 因为在这些位置一定有物体,所以gt的obj conf =1,取余gt为0

        # 回归定位：
            ## iou_loss 都是特征图尺寸下的计算
            ng, anchor_vec = model.module_list[model.yolo_layers[i]].ng, model.module_list[model.yolo_layers[i]].anchor_vec
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[a][:,:-1]
            pa  = torch.atan(ps[:,4]) + anchor_vec[a][:,-1]
            pbox = torch.cat((pxy, pwh, pa.unsqueeze(1)) , 1) 

            if pbox.dtype == torch.float16:   # apex适配gt的类型
                if tbox[i].dtype == torch.float16:  pass
                else:  tbox[i] = tbox[i].half()
            liou = h_iou_loss(tbox[i][:,2:4], pbox[:,2:4]).mean()
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
            if mask.sum()!=0:
                lobj += BCEobj(pi[..., 5][mask], tobj[mask])  # obj loss
                
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
