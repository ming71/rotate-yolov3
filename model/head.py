
import math
import torch
import torch.nn as nn

from model.model_utils import *

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, arc, hyp, yolo_index): 
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 6  # number of outputs
        self.nx = 0  # initialize number of x gridpoints    
        self.ny = 0  # initialize number of y gridpoints
        self.arc = arc
        self.hyp = hyp
        self.yolo_index = yolo_index  # idx: 0 1 2 ...

    def forward(self, p, img_size, var=None):   # p是特征图，img_size是缩放并padding后的尺寸如torch.Size([320, 416])（用来确定原图和特征图的对应位置）
        if p.dim() != 5:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13   # ny nx是特征图的高和宽
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)   # 缩放anchor到特征图尺寸;用特征图像素编码grid cell

            # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
            p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction
        else:
            bs, ny, nx = p.shape[0], p.shape[-3], p.shape[-2]   # ny nx是特征图的高和宽
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)        

        # self继承自nn.Module，其自带属性self.training且默认为True，但是在model.eval()会被设置成False
        if self.training:
            # 如果是training,直接返回yolo fp (bs, anchors, grid, grid, classes + xywh)
            return p
        
        else:   # inference   # 不止返回inference结果还有train的
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy ：预测的偏移 + grid cell id
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wha[...,:-1]    # wh yolo method （加exp化为正数）；wh预测的是一个比例，基准是anchor
            io[..., 4]   = torch.atan(io[..., 4]) + self.anchor_wha[...,-1]
            # 从特征图放大到原图尺寸
            io[..., :4] *= self.stride
            # 整体缩放法
            # io[..., 2:4] /= self.hyp['context_factor']    
            # 取h短边合理缩放
            io[..., 3] /= self.hyp['context_factor']
            io[..., 2] -= io[..., 3]*(self.hyp['context_factor']-1)

            if 'default' in self.arc:  # seperate obj and cls
                # 将obj得分和各类别的得分进行sigmoid处理
                torch.sigmoid_(io[..., 5:])     # in-place操作，慎用
            elif 'BCE' in self.arc:  # unified BCE (80 classes)
                torch.sigmoid_(io[..., 6:])
                io[..., 5] = 1
            elif 'CE' in self.arc:  # unified CE (1 background + 80 classes)
                io[..., 5:] = F.softmax(io[..., 4:], dim=4)
                io[..., 5] = 1

            if self.nc == 1:
                io[..., 6] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # 注意：yolo层返回两个张量
            #   - 一个是三个维度的(分类和置信度得分归一化了)         [1, 507, 85]
            #   - 一个是输入reshape分分离出不同类别得分而已    [1, 3, 13, 13, 85]
            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 6 + self.nc), p



'''
class CLSHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_anchors,
                 num_classes,
                 nO=None):
        super(CLSHead, self).__init__()
        self.na = num_anchors
        self.num_classes = num_classes
        self.convs = nn.Sequential(
                        nn.Conv2d(in_channels, feat_channels, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(feat_channels, momentum=0.1),
                        nn.Conv2d(feat_channels, in_channels, 3, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(in_channels, momentum=0.1)
                        )
        self.head = nn.Conv2d(in_channels, num_anchors * (num_classes+1), 1, 1)  
        self.init_weights()   

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 1.113e-5
        self.head.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self, x):
        x = self.convs(x)
        x = self.head(x)
        bs, c, h, w = x.shape
        return x.view(bs, self.na, (self.num_classes+1), h, w).permute(0, 1, 3, 4, 2).contiguous()  # conf + cls
'''

class CLSHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_anchors,
                 num_classes,
                 nO=None):
        super(CLSHead, self).__init__()
        self.na = num_anchors
        self.nO = nO
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, feat_channels, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(feat_channels, momentum=0.1),
                        nn.Conv2d(feat_channels, in_channels, 3, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(in_channels, momentum=0.1)
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels//nO, feat_channels, 3, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(feat_channels, momentum=0.1),
                        ) 
        self.head = nn.Conv2d(feat_channels, num_anchors * (num_classes+1), 1, 1)
        self.init_weights()   

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 1.113e-5
        self.head.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self, x):
        x = self.conv1(x)
        _bs, _ch, _ny, _nx = x.shape
        O_ch = _ch // self.nO
        x = x.reshape(_bs, O_ch, self.nO, _ny, _nx)
        x = torch.max(x, 2)[0]
        x = self.conv2(x)
        x = self.head(x)
        bs, c, h, w = x.shape
        return x.view(bs, self.na, (self.num_classes+1), h, w).permute(0, 1, 3, 4, 2).contiguous()  # conf + cls



class REGHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_anchors,
                 num_regress):
        super(REGHead, self).__init__()
        self.na = num_anchors
        self.num_regress = num_regress
        self.convs = nn.Sequential(
                        nn.Conv2d(in_channels, feat_channels, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(feat_channels, momentum=0.1),
                        nn.Conv2d(feat_channels, in_channels, 3, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(in_channels, momentum=0.1)
                        )
        self.head = nn.Conv2d(in_channels, num_anchors*num_regress, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.head.weight, gain=0.1)
        torch.nn.init.normal_(self.head.bias, mean=0, std=0.001)

    def forward(self, x):
        x = self.convs(x)
        x = self.head(x)
        bs, c, h, w = x.shape
        return x.view(bs, self.na, self.num_regress, h, w).permute(0, 1, 3, 4, 2).contiguous()



class DualHead(nn.Module):
    def __init__(self,
                 in_channels,
                 cls_feat_channels,
                 reg_feat_channels,
                 num_classes,
                 num_anchors,
                 num_regress,   # x y w h a conf
                 refine=None, 
                 nO = None
                 ):
        super(DualHead, self).__init__()
        self.refine = refine
        self.cls_head = CLSHead(in_channels,cls_feat_channels,num_anchors,num_classes,nO=nO) 
        self.reg_head = REGHead(in_channels,reg_feat_channels,num_anchors,num_regress)


    def forward(self, x):   # (bs, c, h, w)
        cls_res = self.cls_head(x)  # (bs, na, ny, nx, nc+1) conf + nc
        reg_res = self.reg_head(x)  # (bs, na, ny, nx, 5) 6= x y w h a 
        p = torch.cat([reg_res, cls_res], -1)
        return p



