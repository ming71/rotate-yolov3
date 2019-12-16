import torch 
from utils.nms.r_nms import r_nms

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

        if prediction.numel() == 0: # for multi-scale filtered result , in case of 0
            continue

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
        # use_cuda时方案是不限于100个，因为有可能产生很多的高得分proposal，会误删
        if  use_cuda_nms:
            det_max = []
            pred = torch.cat((pred[:, :6], class_conf.unsqueeze(1), class_pred), 1)
            pred = pred[(-pred[:, 5]).argsort()]
            for c in pred[:, -1].unique():
                dc = pred[pred[:, -1] == c]
                dc = dc[(-dc[:, 5]).argsort()]
                # if len(dc)>100:   # 如果proposal实在太多，取100个
                #     dc = dc[:100]

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


