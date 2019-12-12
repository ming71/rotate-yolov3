import numpy as np
import torch
import cv2
import math

import r_nms


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

if __name__ == '__main__':
    boxes = np.array([[150, 150, 100, 100, 0,       0.99,   0.1],
                      [160, 160, 100, 100, 0,       0.88,   0.1],
                      [150, 150, 100, 100, -0.7854, 0.66,   0.1],
                      [300, 300, 100, 100, 0.,      0.77,   0.1]],dtype=np.float32)
    
    dets_th=torch.from_numpy(boxes).cuda()
    import ipdb; ipdb.set_trace()
    iou_thr = 0.1
    inds = r_nms.r_nms(dets_th, iou_thr)
    print(inds)

    img = np.zeros((416*2,416*2,3), np.uint8)
    img.fill(255)
    
    boxes = boxes[:,:-1]
    boxes = [get_rotated_coors(i).reshape(-1,2).astype(np.int32)  for i in boxes]
    for box in boxes:
        img = cv2.polylines(img,[box],True,(0,0,255),1)
        cv2.imshow('anchor_show', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()