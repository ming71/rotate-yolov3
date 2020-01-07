giou: 0.1       # giou loss gain 1.582
cls: 27.76      # cls loss gain  (CE=~1.0, uCE=~20)
cls_pw: 1   # cls BCELoss positive_weight
obj: 20.      # obj loss gain (*=80 for uBCE with 80 classes)
obj_pw: 1   # obj BCELoss positive_weight, pos_weight > 1 will increase the recall while pos_weight < 1 will increase the precision
iou_t: 0.3      # iou training threshold
ang_t: 3.1415926/6
reg: 1.0
fl_gamma: 0.5   # focal loss gamma
context_factor: 1.0 # 按照短边h来设置的,wh的增幅相同； 调试时设为倒数直接检测


# lr
lr0: 0.0001
multiplier:10
warm_epoch:5
lrf: -4.        # final LambdaLR learning rate = lr0 * (10 ** lrf)
momentum: 0.97  # SGD momentum
weight_decay: 0.0004569  # optimizer weight decay


# aug
hsv_s: 0.5      # image HSV-Saturation augmentation (fraction)
hsv_v: 0.3      # image HSV-Value augmentation (fraction)
degrees: 10.0    # image rotation (+/- deg)
translate: 0.2  # image translation (+/- fraction)
scale: 0.1      # image scale (+/- gain)
shear: 0.0
gamma: 0.0
blur:  1.1
noise: 0.001
contrast: 0.0
sharpen: 0.0
copypaste: 0.0  
grayscale: 0.02 


# training
epochs: 400 
batch_size: 2
test_from: 10
save_interval: 50
test_interval: 1
