import argparse
from sys import platform

from models import *  
from utils.datasets import *
from utils.utils import *


def detect(save_txt=False, save_img=False,hyp=None):
    img_size =  opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, hyp, img_size )  # 搭建模型（不连接计算图），只调用构造函数

    # Load weights
    attempt_download(weights)  
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()


    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()    # pytorch原生支持fp16训练

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)  # source是测试的文件夹路径，返回的dataset是一个迭代器

    
    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])                           # .data文件解析成dict并索引类别名的name文件地址
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]  # 配置颜色

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:    # im0s为原图(hwc)，im0s为缩放+padding之后的图(chw)
        t = time.time()
        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:   # 查看数据维度是否为三维，等价于len(img.shape)
            img = img.unsqueeze(0)  # 加个第0维bs，但是detect实际没用

        # 只用到io的结果，不用p；io有三个维度：bs(1),num_proposal(每个yolo层预测其特征图的w*h*3个proposal),num_params(5+classes)
        pred, _ = model(img)        # forward 
        # NMS后返回的张量维度：[(num_detections,7),...] (7=(x1, y1, x2, y2, object_conf, class_conf, class))  (len=bs)
        # 遍历时det是每张图片的bbox属性： (num_detections,7)
        # import ipdb; ipdb.set_trace()
        for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
            # import ipdb; ipdb.set_trace()
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)

            # s 是最后检测打印的字符串，会通过字符串拼接逐渐添加项
            s += '%gx%g ' % img.shape[2:]  # s添加缩放后的图像尺度，如  320x416 
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size  将预测的bbox坐标(前四维)从缩放图放大回原图尺度
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results： 统计各类物体出现的次数
                for c in det[:, -1].unique():       # 取出最后一维类别并去重排序
                    n = (det[:, -1] == c).sum()     # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # s添加检测物体统计

                # Write results
                for *box, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 7 + '\n') % (*box, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(box, im0, label=label, color=colors[int(cls)])

            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/voc.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/all', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.py', help='hyper-parameter path')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    hyp = hyp_parse(opt.hyp)

    with torch.no_grad():
        detect(hyp=hyp)
