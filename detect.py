import argparse
from sys import platform
import time
import os
import glob

from model.models import Darknet
from model.model_utils import attempt_download,parse_data_cfg
from utils.datasets import LoadImages
from utils.utils import *
from utils.datasets import letterbox
from utils.nms.nms import non_max_suppression
from utils.ICDAR.icdar_utils import xywha2icdar, zip_dir

# 我自己写的还快些,FPS更高,性能没有下降
def multi_detect(save_txt=True, save_img=True, hyp=None, multi_scale = False):
    img_size =  opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    
    # Setting txt result folder
    save_type = 'ICDAR'
    if save_txt and save_type == 'ICDAR':
        out_txt = 'icdar_result'
        if os.path.exists(out_txt):
            shutil.rmtree(out_txt)  # delete output folder
        os.makedirs(out_txt)  # make new output folder

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])                           # .data文件解析成dict并索引类别名的name文件地址
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]  # 配置颜色

    #  build model
    model = Darknet(opt.cfg, hyp)  # 搭建模型（不连接计算图），只调用构造函数
        ## Load weights
    attempt_download(weights)  
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        ## Eval mode
    model.to(device).eval()
    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()    # pytorch原生支持fp16训练

    # multi-scale
    if multi_scale:
        size_num = 2
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.3) - 1
        img_sizes = np.random.choice(range(img_sz_min * 32, (img_sz_max+1) * 32, 32), size_num, False)
        img_sizes = [int(x) for x in img_sizes]
        # img_sizes = [608]
        print('use scale: {}'.format(img_sizes))
    else:
        img_sizes = [img_size]



    # load images
    t0 = time.time()
    img_paths = sorted(glob.glob(os.path.join(source, '*.jpg')))
    for id, img_path in enumerate(img_paths):
        img0 = cv2.imread(img_path)
        p, s,  = img_path, ''
        save_path = str(Path(out) / Path(p).name)
        assert img0 is not None, 'Image Not Found ' + path
        print('image %g/%g %s: ' % (id, len(img_paths), img_path), end='')
        # multi-scale inference
        all_pre = []
        t = time.time()
        for img_size in img_sizes:
            img, *_ = letterbox(img0, new_shape=img_size)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float16 if opt.half else np.float32)  # uint8 to fp16/fp32
            img /= 255.0  
            
            #  Get detections
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:   # 查看数据维度是否为三维，等价于len(img.shape)
                img = img.unsqueeze(0)  # 加个第0维bs，但是detect实际没用
            pred, _ = model(img)        # forward 
            # nms
            det = non_max_suppression(pred, opt.conf_thres, 0.95)[0]   # bs只允许1,不支持视频
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size  将预测的bbox坐标(前四维)从缩放图放大回原图尺度
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            all_pre.append(det)
        all_pre = list(filter(is_None,all_pre))
        if all_pre is not None and len(all_pre):
            merged_dets = torch.cat(all_pre, 0)
            final_dets = non_max_suppression(merged_dets.unsqueeze(0), opt.conf_thres, opt.nms_thres)[0]
            # Print results： 统计各类物体出现的次数
            if final_dets is not None and len(final_dets):
                for c in final_dets[:, -1].unique():       # 取出最后一维类别并去重排序
                    n = (final_dets[:, -1] == c).sum()     # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # s添加检测物体统计

                # Write results
                for *box, conf, _, cls in final_dets:
                    if save_txt and save_type == 'ICDAR':  # Write to file
                        save_icdar_path = str(Path(out_txt)/ ('res_'+os.path.splitext(Path(p).name)[0]+'.txt'))
                        with open(save_icdar_path , 'a') as file:
                            file.write(xywha2icdar(box))
                    elif save_txt:
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 7 + '\n') % (*box, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(box, img0, label=label, color=colors[int(cls)])
        print('%sDone. (%.3fs)' % (s, time.time() - t))

        # Stream results
        if view_img:
            cv2.imshow(p, img0)

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, img0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    zip_dir(out_txt, 'icdar_result.zip')
    shutil.rmtree(out_txt)




def detect(save_txt = True, save_img = False, hyp = None):
    img_size =  opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    
    # Setting txt result folder
    save_type = 'ICDAR'
    if save_txt and save_type == 'ICDAR':
        out_txt = 'detections'
        if os.path.exists(out_txt):
            shutil.rmtree(out_txt)  # delete output folder
        os.makedirs(out_txt)  # make new output folder

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, hyp )  # 搭建模型（不连接计算图），只调用构造函数

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
        model.half()    # pytorch原生支持fp16

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
        # 实际上在图像中遍历只会执行一次,这里的i=0就跳出了
        for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
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
                    if save_txt and save_type == 'ICDAR':  # Write to file
                        save_icdar_path = str(Path(out_txt)/ ('res_'+os.path.splitext(Path(p).name)[0]+'.txt'))
                        with open(save_icdar_path , 'a') as file:
                            file.write(xywha2icdar(box))
                    elif save_txt:
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 7 + '\n') % (*box, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(box, im0, label=label, color=colors[int(cls)])
                        # plot_one_box(box, im0, label=label, color=[0,0,255], line_thickness=1)
                        

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
    zip_dir(out_txt, 'icdar_result.zip')
    shutil.rmtree(out_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='cfg/ICDAR/hyp.py', help='hyper-parameter path')
    parser.add_argument('--cfg', type=str, default='cfg/ICDAR/yolov3_608_dh_o8_ga.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/tiny.data', help='*.data file path')
    # parser.add_argument('--data', type=str, default='data/icdar13+15.data', help='coco.data file path')
    # parser.add_argument('--data', type=str, default='data/single.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/tiny/test', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--multi-scale', action='store_true', help='multi-scale testing')
    opt = parser.parse_args()
    print(opt)

    hyp = hyp_parse(opt.hyp)

    with torch.no_grad():
        if opt.multi_scale:
            multi_detect(hyp=hyp, multi_scale = True)
        else:
            # multi_detect(hyp=hyp)
            detect(hyp=hyp)
