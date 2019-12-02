import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread
import math


import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.augment import *
import imgaug.augmenters as iaa

from utils.utils import xyxy2xywh, xywh2xyxy, get_rotated_coors

# 支持的图片和视频类型
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif'] 
vid_formats = ['.mov', '.avi', '.mp4']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

# 初始化后是一个迭代器
# 进行遍历时，先执行__iter__返回一个迭代器对象，然后一直执行__next__直到抛出StopIteration()异常
class LoadImages:  # for inference
    def __init__(self, path, img_size=416, half=False):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))    # 返回路径下所有的文件绝对路径
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]   # 筛选出支持的img_formats图片或者vid_formats视频类型
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)   

        self.img_size = img_size    
        self.files = images + videos    # 所有待检测文件的list，采用images+viedos确保list的前面是图像后面是视频！
        self.nF = nI + nV               # number of files  待检测的文件个数
        self.video_flag = [False] * nI + [True] * nV    # len为文件个数，和上面的file list一一对应，eg.[False, False, False, False, False, True, True]说明有两个视频
        self.mode = 'images'
        self.half = half  # half precision fp16 images
        if any(videos):     # any()全为False才返回False；也就是有video文件时执行下面的语句（除了是 0、空、FALSE 外都算 TRUE）
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    # __iter__方法决定该类是 可迭代类 。只循环一次，每次新的循环都要创建一个新的迭代器
    def __iter__(self):
        self.count = 0
        return self

    # __iter__方法决定该类是 迭代器类 。
    def __next__(self):
        if self.count == self.nF:
            raise StopIteration         # 抛出StopIteration异常终止__next__方法迭代
        path = self.files[self.count]   # 每次遍历取出self.files的一个文件

        if self.video_flag[self.count]: # 判断当前文件是否为video
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read() # ret_val是读取标识位，正确打开视频返回True
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')


        # Padded resize 
        # padding成长边416,短边32倍数，不足部分填充灰色的图像
        img, *_ = letterbox(img0, new_shape=self.img_size)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=416, half=False):
        self.img_size = img_size
        self.half = half  # half precision fp16 images

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img, *_ = letterbox(img0, new_shape=self.img_size)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=416, half=False):
        self.mode = 'images'
        self.img_size = img_size
        self.half = half  # half precision fp16 images

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(0 if s == '0' else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        while cap.isOpened():
            _, self.imgs[index] = cap.read()
        time.sleep(0.030)  # 33.3 FPS to keep buffer empty

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Normalize RGB
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

# 注意继承自Dataset类,才能直接使用dataloader
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_labels=False, cache_images=False):
        path = str(Path(path))  # os-agnostic  train.txt绝对路径
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic 获取img文件的绝对路径list
                              if os.path.splitext(x)[-1].lower() in img_formats]

        n = len(self.img_files)     # 图片个数
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index,将n张图片按照bs大小得到batch数进行图片的batch编号( len(bi)=n )
        nb = bi[-1] + 1  # number of batches 总batch数
        assert n > 0, 'No images found in %s' % path

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]    # 获取label txt文件的绝对路径list

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        # 和inference一样,支持非squre训练(最好同步)
        if self.rect:
            # Read image shapes
            sp = 'data' + os.sep + path.replace('.txt', '.shapes').split(os.sep)[-1]  # shapefile path
            try:
                with open(sp, 'r') as f:  # read existing shapefile
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == n, 'Shapefile out of sync'
            except:
                s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
                np.savetxt(sp, s, fmt='%g')  # overwrites existing (if any)

            # Sort by aspect ratio
            s = np.array(s, dtype=np.float64)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.shapes = s[i]
            ar = ar[i]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [None] * n
        if cache_labels or image_weights:  # cache labels for faster training
            self.labels = [np.zeros((0, 5))] * n
            extract_bounding_boxes = False
            create_datasubset = False
            pbar = tqdm(self.label_files, desc='Reading labels')    # tqdm对象,后面加载label时可以生成进度
            nm, nf, ne, ns = 0, 0, 0, 0  # number missing, number found, number empty, number datasubset
            for i, file in enumerate(pbar):
                try:
                    with open(file, 'r') as f:
                        # l是当前label file的info矩阵, shape:(num_box , 5) # 5 = cxywh
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    continue

                if l.shape[0]:
                    # 对不标准的label抛出异常: 1.info长度不对  2.负样本  3.未归一化(yolo format)
                    assert l.shape[1] == 6, '> 6 label columns: %s' % file
                    assert (l[:, 1:-1] >= 0).all(), 'negative labels: %s' % file
                    assert (l[:, 1:-1] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    assert (l[:, 5] < math.pi/2).all() and (l[:, 5] > -math.pi/2).all(), 'out of angle bounds (-0.5pi,0.5pi)'
                    self.labels[i] = l
                    nf += 1  # file found

                    # Create subdataset (a smaller dataset)
                    if create_datasubset and ns < 1E4:
                        if ns == 0:
                            create_folder(path='./datasubset')
                            os.makedirs('./datasubset/images')
                        exclude_classes = 43
                        if exclude_classes not in l[:, 0]:
                            ns += 1
                            # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                            with open('./datasubset/images.txt', 'a') as f:
                                f.write(self.img_files[i] + '\n')

                    # Extract object detection boxes for a second stage classifier 裁剪出bbox一般没啥用
                    if extract_bounding_boxes:
                        p = Path(self.img_files[i])
                        img = cv2.imread(str(p))
                        h, w, _ = img.shape
                        for j, x in enumerate(l):
                            f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)  # make new output folder
                            box = xywh2xyxy(x[1:].reshape(-1, 4) * np.array([1, 1, 1.5, 1.5])).ravel()
                            b = np.clip(box, 0, 1)  # clip boxes outside of image
                            ret_val = cv2.imwrite(f, img[int(b[1] * h):int(b[3] * h), int(b[0] * w):int(b[2] * w)])
                            assert ret_val, 'Failure extracting classifier boxes'
                else:
                    ne += 1  # file empty

                pbar.desc = 'Reading labels (%g found, %g missing, %g empty for %g images)' % (nf, nm, ne, n)
            assert nf > 0, 'No labels found. Recommend correcting image and label paths.'

        # Cache images into memory for faster training (~5GB)
        if cache_images and augment:  # if training
            for i in tqdm(range(min(len(self.img_files), 10000)), desc='Reading images'):  # max 10k images
                img_path = self.img_files[i]
                img = cv2.imread(img_path)  # BGR
                assert img is not None, 'Image Not Found ' + img_path
                r = self.img_size / max(img.shape)  # size ratio
                if self.augment and r < 1:  # if training (NOT testing), downsize to inference shape
                    h, w, _ = img.shape
                    img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # or INTER_AREA
                self.imgs[i] = img

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    '''
    详细展开这个__getitem__类:
    python的迭代执行对象有两种:迭代器对象和可迭代对象；
    前者需要封装好__iter__和__next__方法；后者可以没有__next__,可迭代对象执行迭代时,python会自行创建一个迭代器；
    而有的迭代器对象甚至没有__iter__方法,但是集成了__getitem__仍可迭代,其迭代的索引index从0开始
    (另: __getitem__除此之外也用作类实例化的key取值)

    为什么要定义?
    继承自dataset类便于使用后面的dataloader读取数据,需要遵循dataset的规范,实现三个方法:
    -     __init__     构造函数,初始化路径配置等参数
    -  (*)__getitem__  处理部分:从文件读数据/处理数据/返回数据(eg.img+label等)  --> 数据预处理如增强等是在这里完成的  
    -     __len__      返回长度
    所以在遍历dataloader的时候遍历的key得到的参数是已经经过该迭代类的__getitem__方法处理并return的结果,如train的时候,for遍历得到的是:
    for imgs, targets, paths, _ in dataloader,这些参数来自__getitem__的return值
    '''
    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        img_path = self.img_files[index]
        label_path = self.label_files[index]
        hyp = self.hyp

        # Load image
        img = self.imgs[index]
        if img is None:
            img = cv2.imread(img_path)  # BGR
            assert img is not None, 'Image Not Found ' + img_path
            r = self.img_size / max(img.shape)  # size ratio  416/图像的长边
            if self.augment and r < 1:  # if training (NOT testing), downsize to inference shape
                h, w, _ = img.shape
                # img按长边缩放到416,shape [h,w,c]
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR fastest



        # Letterbox
        h, w, _ = img.shape
        if self.rect:   
            shape = self.batch_shapes[self.batch[index]]
            img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='rect')
        else:
            shape = self.img_size
            img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='square')

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            x = self.labels[index]  # x就是当前img的box参数,二维矩阵
            if x is None:  # labels not preloaded 一般预加载比较好,可以省区反复读取数据的时间
                with open(label_path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

            if x.size > 0:
                # 不只是简单的回到原图，还进行了padding，使得label和416的padding  img可以适配
                labels = x.copy()   # c x y w h a
                labels[:, 1] = ratiow * w * labels[:, 1] + padw
                labels[:, 2] = ratioh * h * labels[:, 2] + padh
                labels[:, 3] = ratiow * w * labels[:, 3]
                labels[:, 4] = ratioh * h * labels[:, 4]

        # 数据增强
        if self.augment:  
            transform = Transform([Affine( hyp['degrees'] ,hyp['translate'],hyp['scale'],hyp['shear'], p = 0.5),
                                  Contrast(hyp['contrast'], p = 0.3),
                                  Sharpen(hyp['contrast'], p = 0.2),
                                  Noise(hyp['noise'], p = 0.2),
                                  Gamma(hyp['gamma'], p = 0.4),
                                  Blur(hyp['blur'], p = 0.5),
                                  HSV(hyp['hsv_s'],hyp['hsv_v'], p = 0.5),
                                  HorizontalFlip(p = 0.5),
                                  VerticalFlip(p = 0.5),
                                  CopyPaste(sigma = hyp['copypaste'], p=0.1)
                                  ])
            img, labels = transform(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width


        labels_out = torch.zeros((nL, 7))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw


def letterbox(img, new_shape=416, color=(128, 128, 128), mode='square'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    #计算按照长边保持比例缩放到height(416)的新shape列表：[new_height, new_width
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))    

    if mode is 'auto':      # minimum rectangle    在原图缩放到长边416后，短边padding到32的整数倍以确保得到的特征图像素是整数
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding实际padding的像素是dh,dw*2，dh代表两边各padding dh
    elif mode is 'square':  # square    padding到416*416   
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':    # square    padding到416*416
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':   # padding到416*416,但是改变了原来的宽高比例
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    if shape[::-1] != new_unpad:  # resize
        # 长边缩放到416
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))    # 根据padding计算上下左右的填充量
    # 填充color为灰色(128, 128, 128)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratiow, ratioh, dw, dh



def convert_images2bmp():
    # cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
    for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
        folder = os.sep + Path(path).name
        output = path.replace(folder, folder + 'bmp')
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        for f in tqdm(glob.glob('%s*.jpg' % path)):
            save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
            cv2.imwrite(save_name, cv2.imread(f))

    for label_path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
        with open(label_path, 'r') as file:
            lines = file.read()
        lines = lines.replace('2014/', '2014bmp/').replace('.jpg', '.bmp').replace(
            '/Users/glennjocher/PycharmProjects/', '../')
        with open(label_path.replace('5k', '5k_bmp'), 'w') as file:
            file.write(lines)


def create_folder(path='./new_folder'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def point_rotate(angle,x,y,centerx,centery,vis=False):
    x = np.array(x)
    y = np.array(y)
    nRotatex = (x-centerx)*math.cos(angle) - (y-centery)*math.sin(angle) + centerx
    nRotatey = (x-centerx)*math.sin(angle) + (y-centery)*math.cos(angle) + centery
    
    if vis:
        plt.plot([centerx,x],[centery,y])
        plt.plot([centerx,nRotatex],[centery,nRotatey])
        plt.show()
    return np.stack((nRotatex,nRotatey),-1)