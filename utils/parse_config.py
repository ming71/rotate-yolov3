import numpy as np
import math



def cfg2anchors(val):
    if 'ara' in val:   # area, ratio, angle respectly
        val = val[val.index('ara')+3:]
        val = [i for i in val.split('/')  if len(i)!=0]  # ['12130, 42951, 113378 ', ' 4.18, 6.50, 8.75 ', '-60,-30,0,30,60,90']
        areas = [float(i) for i in val[0].split(',')]
        ratios = [float(i) for i in val[1].split(',')]  # w/h
        angles = [float(i) for i in val[2].split(',')]   
        anchors = []
        for area in areas:
            for ratio in ratios:
                for angle in angles:
                    anchor_w = math.sqrt(area*ratio)
                    anchor_h = math.sqrt(area/ratio)
                    angle = angle*math.pi/180
                    anchor   = [anchor_w, anchor_h, angle]
                    anchors.append(anchor)
        assert len(anchors) == len(areas)*len(ratios)*len(angles),'Something wrong in anchor settings.'
        # print(np.array(anchors))
        return np.array(anchors)
    else:    # anchors generated via k-means, input anchor.txt 
        # 默认是15度一个anchor
        anchors_setting = val.strip(' ')
        anchors = np.loadtxt(anchors_setting)
        angle = np.array([i for i in range(-6,6)])*math.pi/12
        anchors = np.concatenate([np.column_stack((np.expand_dims(i,0).repeat(len(angle),0),angle.T)) for i in anchors],0)
        return anchors


# cfg解析函数：
#   将cfg的layer，setting等解析成dict的形式，返回一个包含这些dict的list；
#   lsit的每个元素（dict）对应cfg文件的一个 [] 开头的block（如net等），第一个元素就是该block的性质如{'type': 'net'...}
def parse_model_cfg(path):
    # Parses the yolo-v3 layer configuration file and returns module definitions
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if 'anchors' in key:
                mdefs[-1][key] =  cfg2anchors(val) # np anchors
            else:
                mdefs[-1][key] = val.strip()

    return mdefs

# 像mmdetection一样，将配置文件转码return成dict的键值对形式，便于索引查询
def parse_data_cfg(path):
    # Parses the data configuration file
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
