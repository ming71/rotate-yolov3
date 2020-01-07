import torch.nn.functional as F

from utils.google_utils import *
from utils.parse_config import *
from utils.utils import *



def get_yolo_layers(model):
    # return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3
    return [i for i, x in enumerate(model.module_defs) if x['type'] in ['detection','yolo']]  # [82, 94, 106] for yolov3



# 做了两件事：
    # - 编码grid cell的坐标
    # - 将anchor缩放到特征图尺度（后面在特征图上进行预测）
def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size # ng是传入的特征图宽高tuple
    # 计算降采样步长self.stride 32/16/8
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)   

    # build xy offsets
    # 最终结果self.grid_xy的维度为torch.Size([1, 1, 10, 13, 2])，其中10和13的维度对应的是特征图的每个点，最后的2是其上的编号
    # 如特征图为10*13,则构建的偏移阵列从[0,0]，[1,0]...[12,0],   [0,1].[1,1]...[12,1], ...[12,9]
    # 表示的是特征图每个像素点的位置，也就是原图的grid左上角坐标，和后面预测的cell内偏移共同表示最终预测的物体位置
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device)
    self.anchor_vec[:,:2] /= self.stride    #  只转化wh,角度不变
    self.anchor_wha = self.anchor_vec.view(1, self.na, 1, 1, 3).to(device).type(type)    # torch.Size([1, 18, 1, 1, 3])
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv_layer = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')

# 如果weights指定的权重不存在，则下载；存在则该函数不返回直接pass
def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    msg = weights + ' missing, download from https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI'
    if weights and not os.path.isfile(weights): # 指定路径的权值文件不存在
        file = Path(weights).name   # 分割路径文件名

        if file == 'yolov3-spp.weights':
            gdrive_download(id='1oPCHKsM2JpM-zgyepQciGli9X0MTsJCO', name=weights)
        elif file == 'yolov3-spp.pt':
            gdrive_download(id='1vFlbJ_dXPvtwaLLOu-twnjK4exdFiQ73', name=weights)
        elif file == 'yolov3.pt':
            gdrive_download(id='11uy0ybbOXA2hc-NJkJbbbkDwNX1QZDlz', name=weights)
        elif file == 'yolov3-tiny.pt':
            gdrive_download(id='1qKSgejNeNczgNNiCn9ZF_o55GFk1DjY_', name=weights)
        elif file == 'darknet53.conv.74':
            gdrive_download(id='18xqvs_uwAqfTXp-LJCYLYNHBOcrwbrp0', name=weights)
        elif file == 'yolov3-tiny.conv.15':
            gdrive_download(id='140PnSedCsGGgu3rOD6Ez4oI6cdDzerLC', name=weights)

        else:
            try:  # download from pjreddie.com
                url = 'https://pjreddie.com/media/files/' + file
                print('Downloading ' + url)
                os.system('curl -f ' + url + ' -o ' + weights)
            except IOError:
                print(msg)
                os.system('rm ' + weights)  # remove partial downloads

        assert os.path.exists(weights), msg  # download missing weights from Google Drive

    