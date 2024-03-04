import torch
from torch import nn,optim
import torch.nn.functional as F
import glob
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pycocotools import coco
import matplotlib.pyplot as plt 
import numpy as np
import time 
from torchvision.transforms import v2
import sys
import evaluation
import os
import csv
from backbones_unet.model.unet import Unet
from datetime import datetime
from skimage.morphology import binary_dilation, disk
from numpy.random import default_rng


def sep_noevent_data(data_noevent):

    temp_list = []
    data_nocme = []

    ev_id_prev = 0
    
    for i, noev in enumerate(data_noevent):

        if (noev - ev_id_prev == 1) or (ev_id_prev == 0):

            temp_list.append(i)

        elif (noev - ev_id_prev) > 1:

            if len(temp_list) > 0:
                data_nocme.append(temp_list)

            temp_list = []
    
        ev_id_prev = data_noevent[i]
    
    return data_nocme

class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2*in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out
    
def backward_warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default for PyTorch version < 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output
    
class FrondandDiff(Dataset):
    def __init__(self, transform=None):
        
        rng = default_rng()

        self.transform = transform

        self.coco_obj = coco.COCO("instances_clahe.json")
        

        self.img_ids = self.coco_obj.getImgIds()[:3968]###until last image annotated

        self.annotated = []
        self.events = []

        event_list = []
        temp_list = []

        a_id_prev = 0

        for a in range(0, len(self.img_ids)):
            anns = self.coco_obj.getAnnIds(imgIds=[self.img_ids[a]])

            if(len(anns)>0):

                self.annotated.append(a)

                if (self.img_ids[a] - a_id_prev == 1) or (a_id_prev == 0):

                    temp_list.append(a)

                elif (self.img_ids[a] - a_id_prev) > 1:

                    if len(temp_list) > 0:
                        event_list.append(temp_list)

                    temp_list = []
         
                a_id_prev = self.img_ids[a]

        
        self.events = event_list

        len_train = int(len(self.annotated)*0.7)
        boundary_train = self.annotated[len_train]

        for index, ev in enumerate(self.events):
            if boundary_train in ev:
                index_train = index + 1
                break
            else:
                continue

        train_data = self.events[:index_train]

        len_val = int(len(self.annotated)*0.1)
        boundary_val = self.annotated[len_train+len_val]

        for index, ev in enumerate(self.events):
            if boundary_val in ev:
                index_val = index + 1
                break
            else:
                continue
        
        val_data = self.events[index_train:index_val]

        test_data = self.events[index_val:]

        train_data_cme = []

        for sublist in train_data:
            train_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,2))
        
        train_data_cme = np.array([element for innerList in train_data_cme for element in innerList])

        val_data_cme = []

        for sublist in val_data:
            val_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,2))

        val_data_cme = np.array([element for innerList in val_data_cme for element in innerList])

        test_data_cme = []

        for sublist in test_data:
            test_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,2))

        test_data_cme = np.array([element for innerList in test_data_cme for element in innerList])

        train_data = [element for innerList in train_data for element in innerList]
        train_data = np.array(train_data)

        val_data = [element for innerList in val_data for element in innerList]
        val_data = np.array(val_data)

        test_data = [element for innerList in test_data for element in innerList]
        test_data = np.array(test_data)

        train_data_noevent = np.arange(np.nanmin(train_data), np.nanmin(val_data))
        train_data_noevent = np.setdiff1d(train_data_noevent, train_data) 

        val_data_noevent = np.arange(np.nanmin(val_data), np.nanmin(test_data))
        val_data_noevent = np.setdiff1d(val_data_noevent, val_data)

        test_data_noevent = np.arange(np.nanmin(test_data), np.nanmax(self.img_ids))
        test_data_noevent = np.setdiff1d(test_data_noevent, test_data)
        
        train_data_noevent = sep_noevent_data(train_data_noevent)
        val_data_noevent = sep_noevent_data(val_data_noevent)
        test_data_noevent = sep_noevent_data(test_data_noevent)

        win_size = 2

        train_data_nocme = []

        for sublist in train_data_noevent:
            train_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        train_data_nocme = np.array([element for innerList in train_data_nocme for element in innerList])

        val_data_nocme = []

        for sublist in val_data_noevent:
            val_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        val_data_nocme = np.array([element for innerList in val_data_nocme for element in innerList])

        test_data_nocme = []

        for sublist in test_data_noevent:
            test_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        test_data_nocme = np.array([element for innerList in test_data_nocme for element in innerList])

        not_annotated = np.array(np.setdiff1d(np.arange(0,len(self.img_ids)),np.array(self.annotated)))
        self.not_annotated = not_annotated

        seed = 1997
        np.random.seed(seed)

        train_data = np.concatenate([train_data,np.random.choice(self.not_annotated[(self.not_annotated < boundary_train) & (self.not_annotated > train_data[0])], len(train_data), replace=False)])
        val_data = np.concatenate([val_data,np.random.choice(self.not_annotated[(self.not_annotated < boundary_val) & (self.not_annotated > val_data[0])], len(val_data), replace=False)])
        test_data = np.concatenate([test_data,np.random.choice(self.not_annotated[(self.not_annotated > boundary_val) & (self.not_annotated < test_data[-1])], len(test_data), replace=False)])
        
        train_data_paired = np.concatenate([train_data_cme, train_data_nocme[rng.choice(len(train_data_nocme), size=len(train_data_cme), replace=False)]])
        val_data_paired = np.concatenate([val_data_cme, val_data_nocme[rng.choice(len(val_data_nocme), size=len(val_data_cme), replace=False)]])
        test_data_paired = np.concatenate([test_data_cme, test_data_nocme[rng.choice(len(test_data_nocme), size=len(test_data_cme), replace=False)]])

        self.train_data_idx = sorted(train_data)
        self.val_data_idx = sorted(val_data)
        self.test_data_idx = sorted(test_data)

        self.train_paired_idx = train_data_paired
        self.val_paired_idx = val_data_paired
        self.test_paired_idx = test_data_paired

        all_anns = list(self.annotated) + list(self.not_annotated)
        all_anns = sorted(all_anns)

        self.all_anns = all_anns

    def get_img_and_annotation(self,idx):

        img_id   = self.img_ids[idx]
        img_info = self.coco_obj.loadImgs([img_id])[0]
        img_file_name = img_info["file_name"].split('/')[-1]

        if torch.cuda.is_available():
            
            if os.path.isdir('/home/mbauer/Data/'):
                path = "/home/mbauer/Data/differences_clahe/"
                width_par = 512
            
            elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
                path = "/gpfs/data/fs72241/maibauer/differences_clahe/"
                width_par = 512

            else:
                raise FileNotFoundError('No folder with differences found. Please check path.')
                sys.exit()

        else:
            path = "/Volumes/SSD/differences_clahe/"
            width_par = 128

        height_par = width_par

        # Use URL to load image.

        im = np.asarray(Image.open(path+img_file_name).convert("L"))

        if width_par != 1024:
            im = cv2.resize(im  , (width_par , height_par),interpolation = cv2.INTER_CUBIC)

        GT = []
        annotations = self.coco_obj.getAnnIds(imgIds=img_id)

        if(len(annotations)>0):
            for a in annotations:
                ann = self.coco_obj.loadAnns(a)
                GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
        
        else:
            GT.append(np.zeros((1024,1024)))
        
        if width_par != 1024:
            for i in range(len(GT)):
                GT[i] = cv2.resize(GT[i]  , (width_par , height_par),interpolation = cv2.INTER_CUBIC)

        dilation = False

        if dilation:
            radius = int(3)
            kernel = disk(radius)

            for i in range(len(GT)):
                GT[i] = binary_dilation(GT[i], footprint=kernel)

        GT = np.array(GT)
        
        cme_pix = np.any(GT, axis=0)*255
        bg_pix = np.all(GT==0, axis=0)*255

        seed = np.random.randint(1000)
        torch.manual_seed(seed)
       
        if self.transform:
            im = im.astype(np.uint8)
            im = self.transform(im)
            im = np.asarray(im.convert("L"))/255.0
        else:
            im = im/255.0

        torch.manual_seed(seed)

        if self.transform:
            cme_pix = cme_pix.astype(np.uint8)
            cme_pix = self.transform(cme_pix)
        
            bg_pix = bg_pix.astype(np.uint8)
            bg_pix = self.transform(bg_pix)

            cme_pix = np.asarray(cme_pix.convert("L"))/255.0
            bg_pix = np.asarray(bg_pix.convert("L"))/255.0

        else:
            cme_pix = cme_pix/255.0
            bg_pix = bg_pix/255.0
        
        GT = np.concatenate([cme_pix[None, :, :],bg_pix[None, :, :]],0)

        # make sure to apply same tranform to both

        return torch.tensor(im).unsqueeze(0), torch.tensor(GT)

    def __len__(self):
        return len(self.all_anns)

    def __getitem__(self, index):
       
        return self.get_img_and_annotation(index)


class CNN3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN3D, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[nn.Conv2d(in_channels=self.input_channels,out_channels=64,kernel_size=3,padding=1)])
        self.encoder_conv_01 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)])

        self.encoder_conv_10 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)])
        self.encoder_conv_11 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)])

        self.encoder_conv_20 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_21 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_22 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])

        self.encoder_conv_30 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_31 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_32 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.encoder_conv_40 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_41 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_42 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])


        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_41 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_40 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.decoder_convtr_32 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_31 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_30 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)])

        self.decoder_convtr_22 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_21 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_20 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)])

        self.decoder_convtr_11 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)])
        self.decoder_convtr_10 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)])

        self.decoder_convtr_01 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)])
        self.decoder_convtr_00 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64,out_channels=self.output_channels,kernel_size=3,padding=1)])

    def forward(self, input_img):
        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)
        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)
        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)
        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        x_42 = F.relu(self.encoder_conv_42(x_41))
        x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder


        # Decoder Stage - 5
        x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d = F.relu(self.decoder_convtr_42(x_4d))
        x_41d = F.relu(self.decoder_convtr_41(x_42d))
        x_40d = F.relu(self.decoder_convtr_40(x_41d))

        x_40d = x_40d + x_3

        # Decoder Stage - 4
        x_3d = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = F.relu(self.decoder_convtr_30(x_31d))

        x_30d = x_30d + x_2

        # Decoder Stage - 3
        x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))

        x_20d = x_20d + x_1

        # Decoder Stage - 2
        x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = F.relu(self.decoder_convtr_10(x_11d))


        x_10d = x_10d + x_0

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)

        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)

        return x_00d
    

def train(backbone):

    device = torch.device("cpu")

    batch_size = 4
    num_workers = 2

    aug = True

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")

    elif(torch.cuda.is_available()):
        device = torch.device("cuda:1")
        batch_size = 24
        num_workers = 8

    if backbone == 'cnn3d':
        model = CNN3D(1,2).to(device)

    else:
        model = Unet(
            backbone=backbone, # backbone network name
            in_channels=1,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
            num_classes=2,            # output channels (number of classes in your dataset)
        ).to(device)    
    
    composed = v2.Compose([v2.ToPILImage(), v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation((0, 360)), v2.RandomVerticalFlip(p=0.5)])
    
    if aug == True:
        dataset = FrondandDiff(transform=composed)

    else:
        dataset = FrondandDiff()

    indices = dataset.train_data_idx
       
    dataset_sub = torch.utils.data.Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
                                                dataset_sub,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )
    
    g_optimizer = optim.Adam(model.parameters(),1e-4)

    num_iter = 401

    smax = nn.Softmax2d()

    weights = np.zeros(2)
    #class 0 is cme, class 1 is background
    cme_count = 0
    bg_count = 0

    for data in data_loader:
        for b in range(np.shape(data[1])[0]):
            cme_data = data[1][b][0].float().to(device).cpu().numpy()
            bg_data = data[1][b][1].float().to(device).cpu().numpy()
            cme_count = cme_count + np.sum(cme_data)
            bg_count = bg_count + np.sum(bg_data)

    n_samples = cme_count + bg_count
    n_classes = 2

    weights[0] = n_samples/(n_classes*cme_count)
    weights[1] = n_samples/(n_classes*bg_count)

    weights = torch.tensor(weights).to(device, dtype=torch.float32)

    pixel_looser = nn.CrossEntropyLoss(weight=weights)

    optimizer_data = []
    
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    train_path = 'Model_Train/run_'+dt_string+"_model_"+backbone+'/'

    if not os.path.exists(train_path): 
        os.makedirs(train_path, exist_ok=True) 

    for epoch in range(0, num_iter):
        epoch_loss = 0
        model_name = "model_epoch_{}".format(epoch)

        for num, data in enumerate(data_loader, 0):
            
            start = time.time()
            g_optimizer.zero_grad()

            input_data = data[0].float().to(device)
            mask_data = data[1].float().to(device)

            pred = model(input_data)
            loss = pixel_looser(pred, mask_data)

            loss.backward()
            g_optimizer.step()
            
            epoch_loss += loss.item()
            # print(loss,time.time()-start)

        epoch_loss = epoch_loss/(num+1)

        optimizer_data.append([epoch, epoch_loss])

        hspace = 0.01
        wspace = 0.01

        if epoch % 5 == 0:
            fig,ax = plt.subplots(np.shape(mask_data)[0], 4, figsize=(2*4+wspace*2, 2*np.shape(mask_data)[0]+hspace*(np.shape(mask_data)[0]-1)))

            for b in range(np.shape(mask_data)[0]):
                ax[b][0].imshow(data[0][b][0].detach().cpu().numpy()) #image
                ax[b][1].imshow(data[1][b][0].detach().cpu().numpy()) #mask
                ax[b][2].imshow(smax(pred)[b][0].detach().cpu().numpy()) #pred
                ax[b][3].plot(*zip(*optimizer_data)) #loss

                ax[b][0].axis("off")
                ax[b][1].axis("off")
                ax[b][2].axis("off")
                ax[b][3].axis("off")

                ax[b][0].set_aspect("auto")
                ax[b][1].set_aspect("auto")
                ax[b][2].set_aspect("auto")
                ax[b][3].set_aspect("auto")

                plt.subplots_adjust(wspace=wspace, hspace=hspace)

                im_path = train_path+'images/'
                
                if not os.path.exists(im_path): 
                    os.makedirs(im_path, exist_ok=True) 

                fig.savefig(im_path+'output_'+model_name+'.png', bbox_inches='tight')
                plt.close()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), train_path+model_name+'.pth')
            torch.save(g_optimizer.state_dict(), train_path+model_name+'_weights.pth')

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path + "model_loss.csv", 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(optimizer_data)


@torch.no_grad()
def test(epoch, folder_path):
    start = time.time()

    device = torch.device("cpu")

    batch_size = 4
    num_workers = 2

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")

    elif(torch.cuda.is_available()):
        device = torch.device("cuda:1")
        batch_size = 24
        num_workers = 8

    backbone = folder_path.split('_')[-1]

    if backbone == 'cnn3d':
        model = CNN3D(1,2).to(device)

    else:
        model = Unet(
            backbone=backbone, # backbone network name
            in_channels=1,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
            num_classes=2,            # output channels (number of classes in your dataset)
        ).to(device)  

    train_path = 'Model_Train/' + folder_path + '/'

    epoch = int(epoch)
    
    model_name = "model_epoch_{}".format(epoch)

    model.load_state_dict(torch.load(train_path+model_name + ".pth", map_location=device))
    model.eval()

    dataset = FrondandDiff()
    
    indices = dataset.test_data_idx
    dataset_sub = torch.utils.data.Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
                                                dataset_sub,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )
    save_metrics = []

    for batch_no, data in enumerate(data_loader, 0):

        input_data = data[0].float().to(device)

        pred = model(input_data)

        metrics = evaluation.evaluate(pred.cpu().detach(),data[1].numpy(),data[0].cpu().detach(), model_name, folder_path, batch_no)

        save_metrics.append(metrics)

    save_metrics = np.nanmean(save_metrics, axis=0)

    metrics_path = 'Model_Metrics/' + folder_path + '/'

    if not os.path.exists(metrics_path): 
        os.makedirs(metrics_path, exist_ok=True) 

    np.save(metrics_path+model_name+'.npy', save_metrics)

if __name__ == "__main__":
    backbone = sys.argv[1]
    train(backbone=backbone)
