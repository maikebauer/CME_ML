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
import matplotlib

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

    def get_img_and_annotation(self,idxs):
        
        seed = np.random.randint(1000)
        torch.manual_seed(seed)

        GT_all = []
        im_all = []

        for idx in idxs:
            
            img_id   = self.img_ids[idx]
            img_info = self.coco_obj.loadImgs([img_id])[0]
            img_file_name = img_info["file_name"].split('/')[-1]

            if torch.cuda.is_available():
                
                if os.path.isdir('/home/mbauer/Data/'):
                    path = "/home/mbauer/Data/differences_clahe/"
                    width_par = 128
                
                elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
                    path = "/gpfs/data/fs72241/maibauer/differences_clahe/"
                    width_par = 128

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
            
            if self.transform:
                torch.manual_seed(seed)
                im = im.astype(np.uint8)
                im = self.transform(im)
                im = np.asarray(im.convert("L"))/255.0
            else:
                im = im/255.0
            if self.transform:
                torch.manual_seed(seed)
                cme_pix = cme_pix.astype(np.uint8)
                cme_pix = self.transform(cme_pix)

                torch.manual_seed(seed)
                bg_pix = bg_pix.astype(np.uint8)
                bg_pix = self.transform(bg_pix)

                cme_pix = np.asarray(cme_pix.convert("L"))/255.0
                bg_pix = np.asarray(bg_pix.convert("L"))/255.0

            else:
                cme_pix = cme_pix/255.0
                bg_pix = bg_pix/255.0
            
            GT = np.concatenate([cme_pix[None, :, :],bg_pix[None, :, :]],0)

            GT_all.append(GT)
            im_all.append(im)

            # make sure to apply same tranform to both
        GT_all = np.array(GT_all)
        im_all = np.array(im_all)

        return {'image':torch.tensor(im_all).unsqueeze(1), 'gt':torch.tensor(GT_all)}

    def __len__(self):
        return len(self.all_anns)

    def __getitem__(self, index):
       
        return self.get_img_and_annotation(index)
    

def charbonnier_loss(delta, gamma=0.45, epsilon=1e-6):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), gamma))
    return loss

class spatial_smoothing_loss(nn.Module):
    #from https://akshay-sharma1995.github.io/files/ml_temp_loss.pdf
    def __init__(self, device):
        super(spatial_smoothing_loss, self).__init__()
        self.eps = 1e-6
        self.device = device

    def forward(self, X):  # X is flow map
        u = X[:, 0:1]
        # Rest of the code
        v = X[:,1:2]
        # print("u",u.size())
        hf1 = torch.tensor([[[[0,0,0],[-1,2,-1],[0,0,0]]]]).type(torch.FloatTensor).to(self.device)
        hf2 = torch.tensor([[[[0,-1,0],[0,2,0],[0,-1,0]]]]).type(torch.FloatTensor).to(self.device)
        hf3 = torch.tensor([[[[-1,0,-1],[0,4,0],[-1,0,-1]]]]).type(torch.FloatTensor).to(self.device)
        # diff = torch.add(X, -Y)
        
        u_hloss = F.conv2d(u,hf1,padding=1,stride=1)
        # print("uhloss",type(u_hloss))
        u_vloss = F.conv2d(u,hf2,padding=1,stride=1)
        u_dloss = F.conv2d(u,hf3,padding=1,stride=1)

        v_hloss = F.conv2d(v,hf1,padding=1,stride=1)
        v_vloss = F.conv2d(v,hf2,padding=1,stride=1)
        v_dloss = F.conv2d(v,hf3,padding=1,stride=1)

        u_hloss = charbonier(u_hloss,self.eps)
        u_vloss = charbonier(u_vloss,self.eps)
        u_dloss = charbonier(u_dloss,self.eps)

        v_hloss = charbonier(v_hloss,self.eps)
        v_vloss = charbonier(v_vloss,self.eps)
        v_dloss = charbonier(v_dloss,self.eps)


        # error = torch.sqrt( diff * diff + self.eps )
        # loss = torch.sum(error) 
        loss = u_hloss + u_vloss + u_dloss + v_hloss + v_vloss + v_dloss
        # print('char_losss',loss)
        return loss 

def charbonier(x,eps):
	gamma = 0.45
	# print("x.type",type(x))
	loss = x*x + eps*eps
	loss = torch.pow(loss,gamma)
	loss = torch.mean(loss)
	return loss

def train():

    device = torch.device("cpu")

    batch_size = 4
    num_workers = 2

    aug = False

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
        matplotlib.use('Qt5Agg')
        width_par = 128

    elif(torch.cuda.is_available()):
        device = torch.device("cuda:1")
        batch_size = 24
        num_workers = 8
        width_par = 128
    
    composed = v2.Compose([v2.ToPILImage(), v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation((0, 360)), v2.RandomVerticalFlip(p=0.5)])
    
    if aug == True:
        dataset = FrondandDiff(transform=composed)

    else:
        dataset = FrondandDiff()

    indices = dataset.train_paired_idx
       
    dataset_sub = torch.utils.data.Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
                                                dataset_sub,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )
    
    model = FNet(1).to(device)

    backbone = 'flow_net'
    
    g_optimizer = optim.Adam(model.parameters(),1e-4)

    num_iter = 401

    optimizer_data = []

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    train_path = 'Model_Train/run_'+dt_string+"_model_"+backbone+'/'
    im_path = train_path+'images/'

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(im_path), exist_ok=True)

    s_loss_model = spatial_smoothing_loss(device)

    for epoch in range(num_iter):
        epoch_loss = 0
        model_name = "model_epoch_{}".format(epoch)

        for num, data in enumerate(data_loader):
            
            input_data = data['image'].float().to(device)
            mask_data = data['gt'].float().to(device)

            im1 = input_data[:,0,:,:,:]
            im2 = input_data[:,1,:,:,:]

            mask1 = mask_data[:,0,0,:,:].unsqueeze(1)
            mask2 = mask_data[:,1,0,:,:].unsqueeze(1)

            g_optimizer.zero_grad()
            flow = model(im2,im1)

            bw1 = backward_warp(im1,flow)

            mw1 = backward_warp(mask1,flow)

            difference = im2 - bw1
            loss_mse = F.mse_loss(bw1, im2)
            loss_spatial_char = s_loss_model(flow)

            loss = loss_mse + loss_spatial_char
            #loss = charbonnier_loss(difference)

            loss.backward()
            g_optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss/(num+1)

        optimizer_data.append([epoch, epoch_loss])
        flow_plot = flow[0].detach().cpu().numpy()
        magnitude = np.sqrt(flow_plot[0,:,:]**2+flow_plot[1,:,:]**2)

        hspace = 0.01
        wspace = 0.01

        nrows = 5
        ncols = 2

        figsize=(4*ncols+wspace*2, 4*nrows+hspace*(nrows-1))

        fig,ax = plt.subplots(nrows,ncols,figsize=figsize)

        im1_plot = im1[0].detach().cpu().numpy()
        im2_plot = im2[0].detach().cpu().numpy()
        mask1_plot = mask1[0].detach().cpu().numpy()
        mask2_plot = mask2[0].detach().cpu().numpy()

        difference_plot = difference[0].detach().cpu().numpy()
            
        bw_1_plot = bw1[0].detach().cpu().numpy()
        mw_1_plot = mw1[0].detach().cpu().numpy()

        diff_mw_plot = mw_1_plot - mask2_plot
        ax[0][0].imshow(im1_plot[0])
        ax[0][1].imshow(mask1_plot[0])

        ax[1][0].imshow(im2_plot[0])
        ax[1][1].imshow(mask2_plot[0])

        ax[2][0].imshow(magnitude)
        nvec = 30  # Number of vectors to be displayed along each image dimension
        nl, nc = width_par,width_par
        step = max(nl//nvec, nc//nvec)

        y, x = np.mgrid[:nl:step, :nc:step]
        u_ = flow_plot[0][::step, ::step]
        v_ = flow_plot[1][::step, ::step]

        ax[2][0].quiver(x, y, u_, v_, color='r', units='dots',angles='xy', scale_units='xy', lw=3)
        ax[2][1].plot(*zip(*optimizer_data))

        ax[3][0].imshow(bw_1_plot[0])
        ax[3][1].imshow(mw_1_plot[0])

        ax[4][0].imshow(difference_plot[0],cmap='twilight')
        ax[4][1].imshow(diff_mw_plot[0],cmap='twilight')

        for i in range(nrows):
            for j in range(ncols):
                ax[i][j].axis("off")
                ax[i][j].set_aspect("auto")

        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(im_path+'output_'+model_name+'.png', bbox_inches='tight')
        plt.close('all')
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), train_path+model_name+'.pth')               
            torch.save(g_optimizer.state_dict(), train_path+model_name+'_weights.pth')

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path + "model_loss.csv", 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(optimizer_data)

if __name__ == "__main__":
    train()