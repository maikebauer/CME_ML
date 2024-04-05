import torch
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
from torchvision.transforms import v2
import sys
import os
import csv
from backbones_unet.model.unet import Unet
from datetime import datetime
import matplotlib
from models import FNet
from utils import backward_warp
from dataset import FlowSet
from losses import spatial_smoothing_loss

def train():

    device = torch.device("cpu")

    batch_size = 1
    num_workers = 1
    width_par = 128
    aug = True

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
        matplotlib.use('Qt5Agg')

    elif(torch.cuda.is_available()):
        if os.path.isdir('/home/mbauer/Data/'):
            device = torch.device("cuda:1")
        elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
            device = torch.device("cuda")
            batch_size = 8
            num_workers = 4
            width_par = 512
        else:
            sys.exit("Invalid data path. Exiting...")    
    
    composed = v2.Compose([v2.ToPILImage(), v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)]) #v2.RandomRotation((0, 360))
    
    if aug == True:
        dataset = FlowSet(transform=composed)

    else:
        dataset = FlowSet()

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