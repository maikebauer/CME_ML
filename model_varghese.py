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
from model_flow import FNet, sep_noevent_data, FrondandDiff, backward_warp, charbonnier_loss, spatial_smoothing_loss
from model_torch import CNN3D
from matplotlib.colors import ListedColormap
import evaluation
import copy
from unetr import UNETR

def miou_loss(pred_im2, real_im2):
    """
    Differentiable mean IOU loss, as defined in Varghese 2021, equation 11.
    """
    
    prod_im2 = pred_im2 * real_im2

    sum_im2 = pred_im2 + real_im2

    loss = torch.sum(torch.abs(prod_im2))/torch.sum(torch.abs(sum_im2 - prod_im2))

    return loss

def train(backbone):

    device = torch.device("cpu")

    batch_size = 4
    num_workers = 2

    aug = True

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
        matplotlib.use('Qt5Agg')
        width_par = 128
        os.system("export PYTORCH_ENABLE_MPS_FALLBACK=1")

    elif(torch.cuda.is_available()):
        device = torch.device("cuda:1")
        batch_size = 1
        num_workers = 1
        width_par = 128
    
    composed = v2.Compose([v2.ToPILImage(), v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation((0, 360)), v2.RandomVerticalFlip(p=0.5)])
    
    if aug == True:
        dataset = FrondandDiff(transform=composed)
        dataset_val = FrondandDiff()

    else:
        dataset = FrondandDiff()
        dataset_val = FrondandDiff()

    indices = dataset.train_paired_idx
    indices_val = dataset.val_paired_idx

    dataset_sub = torch.utils.data.Subset(dataset, indices)
    dataset_sub_val = torch.utils.data.Subset(dataset_val, indices_val)

    data_loader = torch.utils.data.DataLoader(
                                                dataset_sub,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )

    data_loader_val = torch.utils.data.DataLoader(
                                                dataset_sub_val,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )
    
    model_flow = FNet(1).to(device)

    backbone_name = 'varghese_' + backbone
    
    g_optimizer_flow = optim.Adam(model_flow.parameters(),1e-4)

    if backbone == 'cnn3d':
        model_seg = CNN3D(1,2).to(device)

    elif backbone == 'unetr':
        model_seg = UNETR(in_channels=1,
        out_channels=2,
        img_size=(width_par, width_par, 2),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0).to(device)

    else:
        model_seg = Unet(
            backbone=backbone, # backbone network name
            in_channels=1,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
            num_classes=2,            # output channels (number of classes in your dataset)
        ).to(device)  
    
    g_optimizer_seg = optim.Adam(model_seg.parameters(),1e-4)

    num_iter = 401

    optimizer_data = []

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    folder_path = "run_"+dt_string+"_model_"+backbone_name+'/'
    train_path = 'Model_Train/'+folder_path
    im_path = train_path+'images/'

    cme_count = 0
    bg_count = 0

    weights = np.zeros(2)

    # for data in data_loader:
    #     mask_data = data['gt'].float().to(device).cpu().numpy()
    #     for b in range(np.shape(mask_data)[0]):
    #         for j in range(2):
    #             cme_data = mask_data[b,j,0,:,:]
    #             bg_data = mask_data[b,j,1,:,:]
    #             cme_count = cme_count + np.sum(cme_data)
    #             bg_count = bg_count + np.sum(bg_data)

    # n_samples = cme_count + bg_count
    # n_classes = 2

    # weights[0] = (n_samples/(n_classes*cme_count))/2
    # weights[1] = (n_samples/(n_classes*bg_count))/1

    weights[0] = 50.0
    weights[1] = 1.0

    weights = torch.tensor(weights).to(device, dtype=torch.float32)

    pixel_looser = nn.CrossEntropyLoss(weight=weights)#weight=weights

    s_loss = spatial_smoothing_loss(device)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(im_path), exist_ok=True)

    smax = nn.Softmax2d()

    mask_cmap = ListedColormap(['#D81B60', '#FFC107', '#1E88E5']) #pink, yellow, blue
    metrics_path = 'Model_Metrics/' + folder_path
    
    epoch_metrics = []
    best_loss = 1e99
    num_no_improvement = 0

    for epoch in range(num_iter):
        epoch_loss = 0
        epoch_loss_val = 0
        model_name = "model_epoch_{}".format(epoch)
        save_metrics = []

        model_flow.train()
        model_seg.train()

        for num, data in enumerate(data_loader):
            
            g_optimizer_flow.zero_grad()
            g_optimizer_seg.zero_grad()

            input_data = data['image'].float().to(device)
            mask_data = data['gt'].float().to(device)

            im1 = input_data[:,0,:,:,:]
            im2 = input_data[:,1,:,:,:]

            mask1 = mask_data[:,0,0,:,:].unsqueeze(1)
            mask2 = mask_data[:,1,0,:,:].unsqueeze(1)

            mask_comb1 = torch.cat((mask1,mask_data[:,0,1,:,:].unsqueeze(1)),1)
            mask_comb2 = torch.cat((mask2,mask_data[:,1,1,:,:].unsqueeze(1)),1)
            

            flow = model_flow(im2,im1)

            if backbone == 'unetr':
                
                im_concat = torch.cat((im1,im2),1)
                im_concat = torch.permute(im_concat, (0, 2, 3, 1)).unsqueeze(1)
                pred1 = model_seg(im_concat)
            
            else:
                pred1 = model_seg(im1)

            print(pred1.shape)


            total_loss = pixel_looser(pred1[:,:,0,:,:], mask_comb1) + pixel_looser(pred1[:,:,1,:,:], mask_comb2)
            # pred_smax1 = smax(pred1)
            # pred_bin1 = torch.where(pred_smax1[:,0,:,:] > 0.5, 1, 0).unsqueeze(1).float().to(device)

            # pred2 = model_seg(im2)
            # pred_smax2 = smax(pred2)
            # pred_bin2 = torch.where(pred_smax2[:,0,:,:] > 0.5, 1, 0).unsqueeze(1).float().to(device)

            # predw1 = backward_warp(pred_bin1,flow)
            # bw1 = backward_warp(im1,flow)
            # mw1 = backward_warp(mask1,flow)
            # difference1 = im2 - bw1

            # loss_seg = pixel_looser(pred1, mask_comb1)
            # tc_loss = 1 - miou_loss(predw1, pred_bin2)
            # mse_loss = F.mse_loss(bw1, im2)
            # ch_loss = s_loss(flow)

            # alpha = 0.75
            # total_loss = (1-alpha)*loss_seg + tc_loss# + mse_loss + ch_loss

            total_loss.backward()
            g_optimizer_flow.step()
            g_optimizer_seg.step()
            epoch_loss += total_loss.item()

        epoch_loss = epoch_loss/(num+1)

        optimizer_data.append([epoch, epoch_loss])
        flow_plot = flow[0].detach().cpu().numpy()
        magnitude = np.sqrt(flow_plot[0,:,:]**2+flow_plot[1,:,:]**2)

        hspace = 0.01
        wspace = 0.01

        nrows = 6
        ncols = 2

        figsize=(3*ncols+wspace*2, 3*nrows+hspace*(nrows-1))

        fig,ax = plt.subplots(nrows,ncols,figsize=figsize)

        im1_plot = im1[0].detach().cpu().numpy()
        im2_plot = im2[0].detach().cpu().numpy()
        mask1_plot = mask1[0].detach().cpu().numpy()
        mask2_plot = mask2[0].detach().cpu().numpy()

        difference_plot = difference1[0].detach().cpu().numpy()
            
        bw_1_plot = bw1[0].detach().cpu().numpy()
        mw_1_plot = mw1[0].detach().cpu().numpy()
        predw_plot = predw1[0].detach().cpu().numpy()
        pred_bin_plot = pred_bin1[0].detach().cpu().numpy()

        diff_mw_plot = predw_plot - mask2_plot
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

        ax[3][0].imshow(im2_plot[0])
        ax[3][1].imshow(pred_bin_plot[0])

        ax[4][0].imshow(bw_1_plot[0])
        ax[4][1].imshow(predw_plot[0])

        ax[5][0].imshow(difference_plot[0],cmap='twilight')
        ax[5][1].imshow(diff_mw_plot[0],cmap=mask_cmap, vmin=-1, vmax=1)

        for i in range(nrows):
            for j in range(ncols):
                ax[i][j].axis("off")
                ax[i][j].set_aspect("auto")

        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(im_path+'output_'+model_name+'.png', bbox_inches='tight')
        plt.close('all')

        with torch.no_grad():
            model_flow.eval()
            model_seg.eval()

            for num, data in enumerate(data_loader_val):

                input_data = data['image'].float().to(device)
                mask_data = data['gt'].float().to(device)

                im1 = input_data[:,0,:,:,:]
                im2 = input_data[:,1,:,:,:]

                mask1 = mask_data[:,0,0,:,:].unsqueeze(1)
                mask2 = mask_data[:,1,0,:,:].unsqueeze(1)

                mask_comb1 = torch.cat((mask1,mask_data[:,0,1,:,:].unsqueeze(1)),1)
                mask_comb2 = torch.cat((mask2,mask_data[:,1,1,:,:].unsqueeze(1)),1)
                
                flow = model_flow(im2,im1)

                pred1 = model_seg(im1)
                pred_smax1 = smax(pred1)
                pred_bin1 = torch.where(pred_smax1[:,0,:,:] > 0.5, 1, 0).unsqueeze(1).float().to(device)

                pred2 = model_seg(im2)
                pred_smax2 = smax(pred2)
                pred_bin2 = torch.where(pred_smax2[:,0,:,:] > 0.5, 1, 0).unsqueeze(1).float().to(device)

                predw1 = backward_warp(pred_bin1,flow)
                bw1 = backward_warp(im1,flow)
                mw1 = backward_warp(mask1,flow)
                difference1 = im2 - bw1

                loss_seg_val = pixel_looser(pred1, mask_comb1)
                tc_loss_val = 1 - miou_loss(predw1, pred_bin2)
                mse_loss_val = F.mse_loss(bw1, im2)
                ch_loss_val = s_loss(flow)
                alpha = 0.75

                total_loss_val = (1-alpha)*loss_seg_val + tc_loss_val# + mse_loss_val + ch_loss_val

                epoch_loss_val += total_loss_val.item()

                metrics1 = evaluation.evaluate(pred1.cpu().detach(),mask_comb1.cpu().detach().numpy(),im1.cpu().detach().numpy(), model_name, folder_path, num)
                save_metrics.append(metrics1)

                metrics2 = evaluation.evaluate(pred2.cpu().detach(),mask_comb2.cpu().detach().numpy(),im2.cpu().detach().numpy(), model_name, folder_path, num)
                save_metrics.append(metrics2)

            epoch_loss_val = epoch_loss_val/(num+1)

            print(f"Epoch: {epoch:.0f}, Loss: {epoch_loss:.5f}, Val Loss: {epoch_loss_val:.4f}, No improvement in {num_no_improvement:.0f} epochs.")

            if epoch_loss_val < best_loss:
                best_loss = epoch_loss_val
                best_model_flow = copy.deepcopy(model_flow.state_dict())
                best_model_seg = copy.deepcopy(model_seg.state_dict())
                best_weights_flow = copy.deepcopy(g_optimizer_flow.state_dict())
                best_weights_seg = copy.deepcopy(g_optimizer_seg.state_dict())
                num_no_improvement = 0
            else:
                num_no_improvement += 1

                if num_no_improvement >= 16:
                    torch.save(best_model_flow, train_path+'model_flow.pth')               
                    torch.save(best_weights_flow, train_path+'model_weights_flow.pth')
                    torch.save(best_model_seg, train_path+'model_seg.pth')               
                    torch.save(best_weights_seg, train_path+'model_weights_seg.pth')     

                    if not os.path.exists(metrics_path): 
                        os.makedirs(metrics_path, exist_ok=True) 

                    np.save(metrics_path+'metrics.npy', epoch_metrics)

                    sys.exit()

        save_metrics = np.nanmean(save_metrics, axis=0)
        epoch_metrics.append(save_metrics)


    

if __name__ == "__main__":
    try:
        backbone = sys.argv[1]
    except IndexError:
        backbone = 'unetr'

    train(backbone=backbone)
