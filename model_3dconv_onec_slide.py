import torch
from torch import nn,optim
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
from torchvision.transforms import v2
import sys
import os
from datetime import datetime
import copy
from models import UNETR_16, CNN3D
from dataset import RundifSequence
import matplotlib
from evaluation import evaluate_onec_slide

def train(backbone):

    device = torch.device("cpu")

    batch_size = 1
    num_workers = 1
    width_par = 128
    aug = True
    win_size = 16
    stride = int(win_size/8)

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
        #matplotlib.use('Qt5Agg')

    elif(torch.cuda.is_available()):
        if os.path.isdir('/home/mbauer/Data/'):
            device = torch.device("cuda:1")
            matplotlib.use('Qt5Agg')
        elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
            device = torch.device("cuda")
            batch_size = 4
            num_workers = 2
            width_par = 512
        else:
            sys.exit("Invalid data path. Exiting...")    
    
    # composed = v2.Compose([v2.ToPILImage(), v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)]) # v2.RandomRotation((0, 360))
    if aug == True:
        composed = v2.Compose([v2.ToTensor(), v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)])
        composed_val = v2.Compose([v2.ToTensor()])
    else:
        composed = v2.Compose([v2.ToTensor()])
        composed_val = v2.Compose([v2.ToTensor()])

    dataset = RundifSequence(transform=composed,mode='train',win_size=win_size,stride=stride)
    dataset_val = RundifSequence(transform=composed_val,mode='val',win_size=win_size,stride=stride)

    indices_train = dataset.train_paired_idx
    indices_val = dataset_val.val_paired_idx

    data_loader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )

    data_loader_val = torch.utils.data.DataLoader(
        
                                                dataset_val,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )

    #mean_train, std_train = get_mean_std(data_loader)
    #normalize_train = v2.Normalize(mean=mean_train, std=std_train)

    #mean_val, std_val = get_mean_std(data_loader_val)
    #normalize_val = v2.Normalize(mean=mean_val, std=std_val)

    backbone_name = '3dconv_' + backbone

    if backbone == 'unetr':
        model_seg = UNETR_16(in_channels=1,
        out_channels=1,
        img_size=(width_par, width_par, win_size),
        feature_size=32,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0).to(device)

    elif backbone == 'cnn3d': 
        model_seg = CNN3D(input_channels=1, output_channels=1).to(device)

    else:
        print('Invalid backbone...')
        sys.exit()

    g_optimizer_seg = optim.Adam(model_seg.parameters(),1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer_seg, 'min', patience=3)
    #scheduler = optim.lr_scheduler.StepLR(g_optimizer_seg, step_size=10, gamma=0.1)
    num_iter = 101

    optimizer_data = []
    optimizer_data_val = []

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

    weights[0] = 1.0
    weights[1] = 1.0

    weights = torch.tensor(weights).to(device, dtype=torch.float32)

    if backbone == 'unetr':
        pixel_looser = nn.BCEWithLogitsLoss(pos_weight=weights[1])
    elif backbone == 'cnn3d':
        pixel_looser = nn.BCELoss()
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(im_path), exist_ok=True)

    sigmoid = nn.Sigmoid()

    metrics_path = 'Model_Metrics/' + folder_path
    
    epoch_metrics = []
    best_loss = 1e99
    num_no_improvement = 0

    for epoch in range(num_iter):
        epoch_loss = 0
        epoch_loss_val = 0

        model_name = "model_epoch_{}".format(epoch)
        save_metrics = []

        model_seg.train()

        for num, data in enumerate(data_loader):
            g_optimizer_seg.zero_grad()
            #input_data = normalize_train(data['image']).float().to(device)

            input_data = data['image'].float().to(device)
            mask_data = data['gt'].float().to(device)

            if backbone == 'unetr':
                im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
                pred_comb = model_seg(im_concat)
                mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))

            elif backbone == 'cnn3d':
                input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
                mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
                pred_comb = model_seg(input_data)

            else:
                print('Invalid backbone...')
                sys.exit()

            batch_loss = 0

            for k in range(win_size):
                
                loss_seg = pixel_looser(pred_comb[:,0,k,:,:], mask_data[:,0,k,:,:])

                batch_loss = batch_loss + loss_seg
    
            batch_loss.backward()

            g_optimizer_seg.step()

            epoch_loss = epoch_loss + batch_loss.item()

        epoch_loss = epoch_loss/(num+1)

        optimizer_data.append([epoch, epoch_loss])

        im_path = train_path+'images/'
            
        if not os.path.exists(im_path): 
            os.makedirs(im_path, exist_ok=True) 

        hspace = 0.01
        wspace = 0.01

        fig,ax = plt.subplots(win_size*np.shape(mask_data)[0], 4, figsize=(2*4+wspace*2, 2*(win_size*np.shape(mask_data)[0])+hspace*(win_size*np.shape(mask_data)[0]-1)))
        
        for b in range(np.shape(mask_data)[0]):
            for w in range(win_size):
                ax[w+win_size*b][0].imshow(data['image'][b][w][0].detach().cpu().numpy()) #image
                ax[w+win_size*b][1].imshow(mask_data[b][0][w].detach().cpu().numpy()) #mask
                if backbone == 'unetr':
                    ax[w+win_size*b][2].imshow(sigmoid(pred_comb[b,0,w,:,:]).detach().cpu().numpy()) #pred
                elif backbone == 'cnn3d':
                    ax[w+win_size*b][2].imshow(pred_comb[b,0,w,:,:].detach().cpu().numpy())
                ax[w+win_size*b][3].plot(*zip(*optimizer_data)) #loss

                ax[w+win_size*b][0].axis("off")
                ax[w+win_size*b][1].axis("off")
                ax[w+win_size*b][2].axis("off")
                ax[w+win_size*b][3].axis("off")

                ax[w+win_size*b][0].set_aspect("auto")
                ax[w+win_size*b][1].set_aspect("auto")
                ax[w+win_size*b][2].set_aspect("auto")
                ax[w+win_size*b][3].set_aspect("auto")

                ax[w+win_size*b][0].axis("off")
                ax[w+win_size*b][1].axis("off")
                ax[w+win_size*b][2].axis("off")
                ax[w+win_size*b][3].axis("off")

                ax[w+win_size*b][0].set_aspect("auto")
                ax[w+win_size*b][1].set_aspect("auto")
                ax[w+win_size*b][2].set_aspect("auto")
                ax[w+win_size*b][3].set_aspect("auto")

                plt.subplots_adjust(wspace=wspace, hspace=hspace)

            fig.savefig(im_path+'output_'+model_name+'.png', bbox_inches='tight')
            plt.close()

        with torch.no_grad():
            model_seg.eval()
            
            pred_save = []

            n_val = (max(indices_val.flatten())-min(indices_val.flatten()))+1

            input_imgs = np.zeros((n_val, width_par, width_par))
            input_masks = np.zeros((n_val, width_par, width_par))

            for val in range(n_val):
                pred_save.append([])

            for num, data in enumerate(data_loader_val):

                #input_data = normalize_val(data['image']).float().to(device)
                input_data = data['image'].float().to(device)
                mask_data = data['gt'].float().to(device)

                if backbone == 'unetr':
                    
                    im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
                    pred_comb = model_seg(im_concat)
                    mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
                    input_data = torch.permute(input_data, (0, 2, 1, 3, 4))

                elif backbone == 'cnn3d':
                    input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
                    mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
                    pred_comb = model_seg(input_data)

                else:

                    print('Invalid backbone...')
                    sys.exit()

                batch_loss_val = 0
                
                for k in range(win_size):
                    current_ind = indices_val[num][k]-min(indices_val.flatten())
                    
                    pred_save[current_ind].append(pred_comb[:,0,k,:,:])

                    if np.all(input_imgs[current_ind]) == 0:
                        input_imgs[current_ind] = input_data[:,0,k,:,:].cpu().detach().numpy()
                        input_masks[current_ind] = mask_data[:,0,k,:,:].cpu().detach().numpy()

                    loss_seg_val = pixel_looser(pred_comb[:,0,k,:,:], mask_data[:,0,k,:,:])
                    batch_loss_val = batch_loss_val + loss_seg_val

                epoch_loss_val = epoch_loss_val + batch_loss_val.item()

            epoch_loss_val = epoch_loss_val/(num+1)
            
            optimizer_data_val.append([epoch, epoch_loss_val])

            metrics_im_path = metrics_path+'images/'
                    
            if not os.path.exists(metrics_im_path): 
                os.makedirs(metrics_im_path, exist_ok=True) 

            thresh = 0.1

            hspace = 0.01
            wspace = 0.01

            fig,ax = plt.subplots(win_size*np.shape(mask_data)[0], 5, figsize=(2*5+wspace*2, 2*(win_size*np.shape(mask_data)[0])+hspace*(win_size*np.shape(mask_data)[0]-1)))
            
            for b in range(np.shape(mask_data)[0]):
                for w in range(win_size):
                    
                    if backbone == 'unetr':
                        bin_pred = sigmoid(pred_comb[b,0,w,:,:]).detach().cpu().numpy()
                    elif backbone == 'cnn3d':
                        bin_pred = pred_comb[b,0,w,:,:].detach().cpu().numpy()

                    ax[w+win_size*b][0].imshow(data['image'][b][w][0].detach().cpu().numpy()) #image
                    ax[w+win_size*b][1].imshow(mask_data[b][0][w].detach().cpu().numpy()) #mask
                    ax[w+win_size*b][2].imshow(bin_pred) #pred
                    ax[w+win_size*b][3].imshow(np.where(bin_pred > thresh, 1, 0)) #pred
                    ax[w+win_size*b][4].plot(*zip(*optimizer_data), color='blue') #loss
                    ax[w+win_size*b][4].plot(*zip(*optimizer_data_val), color='orange') #loss

                    ax[w+win_size*b][0].axis("off")
                    ax[w+win_size*b][1].axis("off")
                    ax[w+win_size*b][2].axis("off")
                    ax[w+win_size*b][3].axis("off")
                    ax[w+win_size*b][4].axis("off")

                    ax[w+win_size*b][0].set_aspect("auto")
                    ax[w+win_size*b][1].set_aspect("auto")
                    ax[w+win_size*b][2].set_aspect("auto")
                    ax[w+win_size*b][3].set_aspect("auto")
                    ax[w+win_size*b][4].set_aspect("auto")

                    ax[w+win_size*b][0].axis("off")
                    ax[w+win_size*b][1].axis("off")
                    ax[w+win_size*b][2].axis("off")
                    ax[w+win_size*b][3].axis("off")
                    ax[w+win_size*b][4].axis("off")

                    ax[w+win_size*b][0].set_aspect("auto")
                    ax[w+win_size*b][1].set_aspect("auto")
                    ax[w+win_size*b][2].set_aspect("auto")
                    ax[w+win_size*b][3].set_aspect("auto")
                    ax[w+win_size*b][4].set_aspect("auto")

                    plt.subplots_adjust(wspace=wspace, hspace=hspace)

                fig.savefig(metrics_im_path+'output_'+model_name+'.png', bbox_inches='tight')
                plt.close()

            input_imgs = np.array(input_imgs)
            for h, pred_arr in enumerate(pred_save):
                for j in range(len(pred_arr)):
                    if backbone == 'unetr':
                        pred_arr[j] = sigmoid(pred_arr[j].cpu().detach())*(j+1)/len(pred_arr)
                    elif backbone == 'cnn3d':
                        pred_arr[j] = pred_arr[j].cpu().detach()*(j+1)/len(pred_arr)
                
                pred_save[h] = np.nanmean(pred_arr,axis=0)
                #pred_save[h] = np.nansum(pred_arr,axis=0)
                #pred_save[h] = np.where(pred_save[h] > 1, 1, pred_save[h])

            pred_save = np.array(pred_save)
            pred_save = torch.Tensor(pred_save[:,0,:,:])
            input_imgs = torch.Tensor(input_imgs)
            input_masks = torch.Tensor(input_masks)

            metrics = evaluate_onec_slide(pred_save.cpu().detach().numpy(),input_masks.cpu().detach().numpy(),input_imgs.cpu().detach().numpy(), model_name, folder_path, num, epoch, thresh=thresh)

            epoch_metrics.append(metrics)

            if epoch_loss_val < best_loss:
                best_loss = epoch_loss_val
                best_model_seg = copy.deepcopy(model_seg.state_dict())
                best_weights_seg = copy.deepcopy(g_optimizer_seg.state_dict())
                num_no_improvement = 0

                torch.save(best_model_seg, train_path+'model_seg.pth')               
                torch.save(best_weights_seg, train_path+'model_weights_seg.pth')     

            else:
                num_no_improvement += 1

                #if num_no_improvement >= 6:
                #    sys.exit()

            if not os.path.exists(metrics_path): 
                os.makedirs(metrics_path, exist_ok=True) 

            np.save(metrics_path+'metrics.npy', epoch_metrics)

            print(f"Epoch: {epoch:.0f}, Loss: {epoch_loss:.5f}, Val Loss: {epoch_loss_val:.4f}, No improvement in {num_no_improvement:.0f} epochs.")
            scheduler.step(epoch_loss_val)

if __name__ == "__main__":
    try:
        backbone = sys.argv[1]
    except IndexError:
        backbone = 'cnn3d'

    train(backbone=backbone)