import torch
from torch import nn,optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import time 
from torchvision.transforms import v2
import sys
from evaluation import evaluate_basic
import os
import csv
from backbones_unet.model.unet import Unet
from datetime import datetime
from dataset import BasicSet
import matplotlib
from models import CNN2D

def train(backbone):

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

    if backbone == 'cnn2d':
        model = CNN2D(1,2).to(device)

    else:
        model = Unet(
            backbone=backbone, # backbone network name
            in_channels=1,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
            num_classes=2,            # output channels (number of classes in your dataset)
        ).to(device)    
    
    composed = v2.Compose([v2.ToPILImage(), v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)]) #v2.RandomRotation((0, 360))
    
    if aug == True:
        dataset = BasicSet(transform=composed)

    else:
        dataset = BasicSet()

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
        if os.path.isdir('/home/mbauer/Data/'):
            device = torch.device("cuda:1")
        elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
            device = torch.device("cuda")
        else:
            sys.exit("Invalid data path. Exiting...")    
        batch_size = 24
        num_workers = 8

    backbone = folder_path.split('_')[-1]

    if backbone == 'cnn2d':
        model = CNN2D(1,2).to(device)

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

    dataset = BasicSet()
    
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

        metrics = evaluate_basic(pred.cpu().detach(),data[1].numpy(),data[0].cpu().detach(), model_name, folder_path, batch_no)

        save_metrics.append(metrics)

    save_metrics = np.nanmean(save_metrics, axis=0)

    metrics_path = 'Model_Metrics/' + folder_path + '/'

    if not os.path.exists(metrics_path): 
        os.makedirs(metrics_path, exist_ok=True) 

    np.save(metrics_path+model_name+'.npy', save_metrics)

if __name__ == "__main__":
    try:
        backbone = sys.argv[1]
    except IndexError:
        backbone='resnet34'
    train(backbone=backbone)
