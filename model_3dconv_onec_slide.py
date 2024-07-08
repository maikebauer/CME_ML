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
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
import json
# from losses import miou_loss
from monai.losses.dice import DiceLoss

def train(backbone, ind_par):

    device = torch.device("cpu")

    batch_size = 4
    num_workers = 2
    width_par = 128
    aug = True
    win_size = 16
    stride = int(2)

    input_params = {'width_par': width_par, 'win_size': win_size, 'backbone': backbone}

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
        #matplotlib.use('Qt5Agg')

    elif(torch.cuda.is_available()):
        if os.path.isdir('/home/mbauer/Data/'):
            device = torch.device("cuda")
            #matplotlib.use('Qt5Agg')

            if(torch.cuda.device_count() >1):
                batch_size = 4
                num_workers = 2

        elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
            device = torch.device("cuda")
            batch_size = 4
            num_workers = 2
            width_par = 128

            if(torch.cuda.device_count() > 1):
                batch_size = 8
                num_workers = 4

        else:
            sys.exit("Invalid data path. Exiting...")    

    if aug == True:
        composed = v2.Compose([v2.ToTensor(), v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)])#, v2.RandomAutocontrast(p=0.25), v2.RandomEqualize(p=0.25), v2.RandomPhotometricDistort(p=0.25)])
        composed_val = v2.Compose([v2.ToTensor()])
    else:
        composed = v2.Compose([v2.ToTensor()])
        composed_val = v2.Compose([v2.ToTensor()])

    dataset = RundifSequence(transform=composed,mode='train',win_size=win_size,stride=stride,width_par=width_par,ind_par=ind_par)
    dataset_val = RundifSequence(transform=composed_val,mode='val',win_size=win_size,stride=stride,width_par=width_par,ind_par=ind_par)
    dataset_test = RundifSequence(transform=composed_val,mode='test',win_size=win_size,stride=stride,width_par=width_par,ind_par=ind_par)

    indices_val = dataset_val.img_ids_win
    indices_test = dataset_test.img_ids_win

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

    data_loader_test = torch.utils.data.DataLoader(
        
                                                dataset_test,
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
    seed = 42

    if backbone == 'unetr':
        torch.manual_seed(seed)
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
        dropout_rate=0.0)

    elif backbone == 'cnn3d':
        torch.manual_seed(seed) 
        model_seg = CNN3D(input_channels=1, output_channels=1)

    else:
        print('Invalid backbone...')
        sys.exit()

    if(torch.cuda.device_count() >1):
        model_seg = torch.nn.DataParallel(model_seg)

    model_seg.to(device)

    g_optimizer_seg = optim.Adam(model_seg.parameters(),1e-5)
    scheduler = optim.lr_scheduler.StepLR(g_optimizer_seg, step_size=40, gamma=0.1)

    num_iter = 101

    optimizer_data = []
    optimizer_data_val = []

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    folder_path = "run_"+dt_string+"_model_"+str(dataset.ind_par)+"_"+backbone_name+'/'
    train_path = 'Model_Train/'+folder_path
    # im_path = train_path+'images/'

    # cme_count = 0
    # bg_count = 0

    # weights = np.zeros(2)

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

    # weights[0] = 1.0
    # weights[1] = 1.0

    # weights = torch.tensor(weights).to(device, dtype=torch.float32)

    if backbone == 'unetr':
        pixel_looser = nn.BCEWithLogitsLoss()
    elif backbone == 'cnn3d':
        pixel_looser = nn.BCELoss()
        # pixel_looser = DiceLoss()

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    #os.makedirs(os.path.dirname(im_path), exist_ok=True)

    train_list_ind = [l.tolist() for l in dataset.img_ids_win]
    val_list_ind = [l.tolist() for l in dataset_val.img_ids_win]
    test_list_ind = [l.tolist() for l in dataset_test.img_ids_win]

    data_indices = {}
    data_indices["training"] = train_list_ind
    data_indices["validation"] = val_list_ind
    data_indices["test"] = test_list_ind

    with open(train_path+"indices.json", "w") as f:
        json.dump(data_indices, f)

    epoch_metrics_val = []
    epoch_metrics_test = []

    best_loss = 1e99
    num_no_improvement = 0

    log_dir = "runs/" + folder_path[:-1]

    sum_writer = SummaryWriter(log_dir)

    for epoch in range(num_iter):
        model_seg.train()

        epoch_loss = 0
        epoch_loss_val = 0

        model_name = "model_epoch_{}".format(epoch)
        save_metrics = []

        batch_loss = 0
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
                
            loss_seg = pixel_looser(pred_comb, mask_data)
            #batch_loss = batch_loss + loss_seg.item()

            loss_seg.backward()

            g_optimizer_seg.step()
            batch_loss = batch_loss + loss_seg.item()
            #print('batch_loss: ', batch_loss)

        epoch_loss = batch_loss/(num+1)
        sum_writer.add_scalar("Loss/train", epoch_loss, epoch)

        optimizer_data.append([epoch, epoch_loss])

        with torch.no_grad():

            val_metrics, epoch_loss_val = evaluate(data_loader_val, model_seg, device, indices_val, pixel_looser, input_params)

            sum_writer.add_scalar("Loss/val", epoch_loss_val, epoch)

            optimizer_data_val.append([epoch, epoch_loss_val])

            epoch_metrics_val.append(val_metrics)
            
            sum_writer.add_scalar("Precision/val", val_metrics[1], epoch)
            sum_writer.add_scalar("Recall/val", val_metrics[2], epoch)
            sum_writer.add_scalar("IOU/val", val_metrics[3], epoch)

            test_metrics, epoch_loss_test = evaluate(data_loader_test, model_seg, device, indices_test, pixel_looser, input_params)

            sum_writer.add_scalar("Loss/test", epoch_loss_test, epoch)

            epoch_metrics_test.append(test_metrics)

            sum_writer.add_scalar("Precision/test", test_metrics[1], epoch)
            sum_writer.add_scalar("Recall/test", test_metrics[2], epoch)
            sum_writer.add_scalar("IOU/test", test_metrics[3], epoch)

            if epoch_loss_val < best_loss:
                best_loss = epoch_loss_val

                if(torch.cuda.device_count() >1):
                    best_model_seg = copy.deepcopy(model_seg.module.state_dict())
                else:
                    best_model_seg = copy.deepcopy(model_seg.state_dict())

                best_weights_seg = copy.deepcopy(g_optimizer_seg.state_dict())
                num_no_improvement = 0

                torch.save(best_model_seg, train_path+'model_seg.pth')               
                torch.save(best_weights_seg, train_path+'model_weights_seg.pth')     

            else:
                num_no_improvement += 1

            print(f"Epoch: {epoch:.0f}, Loss: {epoch_loss:.10f}, Val Loss: {epoch_loss_val:.10f}, Test Loss: {epoch_loss_test:.10f}, No improvement in {num_no_improvement:.0f} epochs.")
        
        scheduler.step()

    sum_writer.close()

def evaluate(data_loader, model_seg, device, indices, pixel_looser, input_params):

    model_seg.eval()

    sigmoid = nn.Sigmoid()

    thresh = 0.1

    width_par = input_params['width_par']
    win_size = input_params['win_size']
    backbone = input_params['backbone']

    pred_save = []

    n_ind = len(set(np.array(indices).flatten()))

    input_imgs = np.zeros((n_ind, width_par, width_par))
    input_masks = np.zeros((n_ind, width_par, width_par))

    for val in range(n_ind):
        pred_save.append([])
    
    batch_loss = 0
    num_batch = 0

    ind_keys = sorted(set(np.array(indices).flatten()))
    ind_set = np.arange(0, len(ind_keys))
    ind_dict = {}

    for A, B in zip(ind_keys, ind_set):
        ind_dict[A] = B

    for num, data in enumerate(data_loader):
        
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

        loss_seg = pixel_looser(pred_comb, mask_data)
        
        for b in range(mask_data.shape[0]):
            for k in range(win_size):
                current_ind = ind_dict[indices[num_batch][k]]
                pred_save[current_ind].append(pred_comb[b,0,k,:,:])

                if np.all(input_imgs[current_ind]) == 0:
                    input_imgs[current_ind] = input_data[b,0,k,:,:].cpu().detach().numpy()
                    input_masks[current_ind] = mask_data[b,0,k,:,:].cpu().detach().numpy()

            num_batch = num_batch + 1

        batch_loss = batch_loss + loss_seg.item()
    
    epoch_loss = batch_loss/(num+1)

    input_imgs = np.array(input_imgs)

    for h, pred_arr in enumerate(pred_save):
        for j in range(len(pred_arr)):
            if backbone == 'unetr':
                pred_arr[j] = sigmoid(pred_arr[j].cpu().detach())
            elif backbone == 'cnn3d':
                pred_arr[j] = pred_arr[j].cpu().detach()
        
        pred_save[h] = np.nanmean(pred_arr,axis=0)

    pred_save = np.array(pred_save)
    pred_save = torch.Tensor(pred_save)
    input_imgs = torch.Tensor(input_imgs)
    input_masks = torch.Tensor(input_masks)

    metrics = evaluate_onec_slide(pred_save.cpu().detach().numpy(),input_masks.cpu().detach().numpy(), thresh=thresh)

    return metrics, epoch_loss

def test(model_name):

    device = torch.device("cpu")

    model_path = 'Model_Train/'+ model_name + '/model_seg.pth'
    weights_path = 'Model_Train/'+ model_name + '/model_weights_seg.pth'

    batch_size = 2
    num_workers = 1
    width_par = 128
    aug = True
    win_size = 16
    stride = int(2)

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
        #matplotlib.use('Qt5Agg')

    elif(torch.cuda.is_available()):
        if os.path.isdir('/home/mbauer/Data/'):
            device = torch.device("cuda")
            #matplotlib.use('Qt5Agg')

            if(torch.cuda.device_count() >1):
                batch_size = 4
                num_workers = 2

        elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
            device = torch.device("cuda")
            batch_size = 4
            num_workers = 2
            width_par = 128

            if(torch.cuda.device_count() >1):
                batch_size = 8
                num_workers = 4

        else:
            sys.exit("Invalid data path. Exiting...")    

    sigmoid = nn.Sigmoid()

    backbone = model_name.split('_')[-1]
    ind_par = int(model_name.split('_')[4])

    input_params = {'width_par': width_par, 'win_size': win_size, 'backbone': backbone}

    if backbone == 'unetr':
        pixel_looser = nn.BCEWithLogitsLoss()
    elif backbone == 'cnn3d':
        pixel_looser = nn.BCELoss()
        # pixel_looser = DiceLoss()

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
        dropout_rate=0.0)

    elif backbone == 'cnn3d': 
        model_seg = CNN3D(input_channels=1, output_channels=1)

    model_seg.load_state_dict(torch.load(model_path, map_location=device))
    model_seg.to(device)

    composed = v2.Compose([v2.ToTensor()])

    dataset = RundifSequence(transform=composed,mode='test',win_size=win_size,stride=stride,ind_par=ind_par)

    data_loader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )    

    indices_test = dataset.img_ids_win

    with torch.no_grad():

        metrics, epoch_loss = evaluate(data_loader, model_seg, device, indices_test, pixel_looser, input_params)
    
    metrics_path = 'Model_Test/' + model_name+ '/'

    if not os.path.exists(metrics_path): 
        os.makedirs(metrics_path, exist_ok=True) 

    np.save(metrics_path+'metrics.npy', metrics)
    print(metrics)

if __name__ == "__main__":
    try:
        backbone = str(sys.argv[1])
    except IndexError:
        backbone = 'cnn3d'

    try:
        mode = str(sys.argv[2])
    except IndexError:
        mode = 'train'

    try:
        ind_par = int(sys.argv[3])
    except IndexError:
        ind_par = None  

    if mode == 'train':
        train(backbone=backbone,ind_par=ind_par)
    
    elif mode == 'test':
        model_name = sys.argv[4]
        test(model_name=model_name)