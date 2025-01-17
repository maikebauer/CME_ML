import torch
from torch import nn,optim
import numpy as np
from torchvision.transforms import v2
import sys
import os
from datetime import datetime
import copy
from dataset import RundifSequence
from evaluation import evaluate_onec_slide
from torch.utils.tensorboard import SummaryWriter
import json
from monai.losses.dice import DiceLoss
import io
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur
from utils import image_grid, parse_yml, load_augmentations, load_model, load_optimizer, load_scheduler, load_loss
import yaml
import shutil

def train():

    mode = 'train'

    config = parse_yml('config.yaml')

    device = config['model']['device']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']
    width_par = config['dataset']['width']

    win_size = config['dataset']['win_size']
    stride = config['dataset']['stride']
    thresh = config['train']['threshold_iou']  

    backbone = config['model']['name']
    data_parallel = config['train']['data_parallel']

    composed = load_augmentations(config)
    composed_val = v2.Compose([v2.ToTensor()])

    quick_run = config['dataset']['quick_run']

    data_path = config['dataset']['data_path']
    annotation_path = config['dataset']['annotation_path']

    dataset = RundifSequence(data_path=data_path,annotation_path=annotation_path,im_transform=composed,mode='train',win_size=win_size,stride=stride,width_par=width_par,include_potential=config['train']['include_potential'],include_potential_gt=config['train']['include_potential_gt'],quick_run=quick_run)
    dataset_val = RundifSequence(data_path=data_path,annotation_path=annotation_path,im_transform=composed_val,mode='val',win_size=win_size,stride=stride,width_par=width_par,include_potential=config['evaluate']['include_potential'],include_potential_gt=config['evaluate']['include_potential_gt'],quick_run=quick_run)

    data_loader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=batch_size,
                                                shuffle=config['train']['shuffle'],
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )

    data_loader_val = torch.utils.data.DataLoader(
        
                                                dataset_val,
                                                batch_size=batch_size,
                                                shuffle=config['evaluate']['shuffle'],
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )
    
    model_seg = load_model(config, mode)

    if config['train']['load_checkpoint']['load_model']:
        checkpoint = torch.load(config['train']['load_checkpoint']['checkpoint_path'], weights_only=True)
        last_epoch = checkpoint['epoch']

    else:
        last_epoch = -1

    if data_parallel:
        model_seg = torch.nn.DataParallel(model_seg)
    
    model_seg.to(device)

    g_optimizer_seg = load_optimizer(config, model_seg.parameters())

    print(f"Initial learning rate: {g_optimizer_seg.param_groups[0]['lr']:.8f}")    

    if config['scheduler']['use_scheduler']:
        scheduler = load_scheduler(config, g_optimizer_seg)

    else:
        scheduler = None

    num_iter = config['train']['epochs']

    optimizer_data = []
    optimizer_data_val = []

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    folder_path = "run_"+dt_string+"_model_"+backbone+"/"
    train_path = config['model']['model_path']+"Model_Train/"+folder_path
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    pixel_looser = load_loss(config)

    train_list_ind = [l.tolist() for l in dataset.img_ids_win]
    val_list_ind = [l.tolist() for l in dataset_val.img_ids_win]

    data_indices = {}
    data_indices["training"] = train_list_ind
    data_indices["validation"] = val_list_ind

    with open(train_path+"indices.json", "w") as f:
        json.dump(data_indices, f)

    shutil.copy("config.yaml", train_path+"config.yaml")

    best_iou = -1e99
    num_no_improvement = 0

    log_dir = "runs/" + folder_path[:-1]

    sum_writer = SummaryWriter(log_dir)
    sigmoid = nn.Sigmoid()

    metrics_batch = []

    for epoch in range(last_epoch+1, last_epoch+1+num_iter+1):
        model_seg.train()

        epoch_loss = 0
        epoch_loss_val = 0

        batch_loss = 0

        for num, data in enumerate(data_loader):
            g_optimizer_seg.zero_grad()
            
            input_data = data['image'].float().to(device)
            mask_data = data['gt'].float().to(device)

            if backbone == 'unetr':
                im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
                pred_comb = model_seg(im_concat)
                mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))

            elif backbone == 'cnn3d' or backbone == 'resunetpp' or backbone == 'maike_cnn3d':
                input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
                mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
                pred_comb = model_seg(input_data)

                if config['train']['binary_gt'] == False:
                    mask_data_gauss = mask_data.detach().clone()
                    gauss_blur = GaussianBlur(kernel_size=3, sigma=2)

                    for i in range(mask_data_gauss.shape[0]):
                        for j in range(mask_data_gauss.shape[2]):
                            mask_data_gauss[i,0,j,:,:] = gauss_blur(mask_data_gauss[i,:,j,:,:])
                            mask_data_gauss[i,0,j,:,:] = torch.where(mask_data_gauss[i,0,j,:,:] > 1, 1, mask_data_gauss[i,0,j,:,:])
            else:
                print('Invalid backbone...')
                sys.exit()

            loss_seg = pixel_looser(pred_comb, mask_data)
            loss_seg.backward()
            batch_loss = batch_loss + loss_seg.item()
            g_optimizer_seg.step()

            if config['model']['final_layer'].lower() == 'none':
                pred_comb = sigmoid(pred_comb)

            metrics_batch.append(evaluate_onec_slide(pred_comb.cpu().detach().numpy(), mask_data.cpu().detach().numpy(), thresh=thresh))


        train_metrics = np.nanmean(metrics_batch, axis=0)
        epoch_loss = batch_loss/(num+1)
        sum_writer.add_scalar("Loss/train", epoch_loss, epoch)

        sum_writer.add_scalar("Precision/train", train_metrics[1], epoch)
        sum_writer.add_scalar("Recall/train", train_metrics[2], epoch)
        sum_writer.add_scalar("IOU/train", train_metrics[3], epoch)
    
        # Prepare the plot
        input_data_train = input_data[0][0].cpu().detach().numpy()
        pred_comb_train = pred_comb[0][0].cpu().detach().numpy()
        mask_data_train = mask_data[0][0].cpu().detach().numpy()
        thresh_data_train = pred_comb_train.copy()
        thresh_data_train[thresh_data_train >= thresh] = 1
        thresh_data_train[thresh_data_train < thresh] = 0

        plot_images_train = np.concatenate((input_data_train, pred_comb_train, thresh_data_train, mask_data_train), axis=0)
        figure_train = image_grid(plot_images_train)

        # Convert to image and log
        
        sum_writer.add_figure("Training Data", figure_train, global_step=epoch)
        
        optimizer_data.append([epoch, epoch_loss])

        with torch.no_grad():
            model_seg.eval()

            val_metrics, epoch_loss_val, input_data_val, pred_comb_val, mask_data_val = evaluate(data_loader_val, model_seg, device, pixel_looser, config, thresh=thresh)

            sum_writer.add_scalar("Loss/val", epoch_loss_val, epoch)

            optimizer_data_val.append([epoch, epoch_loss_val])
            
            sum_writer.add_scalar("Precision/val", val_metrics[1], epoch)
            sum_writer.add_scalar("Recall/val", val_metrics[2], epoch)
            sum_writer.add_scalar("IOU/val", val_metrics[3], epoch)

            # Prepare the plot
            thresh_data_val = pred_comb_val.copy()
            thresh_data_val[thresh_data_val >= thresh] = 1
            thresh_data_val[thresh_data_val < thresh] = 0

            plot_images_val = np.concatenate((input_data_val, pred_comb_val, thresh_data_val, mask_data_val), axis=0)
            figure_val = image_grid(plot_images_val)

            # Convert to image and log
            
            sum_writer.add_figure("Validation Data", figure_val, global_step=epoch)

            if val_metrics[3] > best_iou:
                best_iou = val_metrics[3]

                if data_parallel:
                    best_model_seg = copy.deepcopy(model_seg.module.state_dict())
                else:
                    best_model_seg = copy.deepcopy(model_seg.state_dict())

                if config['scheduler']['use_scheduler']:
                    scheduler_dict = scheduler.state_dict()
                else:
                    scheduler_dict = {}

                num_no_improvement = 0

                torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_seg,
                            'optimizer_state_dict': g_optimizer_seg.state_dict(),
                            'scheduler_state_dict': scheduler_dict,
                            }, train_path+'model_seg.pth')
                
            else:
                num_no_improvement += 1

            print(f"Epoch: {epoch:.0f}, Loss: {epoch_loss:.10f}, Val Loss: {epoch_loss_val:.10f}, No improvement in {num_no_improvement:.0f} epochs.")
        
        if config['scheduler']['use_scheduler']:
            scheduler.step(epoch_loss_val)
    
        print(f"Current learning rate: {g_optimizer_seg.param_groups[0]['lr']:.8f}")

    sum_writer.close()

def evaluate(data_loader, model_seg, device, pixel_looser, config, thresh=0.5):

    sigmoid = nn.Sigmoid()

    backbone = config['model']['name']
    
    batch_loss = 0

    metrics_batch = []

    for num, data in enumerate(data_loader):
        
        input_data = data['image'].float().to(device)
        mask_data = data['gt'].float().to(device)

        if backbone == 'unetr':
            im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
            pred_comb = model_seg(im_concat)
            mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))

        elif backbone == 'cnn3d' or backbone == 'resunetpp' or backbone == 'maike_cnn3d':
            input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
            mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
            pred_comb = model_seg(input_data)

            if config['train']['binary_gt'] == False:
                mask_data_gauss = mask_data.detach().clone()
                gauss_blur = GaussianBlur(kernel_size=3, sigma=2)

                for i in range(mask_data_gauss.shape[0]):
                    for j in range(mask_data_gauss.shape[2]):
                        mask_data_gauss[i,0,j,:,:] = gauss_blur(mask_data_gauss[i,:,j,:,:])
                        mask_data_gauss[i,0,j,:,:] = torch.where(mask_data_gauss[i,0,j,:,:] > 1, 1, mask_data_gauss[i,0,j,:,:])

        else:
            print('Invalid backbone...')
            sys.exit()

        loss_seg = pixel_looser(pred_comb, mask_data)

        batch_loss = batch_loss + loss_seg.item()

        if config['model']['final_layer'].lower() == 'none':
            pred_comb = sigmoid(pred_comb)
        
        metrics_batch.append(evaluate_onec_slide(pred_comb.cpu().detach().numpy(), mask_data.cpu().detach().numpy(), thresh=thresh))

    input_data_plot = input_data[0][0].cpu().detach().numpy()
    pred_comb_plot = pred_comb[0][0].cpu().detach().numpy()
    mask_data_plot = mask_data[0][0].cpu().detach().numpy()
    
    metrics = np.nanmean(metrics_batch, axis=0)
    epoch_loss = batch_loss/(num+1)

    return metrics, epoch_loss, input_data_plot, pred_comb_plot, mask_data_plot
    
def test():

    mode = 'test'
    config = parse_yml('config.yaml')

    device = config['model']['device']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']
    width_par = config['dataset']['width']

    win_size = config['dataset']['win_size']
    stride = config['dataset']['stride']
    thresh = config['train']['threshold_iou']  

    backbone = config['model']['name']
    data_parallel = config['train']['data_parallel']

    composed = v2.Compose([v2.ToTensor()])

    quick_run = config['dataset']['quick_run']

    data_path = config['dataset']['data_path']
    annotation_path = config['dataset']['annotation_path']

    sigmoid = nn.Sigmoid()
    model_seg = load_model(config, mode)

    model_seg.to(device)
    model_seg.eval()

    dataset = RundifSequence(data_path=data_path,annotation_path=annotation_path,im_transform=composed,mode='test',win_size=win_size,stride=stride,width_par=width_par,include_potential=config['train']['include_potential'],include_potential_gt=config['train']['include_potential_gt'],quick_run=quick_run)

    data_loader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=batch_size,
                                                shuffle=config['test']['shuffle'],
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )    

    indices_test = dataset.img_ids_win

    sigmoid = nn.Sigmoid()

    pred_save = []

    n_ind = len(set(np.array(indices_test).flatten()))

    input_imgs = np.zeros((n_ind, width_par, width_par))
    input_masks = np.zeros((n_ind, width_par, width_par))

    for val in range(n_ind):
        pred_save.append([])
    
    num_batch = 0

    ind_keys = sorted(set(np.array(indices_test).flatten()))
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

        elif backbone == 'cnn3d' or backbone == 'resunetpp' or backbone == 'maike_cnn3d':
            input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
            mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
            pred_comb = model_seg(input_data)

        else:
            print('Invalid backbone...')
            sys.exit()
        
        if config['model']['final_layer'].lower() == 'none':
            pred_comb = sigmoid(pred_comb)

        for b in range(mask_data.shape[0]):
            for k in range(win_size):
                current_ind = ind_dict[indices_test[num_batch][k]]
                pred_save[current_ind].append(pred_comb[b,0,k,:,:].cpu().detach().numpy())

                if np.all(input_imgs[current_ind]) == 0:
                    input_imgs[current_ind] = input_data[b,0,k,:,:].cpu().detach().numpy()
                    input_masks[current_ind] = mask_data[b,0,k,:,:].cpu().detach().numpy()

            num_batch = num_batch + 1
    
    input_imgs = np.array(input_imgs)

    for h, pred_arr in enumerate(pred_save):
        pred_save[h] = np.nanmean(pred_arr,axis=0)

    pred_save = np.array(pred_save)
    pred_save = torch.Tensor(pred_save)
    input_imgs = torch.Tensor(input_imgs)
    input_masks = torch.Tensor(input_masks)

    for t in thresh:
        metrics = evaluate_onec_slide(pred_save.cpu().detach().numpy(),input_masks.cpu().detach().numpy(), thresh=t)

    metrics_path = '/'.join(config['test']['model_path'].split('/')[:-1])

    if not os.path.exists(metrics_path): 
        os.makedirs(metrics_path, exist_ok=True) 

    np.save(metrics_path+'metrics.npy', metrics)
    print(metrics)

if __name__ == "__main__":

    try:
        mode = str(sys.argv[1])
    except IndexError:
        mode = 'train'

    if mode == 'train':
        train()
    
    elif mode == 'test':
        test()