import torch
from torch import nn,optim
import numpy as np
from torchvision.transforms import v2
import sys
import os
from datetime import datetime
import copy
from dataset import RundifSequence, RundifSequenceNew
from evaluation import evaluate_onec_slide, Kappa_cohen, precision_recall, IoU, dice, Accuracy
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
    
    if type(thresh) is not list:
        thresh = [thresh]
    
    backbone = config['model']['name']
    data_parallel = config['train']['data_parallel']

    composed = load_augmentations(config)
    composed_val = v2.Compose([v2.ToTensor()])

    quick_run = config['dataset']['quick_run']

    data_path = config['dataset']['data_path']
    annotation_path = config['dataset']['annotation_path']

    use_cross_validation = config['train']['cross_validation']['use_cross_validation']
    fold_file = config['train']['cross_validation']['fold_file']
    fold_definition = config['train']['cross_validation']['fold_definition']

    try:
        dataloader_type = config['dataset']['dataloader_parameters']['name']
    except KeyError:
        dataloader_type = 'RundifSequence'

    if dataloader_type == "RundifSequenceNew":
        cadence_minutes = config['dataset']['dataloader_parameters']['cadence_minutes']
        seed = config['dataset']['dataloader_parameters']['seed']

        dataset = RundifSequenceNew(data_path=data_path,
                                    annotation_path=annotation_path,
                                    im_transform=composed,
                                    mode='train',
                                    win_size=win_size,
                                    stride=stride,
                                    width_par=width_par,
                                    cadence_minutes=cadence_minutes,
                                    include_potential=config['train']['include_potential'],
                                    use_cross_validation=use_cross_validation,
                                    fold_definition=fold_definition,
                                    quick_run=quick_run,
                                    seed=seed)
        
        dataset_val = RundifSequenceNew(data_path=data_path,
                                        annotation_path=annotation_path,
                                        im_transform=composed_val,
                                        mode='val',
                                        win_size=win_size,
                                        stride=stride,
                                        width_par=width_par,
                                        cadence_minutes=cadence_minutes,
                                        include_potential=config['evaluate']['include_potential'],
                                        use_cross_validation=use_cross_validation,
                                        fold_definition=fold_definition,
                                        quick_run=quick_run,
                                        seed=seed)
        
    else:
        dataset = RundifSequence(data_path=data_path,annotation_path=annotation_path,im_transform=composed,mode='train',win_size=win_size,stride=stride,width_par=width_par,include_potential=config['train']['include_potential'],include_potential_gt=config['train']['include_potential_gt'],quick_run=quick_run,cross_validation=use_cross_validation,fold_file=fold_file,fold_definition=fold_definition)
        dataset_val = RundifSequence(data_path=data_path,annotation_path=annotation_path,im_transform=composed_val,mode='val',win_size=win_size,stride=stride,width_par=width_par,include_potential=config['evaluate']['include_potential'],include_potential_gt=config['evaluate']['include_potential_gt'],quick_run=quick_run,cross_validation=use_cross_validation,fold_file=fold_file,fold_definition=fold_definition)
    
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
        checkpoint = torch.load(config['train']['load_checkpoint']['checkpoint_path'], weights_only=True,map_location='cpu')
        last_epoch = checkpoint['epoch']+1

    else:
        last_epoch = 0

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

    # train_list_ind = [l.tolist() for l in dataset.img_ids_train_win]
    # val_list_ind = [l.tolist() for l in dataset_val.img_ids_val_win]
    # test_list_ind = [l.tolist() for l in dataset.img_ids_test_win]

    # data_indices = {}
    # data_indices["training"] = train_list_ind
    # data_indices["validation"] = val_list_ind
    # data_indices["test"] = test_list_ind

    # with open(train_path+"indices.json", "w") as f:
    #     json.dump(data_indices, f)

    shutil.copy("config.yaml", train_path+"config.yaml")

    best_iou = -1e99
    num_no_improvement = 0

    log_dir = "runs_new/" + folder_path[:-1]

    sum_writer = SummaryWriter(log_dir)
    sigmoid = nn.Sigmoid()


    if config['train']['load_checkpoint']['load_model']:
        total_epochs = num_iter
        print(f"Resuming training from epoch {last_epoch} for a total of {total_epochs} epochs.")

        if total_epochs < last_epoch:
            print(f"Total epochs {total_epochs} is less than last epoch {last_epoch}.")
            sys.exit()

    else:
        total_epochs = num_iter
        print(f"Starting training for {total_epochs} epochs.")
    
    for epoch in range(last_epoch, total_epochs+1):
        model_seg.train()

        epoch_loss = 0
        epoch_loss_val = 0

        batch_loss = 0
        metrics_batch_confusion = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        # metrics_batch = []

        for num, data in enumerate(data_loader):
            g_optimizer_seg.zero_grad()
            
            input_data = data['image'].float().to(device)
            mask_data = data['gt'].float().to(device)
            name_data = data['names']

            if backbone == 'unetr':
                im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
                pred_comb = model_seg(im_concat)
                mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))

            elif backbone == 'cnn3d' or backbone == 'resunetpp' or backbone == 'maike_cnn3d':
                input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
                mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
                name_data = np.transpose(name_data, (1,0))
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

            #metrics_batch.append(evaluate_onec_slide(pred_comb.cpu().detach().numpy(), mask_data.cpu().detach().numpy(), thresh=thresh))
            metrics_confusion = evaluate_onec_slide(pred_comb.cpu().detach().numpy(), mask_data.cpu().detach().numpy(), thresh=thresh)

            metrics_batch_confusion['TP'] = metrics_batch_confusion['TP'] + metrics_confusion['TP']
            metrics_batch_confusion['FP'] = metrics_batch_confusion['FP']+ metrics_confusion['FP']
            metrics_batch_confusion['FN'] = metrics_batch_confusion['FN'] + metrics_confusion['FN']
            metrics_batch_confusion['TN'] = metrics_batch_confusion['TN'] + metrics_confusion['TN']

        train_kapa = Kappa_cohen(metrics_batch_confusion['TP'], metrics_batch_confusion['FP'], metrics_batch_confusion['FN'], metrics_batch_confusion['TN'], (width_par,width_par), (width_par,width_par))
        train_precision, train_recall = precision_recall(metrics_batch_confusion['TP'], metrics_batch_confusion['FP'], metrics_batch_confusion['FN'])
        train_iou = IoU(metrics_batch_confusion['TP'], metrics_batch_confusion['FP'], metrics_batch_confusion['FN'])
        train_acc = Accuracy(metrics_batch_confusion['TP'], metrics_batch_confusion['FP'], metrics_batch_confusion['FN'], metrics_batch_confusion['TN'])

        train_metrics = [train_kapa, train_precision, train_recall, train_iou, train_acc]

        # train_metrics = np.nanmean(metrics_batch, axis=0)
        epoch_loss = batch_loss/(num+1)
        sum_writer.add_scalar("Loss/train", epoch_loss, epoch)

        sum_writer.add_scalar("Precision/train", train_metrics[1], epoch)
        sum_writer.add_scalar("Recall/train", train_metrics[2], epoch)
        sum_writer.add_scalar("IOU/train", train_metrics[3], epoch)
    
        # Prepare the plot
        input_data_train = input_data[0][0].cpu().detach().numpy()
        pred_comb_train = pred_comb[0][0].cpu().detach().numpy()
        mask_data_train = mask_data[0][0].cpu().detach().numpy()
        name_data_train = name_data[0]

        if len(thresh) == 1:
            thresh_plot = thresh[0]
        else:
            thresh_plot = thresh[int(np.ceil(len(thresh)/2))]
            
        thresh_data_train = pred_comb_train.copy()
        thresh_data_train[thresh_data_train >= thresh_plot] = 1
        thresh_data_train[thresh_data_train < thresh_plot] = 0

        plot_images_train = np.concatenate((input_data_train, pred_comb_train, thresh_data_train, mask_data_train), axis=0)
        figure_train = image_grid(plot_images_train, name_data_train)

        # Convert to image and log
        
        sum_writer.add_figure("Training Data", figure_train, global_step=epoch)
        
        optimizer_data.append([epoch, epoch_loss])

        with torch.no_grad():
            model_seg.eval()

            val_metrics, epoch_loss_val, input_data_val, pred_comb_val, mask_data_val, name_data_val = evaluate(data_loader_val, model_seg, device, pixel_looser, config, thresh=thresh)

            sum_writer.add_scalar("Loss/val", epoch_loss_val, epoch)

            optimizer_data_val.append([epoch, epoch_loss_val])
            
            sum_writer.add_scalar("Precision/val", val_metrics[1], epoch)
            sum_writer.add_scalar("Recall/val", val_metrics[2], epoch)
            sum_writer.add_scalar("IOU/val", val_metrics[3], epoch)

            # Prepare the plot
            thresh_data_val = pred_comb_val.copy()
            thresh_data_val[thresh_data_val >= thresh_plot] = 1
            thresh_data_val[thresh_data_val < thresh_plot] = 0

            plot_images_val = np.concatenate((input_data_val, pred_comb_val, thresh_data_val, mask_data_val), axis=0)
            figure_val = image_grid(plot_images_val, name_data_val)

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

def evaluate(data_loader, model_seg, device, pixel_looser, config, thresh=[0.5]):

    sigmoid = nn.Sigmoid()

    backbone = config['model']['name']
    
    batch_loss = 0

    # metrics_batch = []
    metrics_batch_eval_confusion = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    for num, data in enumerate(data_loader):
        
        input_data = data['image'].float().to(device)
        mask_data = data['gt'].float().to(device)
        name_data = data['names']

        if backbone == 'unetr':
            im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
            pred_comb = model_seg(im_concat)
            mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))

        elif backbone == 'cnn3d' or backbone == 'resunetpp' or backbone == 'maike_cnn3d':
            input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
            mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
            pred_comb = model_seg(input_data)
            name_data = np.transpose(name_data, (1,0))

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
        
        # metrics_batch.append(evaluate_onec_slide(pred_comb.cpu().detach().numpy(), mask_data.cpu().detach().numpy(), thresh=thresh))
        
        metrics_confusion = evaluate_onec_slide(pred_comb.cpu().detach().numpy(), mask_data.cpu().detach().numpy(), thresh=thresh)
        metrics_batch_eval_confusion['TP'] += metrics_confusion['TP']
        metrics_batch_eval_confusion['FP'] += metrics_confusion['FP']
        metrics_batch_eval_confusion['FN'] += metrics_confusion['FN']
        metrics_batch_eval_confusion['TN'] += metrics_confusion['TN']

    input_data_plot = input_data[0][0].cpu().detach().numpy()
    pred_comb_plot = pred_comb[0][0].cpu().detach().numpy()
    mask_data_plot = mask_data[0][0].cpu().detach().numpy()
    name_data_plot = name_data[0]

    eval_iou = IoU(metrics_batch_eval_confusion['TP'], metrics_batch_eval_confusion['FP'], metrics_batch_eval_confusion['FN'])
    eval_kapa = Kappa_cohen(metrics_batch_eval_confusion['TP'], metrics_batch_eval_confusion['FP'], metrics_batch_eval_confusion['FN'], metrics_batch_eval_confusion['TN'], input_data_plot.shape, mask_data_plot.shape)
    eval_precision, eval_recall = precision_recall(metrics_batch_eval_confusion['TP'], metrics_batch_eval_confusion['FP'], metrics_batch_eval_confusion['FN'])
    eval_acc = Accuracy(metrics_batch_eval_confusion['TP'], metrics_batch_eval_confusion['FP'], metrics_batch_eval_confusion['FN'], metrics_batch_eval_confusion['TN'])
    eval_metrics = [eval_kapa, eval_precision, eval_recall, eval_iou, eval_acc]

    #metrics = np.nanmean(metrics_batch, axis=0)
    epoch_loss = batch_loss/(num+1)

    return eval_metrics, epoch_loss, input_data_plot, pred_comb_plot, mask_data_plot, name_data_plot

if __name__ == "__main__":

    try:
        mode = str(sys.argv[1])
    except IndexError:
        mode = 'train'

    if mode == 'train':
        train()