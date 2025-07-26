import torch
from torch import nn
import numpy as np
import sys
from dataset import RundifSequence, RundifSequenceNew
from utils import load_model, parse_yml
from torchvision.transforms import v2


def main(mdls, ml_path, mode='test'):

    model_paths = []
    config_paths = []
    sigmoid = nn.Sigmoid()

    for mdl in mdls:

        model_paths.append(ml_path + mdl + '/model_seg.pth')
        config_paths.append(ml_path + mdl + '/config.yaml')

    for cf_idx, model_path in enumerate(model_paths):

        config = parse_yml(config_paths[cf_idx])

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

        data_path = config['dataset']['data_path']
        annotation_path = config['dataset']['annotation_path']

        use_cross_validation = config['train']['cross_validation']['use_cross_validation']
        fold_file = config['train']['cross_validation']['fold_file']
        fold_definition = config['train']['cross_validation']['fold_definition']

        quick_run = config['dataset']['quick_run']

        try:
            dataloader_type = config['dataset']['dataloader_parameters']['name']
        except KeyError:
            dataloader_type = 'RundifSequence'

        model_seg = load_model(config, 'test', model_path)

        model_seg.to(device)
        model_seg.eval()

        if dataloader_type == "RundifSequenceNew":
            cadence_minutes = config['dataset']['dataloader_parameters']['cadence_minutes']
            seed = config['dataset']['dataloader_parameters']['seed']

            dataset = RundifSequenceNew(data_path=data_path,
                                        annotation_path=annotation_path,
                                        im_transform=composed,
                                        mode=mode,
                                        win_size=win_size,
                                        stride=stride,
                                        width_par=width_par,
                                        cadence_minutes=cadence_minutes,
                                        include_potential=config['train']['include_potential'],
                                        use_cross_validation=use_cross_validation,
                                        fold_definition=fold_definition,
                                        quick_run=quick_run,
                                        seed=seed)
        else:
        
            dataset = RundifSequence(data_path=data_path,annotation_path=annotation_path,im_transform=composed,mode=mode,win_size=win_size,stride=stride,width_par=width_par,include_potential=config['train']['include_potential'],include_potential_gt=config['train']['include_potential_gt'],quick_run=False,cross_validation=use_cross_validation,fold_file=fold_file,fold_definition=fold_definition)

        data_loader = torch.utils.data.DataLoader(
                                                    dataset,
                                                    batch_size=batch_size,
                                                    shuffle=config['test']['shuffle'],
                                                    num_workers=num_workers,
                                                    pin_memory=False
                                                )    

        indices_test = dataset.img_ids_win

        clean_set = set(np.array(indices_test).flatten())
        clean_set.discard(None)  # Remove None values if they exist

        pred_save = []

        n_ind = len(clean_set)

        input_imgs = np.zeros((n_ind, width_par, width_par))
        input_masks = np.zeros((n_ind, width_par, width_par))

        input_names = np.chararray((n_ind), itemsize=25)

        for val in range(n_ind):
            pred_save.append([])
        
        num_batch = 0

        ind_keys = sorted(clean_set)
        ind_set = np.arange(0, len(ind_keys))
        ind_dict = {}

        for A, B in zip(ind_keys, ind_set):
            ind_dict[A] = B

        num_batch = 0

        for num, data in enumerate(data_loader):
            print(f'Processing: {num} of {dataset.__len__()}')

            input_data = data['image'].float().to(device)
            mask_data = data['gt'].float().to(device)
            name_data = np.array(data['names'])

            if backbone == 'unetr':
                
                im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
                pred_comb = model_seg(im_concat)
                mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
                input_data = torch.permute(input_data, (0, 2, 1, 3, 4))

            elif backbone == 'cnn3d' or backbone == 'resunetpp' or backbone == 'maike_cnn3d':
                
                try:
                    input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
                    mask_data = torch.permute(mask_data, (0, 2, 1, 3, 4))
                    name_data = np.transpose(name_data)

                    pred_comb = model_seg(input_data)

                    if config['model']['final_layer'].lower() == 'none':
                        pred_comb = sigmoid(pred_comb)

                except RuntimeError:
                    continue
                    
            else:
                print('Invalid backbone...')
                sys.exit()

            
            for b in range(pred_comb.shape[0]):
                for k in range(pred_comb.shape[2]):
                    
                    try:
                        current_ind = ind_dict[indices_test[num_batch][k]]

                        pred_save[current_ind].append(pred_comb[b,0,k,:,:].cpu().detach().numpy())

                        if np.all(input_imgs[current_ind]) == 0:
                            input_imgs[current_ind] = input_data[b,0,k,:,:].cpu().detach().numpy()
                            input_names[current_ind] = name_data[b,k]
                            input_masks[current_ind] = mask_data[b,0,k,:,:].cpu().detach().numpy()

                    except KeyError:
                        continue

                num_batch = num_batch + 1

        input_imgs = np.array(input_imgs)

        pred_final = np.zeros((n_ind, width_par, width_par))
        pred_final_mean = np.zeros((n_ind, width_par, width_par))
        pred_final_median = np.zeros((n_ind, width_par, width_par))
        pred_final_max = np.zeros((n_ind, width_par, width_par))

        for h, pred_arr in enumerate(pred_save):
            pred_prep = np.zeros((len(pred_arr), width_par, width_par))
            
            for j in range(len(pred_arr)):
                pred_prep[j] = pred_arr[j]

            pred_final[h] = np.nanmedian(pred_prep, axis=0)
            pred_final_mean[h] = np.nanmean(pred_prep, axis=0)
            pred_final_median[h] = np.nanmedian(pred_prep, axis=0)
            pred_final_max[h] = np.nanmax(pred_prep, axis=0)

        save_path = ml_path + mdls[cf_idx] + '/'

        np.savez_compressed(save_path+"segmentation_masks_median_"+mode+".npz", masks=pred_final_median.astype(np.float16), gt=input_masks.astype(int), filenames=input_names)
        np.savez_compressed(save_path+"segmentation_masks_mean_"+mode+".npz", masks=pred_final_mean.astype(np.float16), gt=input_masks.astype(int), filenames=input_names)
        np.savez_compressed(save_path+"segmentation_masks_max_"+mode+".npz", masks=pred_final_max.astype(np.float16), gt=input_masks.astype(int), filenames=input_names)

if __name__ == '__main__':

    config = parse_yml('config_evaluation.yaml')

    mode = config['mode']
    mdls_event_based = config['mdls_event_based']
    ml_path = config['paths']['ml_path']

    main(mdls=mdls_event_based, ml_path=ml_path, mode=mode)
