import torch
from torch import nn
import numpy as np
import sys
from dataset import RundifSequence_Test
from utils import load_model, parse_yml
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import math
import os

def plot_tracking_grid_gap(
    input_dates_plot,
    input_images,
    fronts_by_date,
    fronts_by_date_gt,
    save_path,
    gap=1
):

    os.makedirs(save_path,exist_ok=True)

    al = 1

    c_pink = mpl.colors.colorConverter.to_rgba('#dc2580', alpha=al)      # FP (segmentation + tracking)
    c_purple = mpl.colors.colorConverter.to_rgba('#785ef1', alpha=al)    # TP (segmentation + tracking)
    c_orange = mpl.colors.colorConverter.to_rgba('#ff6100',alpha=al)     # FN
    c_tp_gt = mpl.colors.colorConverter.to_rgba('#2f00ff',alpha=al)      # GT

    num_per_fig = 8
    num_cols = 4
    num_rows = 2

    # Apply the gap to select images and dates
    sampled_indices = list(range(0, len(input_dates_plot), gap))
    sampled_dates = [input_dates_plot[i] for i in sampled_indices if i < len(input_dates_plot)]
    sampled_images = [input_images[i] for i in sampled_indices if i < len(input_images)]

    total = len(sampled_dates)
    num_figs = math.ceil(total / num_per_fig)
    figsize = (num_cols * 2, num_rows * 2)

    for i in range(num_figs):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

        axes = axes.flatten()

        for j in range(num_per_fig):
            idx = i * num_per_fig + j
            if idx >= total:
                axes[j].axis('off')
                continue

            dt = sampled_dates[idx]
            img = sampled_images[idx]
        
            fronts = fronts_by_date.get(dt, [])
            fronts_gt = fronts_by_date_gt.get(dt, [])

            ax = axes[j]
            ax.imshow(np.flipud(img), aspect='equal', cmap='gray', vmin=np.nanmedian(img)-0.4*np.nanstd(img), vmax=np.nanmedian(img)+0.4*np.nanstd(img),extent=(0, img.shape[1], img.shape[0], 0))


            # Tracking overlays: plot fronts
            for front in fronts:
                img_dim = img.shape[0]
                xcoords = front['x_coords']
                ycoords = front['y_coords']
                label = front['label']
                color = c_purple if label == "TP" else c_pink
                ax.scatter(xcoords, img_dim-ycoords, s=10, color=color, label=label, alpha=0.9, marker='x')

            for front in fronts_gt:
                img_dim = img.shape[0]
                xcoords = front['x_coords']
                ycoords = front['y_coords']
                label = front['label']
                color = c_orange if label == "FN" else c_tp_gt
                ax.scatter(xcoords, img_dim-ycoords, s=25, color=color, label=label, alpha=0.9, marker='*', edgecolors='face',linewidths=2) 


            # Add text in the top middle of each image (not as a title)
            ax.text(
                0.5, 0.05, 
                datetime.datetime.strftime(dt, '%Y%m%d_%H%M%S'), 
                fontsize=10, 
                color='white', 
                ha='center', 
                va='bottom', 
                transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
            )
            #ax.text(.01, .99, dt.strftime('%Y-%m-%d %H:%M'), ha='left', va='top', fontsize=8, transform=ax.transAxes)
            
            #ax.set_title(dt.strftime('%Y-%m-%d %H:%M'), fontsize=8)
            ax.axis('off')
            ax.set_frame_on(False)

        #plt.tight_layout()
        plt.savefig(save_path+ 'tracking_segmentation_' + datetime.datetime.strftime(dt, '%Y%m%d_%H%M%S') + '.jpg', dpi=300, bbox_inches='tight',pad_inches=0.0)
        #plt.show()
        plt.close()

def main(mdl, ml_path, timepairs, data_paths, best_method):

    sigmoid = nn.Sigmoid()
    model_path = ml_path + mdl + '/model_seg.pth'
    config_path = ml_path + mdl + '/config.yaml'

    for pair_idx, pair in enumerate(timepairs):

        config = parse_yml(config_path)
        batch_size = config['train']['batch_size']
        num_workers = config['train']['num_workers']
        backbone = config['model']['name']
        

        device = config['model']['device']
        width_par = config['dataset']['width']
        win_size = config['dataset']['win_size']
        stride = config['dataset']['stride']

        model_seg = load_model(config, 'test', model_path)

        model_seg.to(device)
        dataset = RundifSequence_Test(data_path=data_paths[pair_idx],pair=pair,win_size=win_size,stride=stride,width_par=width_par)

        data_loader = torch.utils.data.DataLoader(
                                                    dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=False
                                                )

        test_list_names = np.array([l.tolist() for l in dataset.files_strided])

        pred_save = []

        n_val = len(set(test_list_names.flatten()))

        input_imgs = np.zeros((n_val, width_par, width_par))
        input_names = np.chararray((n_val), itemsize=len(test_list_names[0][0]))

        for val in range(n_val):
            pred_save.append([])

        model_seg.eval()

        ind_keys = sorted(set(np.array(test_list_names).flatten()))
        ind_set = np.arange(0, len(ind_keys))
        ind_dict = {}

        for A, B in zip(ind_keys, ind_set):
            ind_dict[A] = B

        num_batch = 0

        for num, data in enumerate(data_loader):
            print(f'Processing: {num} of {dataset.__len__()}')
            input_data = data['image'].float().to(device)
            name_data = np.array(data['names'])

            if backbone == 'unetr':
                
                im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
                pred_comb = model_seg(im_concat)
                input_data = torch.permute(input_data, (0, 2, 1, 3, 4))

            elif backbone == 'cnn3d' or backbone == 'resunetpp' or backbone == 'maike_cnn3d':

                input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
                name_data = np.transpose(name_data)

                pred_comb = model_seg(input_data)

                if config['model']['final_layer'].lower() == 'none':
                    pred_comb = sigmoid(pred_comb)
                    
            else:
                print('Invalid backbone...')
                sys.exit()

            
            for b in range(pred_comb.shape[0]):
                for k in range(win_size):

                    current_ind = ind_dict[test_list_names[num_batch][k]]

                    pred_save[current_ind].append(pred_comb[b,0,k,:,:].cpu().detach().numpy())

                    if np.all(input_imgs[current_ind]) == 0:
                        input_imgs[current_ind] = input_data[b,0,k,:,:].cpu().detach().numpy()
                        input_names[current_ind] = name_data[b,k]

                num_batch = num_batch + 1

        input_imgs = np.array(input_imgs)

        pred_final = np.zeros((n_val, width_par, width_par))

        for h, pred_arr in enumerate(pred_save):
            pred_prep = np.zeros((len(pred_arr), width_par, width_par))
            
            for j in range(len(pred_arr)):
                pred_prep[j] = pred_arr[j]

            if best_method == 'mean':
                pred_final[h] = np.nanmean(pred_prep, axis=0)
            elif best_method == 'max':
                pred_final[h] = np.nanmax(pred_prep, axis=0)
            elif best_method == 'median':
                pred_final[h] = np.nanmedian(pred_prep, axis=0)
            else:
                print('Invalid mode selected...')
                sys.exit()

        save_path = ml_path + mdl + '/'

        np.savez_compressed(save_path+"segmentation_masks_"+best_method+"_"+pair['start']+"_"+pair['end']+".npz", masks=pred_final.astype(np.float16), filenames=input_names)
                
if __name__ == '__main__':

    config = parse_yml('config_evaluation.yaml')

    mdls_operational = config['mdls_operational']

    ml_path = config['paths']['ml_path']
    data_paths = config['paths']['data_paths']

    time_pairs = config['time_pairs']
    timepairs = [{'start': time_pairs['start'][i], 'end': time_pairs['end'][i]} for i in range(len(time_pairs['start']))]

    date_str = None

    if date_str is not None:
            
        best_segmentation_path = ml_path + 'results_science/'+date_str+'/segmentation_results_science.txt'

        with open(best_segmentation_path, 'r') as f:
            lines = f.readlines()

        best_method = lines[-2].split('Best Method:')[-1].rstrip().strip(' ')
        best_threshold = float(lines[-1].split('Best Threshold:')[-1].rstrip().strip(' '))

    else:
        best_method = 'mean'
        best_threshold = 0.5

    main(mdl=mdls_operational, ml_path=ml_path, timepairs=timepairs, data_paths=data_paths, best_method=best_method)

