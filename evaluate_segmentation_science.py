import numpy as np
import os
from datetime import datetime
from skimage.morphology import disk
from scipy import ndimage
from skimage.measure import label
from skimage import morphology, feature, filters, transform
import astropy.io.fits as fits
from astropy import wcs
from astropy.wcs import FITSFixedWarning
import warnings
import sys
import matplotlib.pyplot as plt
warnings.simplefilter('ignore', category=FITSFixedWarning)
from matplotlib import colors
from utils import parse_yml
from pycocotools import coco
from PIL import Image
from collections import namedtuple
from scipy.optimize import linear_sum_assignment
from utils_evaluation import load_results, post_processing
from evaluation import IoU, precision_recall, dice

def main(mdls, ml_path, date_str, mode='test', methods=['median', 'mean', 'max'], plotting=False, rdif_path=None):

    if plotting and rdif_path is None:
        print('Plotting is enabled, but no path to the differences is provided. Please provide a valid path.')
        sys.exit()
    
    load_paths = []

    thresholds = np.round(np.linspace(0.05, 0.95, 10),2)
    preciscion_arr = np.zeros((len(methods), len(mdls), len(thresholds)))
    recall_arr = np.zeros((len(methods), len(mdls), len(thresholds)))
    IoU_arr = np.zeros((len(methods), len(mdls), len(thresholds)))
    dice_arr = np.zeros((len(methods), len(mdls), len(thresholds)))


    for mdl in mdls:
        for method in methods:
            load_paths.append(ml_path + mdl + '/segmentation_masks_'+method+'_'+mode+'.npz')

    for path_idx, load_path in enumerate(load_paths):
        print('Working on model: ', load_path)
        filenames, pred, gt = load_results(load_path, load_gt=True)

        mdl_idx = mdls.index(load_path.split('/')[-2])
        method_idx = methods.index(load_path.split('_')[-2])

        # iou_temp = []
        # prec_temp = []
        # rec_temp = []
        # dice_temp = []

        for t_idx, t in enumerate(thresholds):
            print('Threshold: ', t)
            TP_all = 0
            FP_all = 0
            FN_all = 0
            pred_processed = post_processing(pred, t=t, return_labeled=False)

            for i in range(len(pred_processed)):

                TP_all = TP_all + np.sum(np.logical_and(pred_processed[i] == 1, gt[i] == 1))
                FP_all = FP_all + np.sum(np.logical_and(pred_processed[i] == 1, gt[i] == 0))
                FN_all = FN_all + np.sum(np.logical_and(pred_processed[i] == 0, gt[i] == 1))

                # TP = np.sum(np.logical_and(pred_processed[i] == 1, gt[i] == 1))
                # FP = np.sum(np.logical_and(pred_processed[i] == 1, gt[i] == 0))
                # FN = np.sum(np.logical_and(pred_processed[i] == 0, gt[i] == 1))

                # iou_temp.append(IoU(TP, FP, FN))
                # prec, rec = precision_recall(TP, FP, FN)
                # prec_temp.append(prec)
                # rec_temp.append(rec)
                # dice_temp.append(dice(TP, FP, FN))

            IoU_arr[method_idx, mdl_idx, t_idx] = IoU(TP_all, FP_all, FN_all)
            prec_temp, rec_temp = precision_recall(TP_all, FP_all, FN_all)
            preciscion_arr[method_idx, mdl_idx, t_idx] = prec_temp
            recall_arr[method_idx, mdl_idx, t_idx] = rec_temp
            dice_arr[method_idx, mdl_idx, t_idx] = dice(TP_all, FP_all, FN_all)
            
            # IoU_arr[method_idx, mdl_idx, t_idx] = np.nanmean(iou_temp)
            # preciscion_arr[method_idx, mdl_idx, t_idx] = np.nanmean(prec_temp)
            # recall_arr[method_idx, mdl_idx, t_idx] = np.nanmean(rec_temp)
            # dice_arr[method_idx, mdl_idx, t_idx] = np.nanmean(dice_temp)


    iou_final = np.zeros((len(methods), len(thresholds)))
    precision_final = np.zeros((len(methods), len(thresholds)))
    recall_final = np.zeros((len(methods), len(thresholds)))
    dice_final = np.zeros((len(methods), len(thresholds)))

    for method_idx, method in enumerate(methods):
        iou_final[method_idx] = np.nanmean(IoU_arr[method_idx], axis=0)
        precision_final[method_idx] = np.nanmean(preciscion_arr[method_idx], axis=0)
        recall_final[method_idx] = np.nanmean(recall_arr[method_idx], axis=0)
        dice_final[method_idx] = np.nanmean(dice_arr[method_idx], axis=0)

    best_thresholds = np.zeros(len(mdls))

    max_ious = np.zeros(len(methods))
    max_precisions = np.zeros(len(methods))
    max_recalls = np.zeros(len(methods))
    max_dices = np.zeros(len(methods))

    folder_save = ml_path + '/results_science/' + date_str + '/'

    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    with open(folder_save+'segmentation_results_science.txt' , 'w') as f:
        print('Models used: ', mdls,file=f)
        print('\n',file=f)
        for i in range(iou_final.shape[0]):
            max_idx = np.argmax(iou_final[i])

            max_ious[i] = np.max(iou_final[i])
            max_precisions[i] = precision_final[i][max_idx]
            max_recalls[i] = recall_final[i][max_idx]
            max_dices[i] = dice_final[i][max_idx]

            print('Method: ', methods[i],file=f)
            print('Best Threshold', thresholds[max_idx],file=f)

            print('Best Precision: ', np.round(precision_final[i][max_idx],2), u'\u00B1' + str( ), np.round(np.std(preciscion_arr[i,:,max_idx]),2),file=f)
            print('Best Recall: ', np.round(recall_final[i][max_idx],2), u'\u00B1' + str( ), np.round(np.std(recall_arr[i,:,max_idx]),2),file=f)
            print('Best IoU: ', np.round(np.max(iou_final[i]),2), u'\u00B1' + str( ), np.round(np.std(IoU_arr[i,:,max_idx]),2),file=f)
            print('Best Dice: ', np.round(dice_final[i][max_idx],2), u'\u00B1' + str( ), np.round(np.std(dice_arr[i,:,max_idx]),2),file=f)

            # sum_metrics[i] = np.round(iou_final[i, max_idx] + precision_final[i, max_idx] + recall_final[i, max_idx] + dice_final[i, max_idx],2)
            best_thresholds[i] = thresholds[max_idx]
            print('\n',file=f)
        
        best_method_idx = np.argwhere(max_ious == np.amax(max_ious)).flatten()

        if len(best_method_idx) > 1:
            sum_metrics = np.zeros(len(best_method_idx))

            for can_idx, best_candidate in enumerate(best_method_idx):
                sum_metrics[can_idx] = max_precisions[best_candidate] + max_recalls[best_candidate] + max_dices[best_candidate]
            
            best_method_idx = np.argwhere(sum_metrics == np.amax(sum_metrics)).flatten()[0]

        else:
            best_method_idx = best_method_idx[0]

        print('------------------------------------------------------',file=f)
        print('Best Method:', methods[best_method_idx],file=f)
        print('Best Threshold:', best_thresholds[best_method_idx],file=f)
        
        best_method = methods[best_method_idx]
        best_threshold = best_thresholds[best_method_idx]
        best_threshold_idx = np.argwhere(thresholds == best_threshold).flatten()[0]
        best_mdl = mdls[np.argmax(IoU_arr[best_method_idx,:,best_threshold_idx])]

    if plotting:

        geo_green  = (5/255, 46/255, 55/255)
        geo_lime  = (191/255,206/255,64/255)
        geo_magenta     = (140/255,17/255,170/255)
        clrs = [geo_green, geo_lime, geo_magenta]
        mrkrs = ['o', 'v', '*']

        SMALL_SIZE = 8
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 16

        MARKER_SIZE = 9


        # Create a figure with four subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the Precision-Recall curve
        axes[0].plot(recall_final[0], precision_final[0], marker=mrkrs[2], linestyle='-', color=clrs[2], label='Median', markersize=MARKER_SIZE)
        axes[0].plot(recall_final[1], precision_final[1], marker=mrkrs[1], linestyle='-', color=clrs[1], label='Mean', markersize=MARKER_SIZE)
        axes[0].plot(recall_final[2], precision_final[2], marker=mrkrs[0], linestyle='-', color=clrs[0], label='Maximum', markersize=MARKER_SIZE)

        axes[0].set_xlabel('Recall', fontsize=BIGGER_SIZE)
        axes[0].set_ylabel('Precision', fontsize=BIGGER_SIZE)
        axes[0].set_title(f"Precision-Recall Curve", fontsize=BIGGER_SIZE)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        axes[0].grid(True)
        axes[0].tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)

        # Plot the IoU curve
        axes[1].plot(thresholds, iou_final[0], marker=mrkrs[2], linestyle='-', color=clrs[2], label='Median', markersize=MARKER_SIZE)
        axes[1].plot(thresholds, iou_final[1], marker=mrkrs[1], linestyle='-', color=clrs[1], label='Mean', markersize=MARKER_SIZE)
        axes[1].plot(thresholds, iou_final[2], marker=mrkrs[0], linestyle='-', color=clrs[0], label='Maximum', markersize=MARKER_SIZE)

        axes[1].set_xlabel('Threshold', fontsize=BIGGER_SIZE)
        axes[1].set_ylabel('IoU', fontsize=BIGGER_SIZE)
        axes[1].set_title(f"IoU Curve", fontsize=BIGGER_SIZE)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 0.6])
        axes[1].grid(True)
        axes[1].tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)

        # Plot the Dice curve
        axes[2].plot(thresholds, dice_final[0], marker=mrkrs[2], linestyle='-', color=clrs[2], label='Median', markersize=MARKER_SIZE)
        axes[2].plot(thresholds, dice_final[1], marker=mrkrs[1], linestyle='-', color=clrs[1], label='Mean', markersize=MARKER_SIZE)
        axes[2].plot(thresholds, dice_final[2], marker=mrkrs[0], linestyle='-', color=clrs[0], label='Maximum', markersize=MARKER_SIZE)

        axes[2].set_xlabel('Threshold', fontsize=BIGGER_SIZE)
        axes[2].set_ylabel('Dice Coefficient', fontsize=BIGGER_SIZE)
        axes[2].set_title(f"Dice Curve", fontsize=BIGGER_SIZE)
        axes[2].set_xlim([0, 1])
        axes[2].set_ylim([0,0.6])
        axes[2].grid(True)
        axes[2].tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)

        # Add one common legend outside the plot
        lines_labels = axes[0].get_legend_handles_labels()
        lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=BIGGER_SIZE, frameon=False)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(folder_save+'segmentation_curves_science.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        #plt.show()

        # best_paths = []
        best_path = ml_path + best_mdl + '/segmentation_masks_'+best_method+'_'+mode+'.npz'
        # for mdl in mdls:
        #     best_paths.append(ml_path + mdl + '/segmentation_masks_'+best_method+'_'+mode+'.npz')

        # my_path = best_paths[0]

        filenames, pred, gt = load_results(best_path, load_gt=True)
        filenames = [rdif_path+f.decode('utf-8') for f in filenames]

        pred_processed = post_processing(pred, t=best_threshold, return_labeled=False)

        TP_img = np.logical_and(pred_processed == 1, gt == 1).astype(int)
        FP_img = np.logical_and(pred_processed == 1, gt == 0).astype(int)
        FN_img = np.logical_and(pred_processed == 0, gt ==  1).astype(int)

        imgs = []

        for i in range(len(filenames)):
            
            try:
                imgs.append(transform.resize(np.load(filenames[i]), (128,128)))

            except FileNotFoundError:
                imgs.append(np.zeros((128,128)))


        # Create a grid of subplots with r rows and c columns

        c_black = colors.colorConverter.to_rgba('black',alpha=0)
        c_orange= colors.colorConverter.to_rgba('#ff6100',alpha=1)
        c_pink= colors.colorConverter.to_rgba('#dc2580',alpha=1)
        c_purple= colors.colorConverter.to_rgba('#785ef1',alpha=1)

        cmap_fn = colors.ListedColormap([c_black,c_orange])
        cmap_fp = colors.ListedColormap([c_black,c_pink])
        cmap_tp = colors.ListedColormap([c_black,c_purple])

        r = 2
        c = 4
        win_size = int(r*c)
        gap = 3
        indices = np.arange(0, len(imgs)+1, win_size)

        # wspace = 0
        # hspace = 0
        fac = 2

        for init in indices:

            plot_images = imgs[init:init+win_size*3][::gap]
            plt_tp = TP_img[init:init+win_size*3][::gap]
            plt_fp = FP_img[init:init+win_size*3][::gap]
            plt_fn = FN_img[init:init+win_size*3][::gap]

            if np.any(plt_tp):
                fig,axes = plt.subplots(r,c, figsize=(c*2,r*2))
                fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

                # Iterate over the images and plot them in the subplots
                for i, image in enumerate(plot_images):
                    # Calculate the row and column indices for the current image
                    row = i // c
                    col = i % c
                    axes[row, col].imshow(np.flipud(image), aspect='equal', cmap='gray', interpolation='none', vmin=np.nanmedian(image)-fac*np.nanstd(image), vmax=np.nanmedian(image)+fac*np.nanstd(image))
                    axes[row, col].imshow(np.flipud(plt_tp[i]), cmap=cmap_tp, alpha=0.5, interpolation='none')
                    axes[row, col].imshow(np.flipud(plt_fp[i]), cmap=cmap_fp, alpha=0.5, interpolation='none')
                    axes[row, col].imshow(np.flipud(plt_fn[i]), cmap=cmap_fn, alpha=0.5, interpolation='none')
                    axes[row, col].axis('off')
                    
                    # Add text in the top middle of each image (not as a title)
                    axes[row, col].text(
                        0.5, 0.05, 
                        f'{filenames[init+i].split("/")[-1].split(".")[0][:15]}', 
                        fontsize=10, 
                        color='white', 
                        ha='center', 
                        va='bottom', 
                        transform=axes[row, col].transAxes,
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
                    )
                    axes[row,col].set_frame_on(False)

                # Adjust the spacing between subplots
                folder_save_imgs = folder_save + 'segmentation_imgs_' + best_mdl + '/'

                if not os.path.exists(folder_save_imgs):
                    os.makedirs(folder_save_imgs)
                
                plt.savefig(folder_save_imgs+filenames[init].split('/')[-1].split('.')[0] + '.png', dpi=300, bbox_inches='tight',pad_inches=0.0)
                plt.close()

if __name__ == "__main__":

    config = parse_yml('config_evaluation.yaml')

    mode = config['mode']
    mdls_event_based = config['mdls_event_based']
    ml_path = config['paths']['ml_path']

    methods = config['method']
    plotting = config['plotting']

    rdif_path = config['paths']['rdif_path']

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    
    main(mdls=mdls_event_based, ml_path=ml_path, date_str=date_str, mode=mode, methods=methods, plotting=plotting, rdif_path=rdif_path)