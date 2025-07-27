import numpy as np
import os
import datetime
from skimage import transform
from astropy.wcs import FITSFixedWarning
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.simplefilter('ignore', category=FITSFixedWarning)
from collections import namedtuple
from scipy.optimize import linear_sum_assignment
import math
from utils_evaluation import load_results, evaluate_matches
from utils import parse_yml

def collect_tracking_fronts_by_date(input_dates_plot, cme_dictionary, matches, unmatched):
    """
    Organizes predicted CME fronts by timestamp and tags each as TP or FP for tracking.

    Parameters:
    - input_dates_plot: list of datetime objects to include
    - cme_dictionary: dict of predicted CMEs with CME name as key and:
        - 'times': list of timestamps
        - 'cme_front_pixels': list of (y_coords, x_coords) pairs
    - matches: dict with CME name as key and match details (implies TP)
    - unmatched: list of CME names (implies FP)

    Returns:
    - fronts_by_date: dict where each key is a datetime in input_dates_plot,
                      and the value is a list of (cme_name, x_coords, y_coords, code)
                      with code ∈ {'TP', 'FP'}
    """

    fronts_by_date = {ts: [] for ts in input_dates_plot}

    for cme_name, cme_data in cme_dictionary.items():
        times = cme_data.get('times', [])
        fronts = cme_data.get('cme_front_pixels', [])
        if cme_name in matches:
            code = "TP"
        elif cme_name in unmatched:
            code = "FP"
        else:
            continue  # CME was not evaluated, skip it

        for idx, ts in enumerate(times):
            if ts not in fronts_by_date:
                continue

            if idx >= len(fronts):
                continue  # Safety check

            y_coords, x_coords = fronts[idx]

            # Only add if front has actual pixels
            if len(x_coords) == 0 or len(y_coords) == 0:
                continue

            fronts_by_date[ts].append({'name':cme_name, 'x_coords':x_coords, 'y_coords':y_coords,'label':code})

    return fronts_by_date

def collect_segmentation_results_by_date(input_dates_plot, cme_dictionary, ground_truth_dictionary):
    """
    For each date, computes segmentation TP, FP, FN masks.
    
    Parameters:
    - input_dates_plot: list of datetime objects
    - cme_dictionary: predicted CMEs with 'time' and 'cme_mask' (list of 128x128 binary arrays)
    - ground_truth_dictionary: GT CMEs with same structure
    
    Returns:
    - segmentation_results_by_date: dict with each date as key and value as:
        {
            'tp': binary_mask,
            'fp': binary_mask,
            'fn': binary_mask
        }
    """
    segmentation_results_by_date = {}

    for current_time in input_dates_plot:
        pred_mask = np.zeros((128, 128), dtype=int)
        gt_mask = np.zeros((128, 128), dtype=int)

        # Aggregate predicted masks for this timestep
        for cme in cme_dictionary.values():
            if current_time in cme.get('times', []):
                idx = cme['times'].index(current_time)

                if idx < len(cme['areas']):
                    pred_mask = np.logical_or(pred_mask,cme['areas'][idx])

        # Aggregate ground truth masks for this timestep
        for cme in ground_truth_dictionary.values():
            if current_time in cme.get('times', []):
                idx = cme['times'].index(current_time)
                if idx < len(cme['areas']):
                    gt_mask = np.logical_or(gt_mask,cme['areas'][idx])

        tp = np.logical_and(pred_mask,gt_mask)
        fp = np.logical_and(pred_mask,np.logical_not(gt_mask))
        fn = np.logical_and(gt_mask,np.logical_not(pred_mask))

        segmentation_results_by_date[current_time] = {
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return segmentation_results_by_date

def match_cmes_unique(ground_truth_cmes, predicted_cmes, overlap_threshold=0.25, iou_threshold=0.1):
    Range = namedtuple('Range', ['start', 'end'])
    pred_names = list(predicted_cmes.keys())
    gt_names = list(ground_truth_cmes.keys())

    num_pred = len(pred_names)
    num_gt = len(gt_names)

    cost_matrix = np.full((num_pred, num_gt), 1e10)
    match_info = {}  # (pred_idx, gt_idx) -> (start_err, end_err, total_err, iou)

    for i, pred_name in enumerate(pred_names):
        pred_data = predicted_cmes[pred_name]
        pred_times = np.array(pred_data["times"])
        pred_range = Range(start=pred_times[0], end=pred_times[-1])

        for j, gt_name in enumerate(gt_names):
            gt_data = ground_truth_cmes[gt_name]
            gt_times = np.array(gt_data["times"])
            gt_range = Range(start=gt_times[0], end=gt_times[-1])

            latest_start = max(pred_range.start, gt_range.start)
            earliest_end = min(pred_range.end, gt_range.end)
            earliest_start = min(pred_range.start, gt_range.start)
            latest_end = max(pred_range.end, gt_range.end)

            overlap = max(0, (earliest_end - latest_start).total_seconds())
            total_duration = (latest_end - earliest_start).total_seconds()
            overlap_fraction = overlap / total_duration if total_duration > 0 else 0

            if overlap_fraction < overlap_threshold:
                continue

            intersect_times, pred_inds, gt_inds = np.intersect1d(pred_times, gt_times, return_indices=True)
            
            if len(intersect_times) == 0:
                continue

            ious = []
            unique_times_pred = pred_times[~np.in1d(pred_times,gt_times)]

            if len(unique_times_pred) > 0:
                for pi in unique_times_pred:
                    ious.append(0)

            tp_all = 0
            fp_all = 0
            fn_all = 0

            for pi, gi in zip(pred_inds, gt_inds):
                pred_mask = pred_data["areas"][pi]
                gt_mask = gt_data["areas"][gi]

                tp = np.sum(np.logical_and(pred_mask, gt_mask))
                fp = np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))
                fn = np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))

                tp_all += tp
                fp_all += fp
                fn_all += fn

                # denom = tp + fp + fn
                # iou = tp / denom if denom != 0 else 0
                # ious.append(iou)

            denom = tp_all + fp_all + fn_all
            if denom == 0:
                mean_iou = np.nan
            else:
                mean_iou = tp_all / denom

            #mean_iou = np.nanmean(ious)

            if np.isnan(mean_iou) or mean_iou < iou_threshold:
                continue

            start_err = abs((gt_range.start - pred_range.start).total_seconds()) / 3600
            end_err = abs((gt_range.end - pred_range.end).total_seconds()) / 3600
            total_err = start_err + end_err

            cost_matrix[i, j] = total_err
            match_info[(i, j)] = {
                "gt_match": gt_name,
                "start_error": start_err,
                "end_error": end_err,
                "total_error": total_err,
                "mean_iou": mean_iou
            }

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] == 1e10:
            continue  # skip invalid match
        pred_name = pred_names[r]
        matches[pred_name] = match_info[(r, c)]
        matched_pred.add(pred_name)
        matched_gt.add(gt_names[c])

    unmatched_gt = [gt for gt in gt_names if gt not in matched_gt]
    unmatched_pred = [pred for pred in pred_names if pred not in matched_pred]

    return matches, unmatched_gt, unmatched_pred

def plot_segmentation_and_tracking_grid_gap(
    input_dates_plot,
    input_images,
    segmentation_results_by_date,
    fronts_by_date,
    save_path,
    gap=1,
    img_size = 128
):

    os.makedirs(save_path, exist_ok=True)

    al = 1
    c_black = mpl.colors.colorConverter.to_rgba('#000000', alpha=0)
    c_orange = mpl.colors.colorConverter.to_rgba('#ff6100', alpha=al)  # FN (segmentation)
    c_pink = mpl.colors.colorConverter.to_rgba('#dc2580', alpha=al)    # FP (segmentation + tracking)
    c_purple = mpl.colors.colorConverter.to_rgba('#785ef1', alpha=al)  # TP (segmentation + tracking)

    cmap_tp = mpl.colors.ListedColormap([c_black, c_purple])
    cmap_fp = mpl.colors.ListedColormap([c_black, c_pink])
    cmap_fn = mpl.colors.ListedColormap([c_black, c_orange])

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
            seg_res = segmentation_results_by_date.get(
                dt,
                {'tp': np.zeros_like(img), 'fp': np.zeros_like(img), 'fn': np.zeros_like(img)}
            )
            fronts = fronts_by_date.get(dt, [])

            ax = axes[j]
            ax.imshow(np.flipud(img), aspect='equal', cmap='gray', vmin=np.nanmedian(img)-0.35*np.nanstd(img), vmax=np.nanmedian(img)+0.35*np.nanstd(img), extent=(0, img.shape[1], img.shape[0], 0),interpolation='none')

            if(img_size!=128):
                tp = transform.resize(np.flipud(seg_res['tp']), (img_size,img_size)).astype(int)
                fp = transform.resize(np.flipud(seg_res['fp']), (img_size,img_size)).astype(int) 
                fn =transform.resize(np.flipud(seg_res['fn']), (img_size,img_size)).astype(int) 
            # Segmentation overlays
            ax.imshow(tp, cmap=cmap_tp, alpha=0.4, interpolation='none', extent=(0, img.shape[1], img.shape[0], 0))
            ax.imshow(fp, cmap=cmap_fp, alpha=0.4, interpolation='none', extent=(0, img.shape[1], img.shape[0], 0))
            ax.imshow(fn, cmap=cmap_fn, alpha=0.4, interpolation='none', extent=(0, img.shape[1], img.shape[0], 0))

            # Tracking overlays: plot fronts
            for front in fronts:
                img_dim = img.shape[0]
                xcoords = front['x_coords']
                ycoords = front['y_coords']

                if(img_size!=128):
                    xcoords = np.clip(xcoords*8.0, 0, img_size-1).astype(int)
                    ycoords = np.clip(ycoords*8.0, 0, img_size-1).astype(int)

                label = front['label']
                color = c_purple if label == "TP" else c_pink
                ax.scatter(xcoords, img_dim-ycoords, s=10, color=color, label=label, alpha=0.9, marker='x')

            # Add text in the top middle of each image (not as a title)
            # ax.text(
            #     0.5, 0.05, 
            #     datetime.datetime.strftime(dt, '%Y%m%d_%H%M%S'), 
            #     fontsize=10, 
            #     color='white', 
            #     ha='center', 
            #     va='bottom', 
            #     transform=ax.transAxes,
            #     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
            # )

            ax.axis('off')
            ax.set_frame_on(False)

        plt.savefig(save_path + 'tracking_segmentation_' + datetime.datetime.strftime(dt, '%Y%m%d_%H%M%S') + '.jpg',
                    dpi=300, bbox_inches='tight', pad_inches=0.0)

        plt.close()

def main(mdls, ml_path, best_method, best_threshold, date_str, mode='test', plotting=False, rdif_path=None,img_size=128):

    final_mae_start = []
    final_mae_end = []
    final_iou = []
    final_precision = []
    final_recall = []
    final_tp = []
    final_fp = []
    final_fn = []
    final_total = []

    filenames_final = []

    folder_save = ml_path + 'results_science/' + date_str + '/'
    # folder_save = '/home/lelouedec/testml/results_science/' + date_str + '/'
    
    os.makedirs(folder_save, exist_ok=True)

    for mdl in mdls:

        load_path = ml_path+mdl+'/segmentation_masks_'+best_method+'_'+mode+'.npz'
        filenames, _ = load_results(load_path)
        filenames = [name.decode('utf-8') for name in filenames]
        filenames_final.append(filenames)

        groundtruth_dict = np.load(ml_path+mdl+'/segmentation_dictionary_gt_'+mode+'.npy',allow_pickle=True).item()
        prediction_dict = np.load(ml_path+mdl+'/segmentation_dictionary_'+best_method+'_'+mode+'_'+str(best_threshold).split('.')[-1]+'.npy',allow_pickle=True).item()

        matches, unmatched_gt, unmatched_pred = match_cmes_unique(groundtruth_dict, prediction_dict)
        result = evaluate_matches(matches, unmatched_gt, unmatched_pred)

        final_mae_start.append(result['mean_start_error'])
        final_mae_end.append(result['mean_end_error'])
        final_iou.append(result['iou'])
        final_precision.append(result['precision'])
        final_recall.append(result['recall'])
        final_tp.append(result['num_true_positives'])
        final_fp.append(result['num_false_positives'])
        final_fn.append(result['num_false_negatives'])
        final_total.append(result['total_cmes_gt'])

    with open(folder_save+'event_based_tracking_results_science.txt' , 'w') as f:
        print('Final Results:',file=f)
        print('Models',mdls,file=f)
        print('\n',file=f)
        print('MAE Start:', np.round(np.mean(final_mae_start),2), u'\u00B1' + str( ), np.round(np.std(final_mae_start),2), file=f)
        print('MAE End:', np.round(np.mean(final_mae_end),2), u'\u00B1' + str( ), np.round(np.std(final_mae_end),2), file=f)
        print('IoU:', np.round(np.mean(final_iou),2), u'\u00B1' + str( ), np.round(np.std(final_iou),2), file=f)
        print('Precision:', np.round(np.mean(final_precision),2), u'\u00B1' + str( ), np.round(np.std(final_precision),2), file=f)
        print('Recall:', np.round(np.mean(final_recall),2), u'\u00B1' + str( ), np.round(np.std(final_recall),2), file=f)
        print('\n',file=f)
        print('True Positives:', np.round(np.sum(final_tp),0), u'\u00B1' + str( ), np.round(np.std(final_tp),0), file=f)
        print('False Positives:', np.round(np.sum(final_fp),0), u'\u00B1' + str( ), np.round(np.std(final_fp),0),file=f)
        print('False Negatives:', np.round(np.sum(final_fn),0), u'\u00B1' + str( ), np.round(np.std(final_fn),0), file=f)
        print('Total CMEs in GT', np.round(np.sum(final_total),0),file=f)
        print('\n',file=f)
        print('Best Method:', best_method, file=f)
        print('Best Threshold:', str(best_threshold), file=f)

    if plotting:
        best_mdl = mdls[np.argmax(final_iou)]
        load_path = ml_path+best_mdl+'/segmentation_masks_'+best_method+'_'+mode+'.npz'
        
        groundtruth_dict = np.load(ml_path+best_mdl+'/segmentation_dictionary_gt_'+mode+'.npy',allow_pickle=True).item()
        prediction_dict = np.load(ml_path+best_mdl+'/segmentation_dictionary_'+best_method+'_'+mode+'_'+str(best_threshold).split('.')[-1]+'.npy',allow_pickle=True).item()
        
        filenames, _ = load_results(load_path)
        filenames = [name.decode('utf-8') for name in filenames]

        input_dates_plot = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d_%H%M%S') for name in filenames if name.split('/')[-1]]

        img_paths = []
        img_paths = [rdif_path + tim + '_1bh1A.npy' for tim in input_dates_plot]

        input_imgs = []
        for file in img_paths:
            input_imgs.append(transform.resize(np.load(file), (img_size,img_size)))


        input_dates_datetime = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S') for name in filenames if name.split('/')[-1]]

        seg_by_date = collect_segmentation_results_by_date(input_dates_datetime, prediction_dict, groundtruth_dict)
        fronts_by_date = collect_tracking_fronts_by_date(input_dates_datetime, prediction_dict, matches, unmatched_pred)

        plot_segmentation_and_tracking_grid_gap(
        input_dates_datetime,
        input_imgs,
        seg_by_date,
        fronts_by_date,
        save_path=folder_save+'event_based_tracking_imgs_'+best_mdl+'/',
        gap=3,
        img_size=img_size
        )

if __name__ == '__main__':

    config = parse_yml('config_evaluation.yaml')

    mode = config['mode']
    mdls_event_based = config['mdls_event_based']
    ml_path = config['paths']['ml_path']

    plotting = config['plotting']
    rdif_path = config['paths']['rdif_path']

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
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")


    main(mdls=mdls_event_based, ml_path=ml_path, best_method=best_method, best_threshold=best_threshold, date_str=date_str, mode=mode, plotting=plotting, rdif_path=rdif_path,img_size=config["img_size"])