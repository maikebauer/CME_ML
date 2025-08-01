import numpy as np
import os
import datetime
from skimage import transform
from astropy.wcs import FITSFixedWarning
import warnings
import sys
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.simplefilter('ignore', category=FITSFixedWarning)
from collections import namedtuple
import pandas as pd
import matplotlib.colors as mpl_colors
from utils_evaluation import load_results, evaluate_matches, match_cmes_extra, get_binary_range_revised
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import json
from utils import parse_yml

def match_cmes_unique(ground_truth_cmes, predicted_cmes, overlap_threshold=0.25, elon_threshold=3):
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

            pa_gt = np.unique(np.array(ground_truth_cmes[gt_name]['cme_front_wcs'])[:,0])

            if len(pa_gt) == 1:
                pa_gt = pa_gt[0]
            else:
                print(f'Multiple PA values in HELCATS track {gt_name} (PAs: {pa_gt})')
                sys.exit()

            overlap = max(0, (earliest_end - latest_start).total_seconds())
            total_duration = (latest_end - earliest_start).total_seconds()
            overlap_fraction = overlap / total_duration if total_duration > 0 else 0

            if overlap_fraction < overlap_threshold:
                continue

            intersect_times, pred_inds, gt_inds = np.intersect1d(pred_times, gt_times, return_indices=True)
            
            if len(intersect_times) == 0:
                continue

            elon_diff = []

            # Potentially add weight to elongation difference if there is no overlap in time
            # unique_times_pred = pred_times[~np.in1d(pred_times,gt_times)]

            # if len(unique_times_pred) > 0:
            #     for pi in unique_times_pred:
            #         elon_diff.append(99)

            for pi, gi in zip(pred_inds, gt_inds):
                elon_ind = np.where((predicted_cmes[pred_name]['cme_front_wcs'][pi][0] >= pa_gt-2) & (predicted_cmes[pred_name]['cme_front_wcs'][pi][0] <= pa_gt+2))[0]
                elon_along_pa = np.nanmean(predicted_cmes[pred_name]['cme_front_wcs'][pi][1][elon_ind])

                e_diff = np.abs(elon_along_pa - ground_truth_cmes[gt_name]['cme_front_wcs'][gi][1])

                if np.isnan(e_diff):
                    elon_diff.append(np.nan) # Potentially add weight to elongation difference if there is no overlap in time
                else:
                    elon_diff.append(e_diff)

            mean_elon_diff = np.nanmean(elon_diff)

            if np.isnan(mean_elon_diff):
                continue

            if mean_elon_diff > elon_threshold:
                continue

            start_err = abs((gt_range.start - pred_range.start).total_seconds()) / 3600
            end_err = abs((gt_range.end - pred_range.end).total_seconds()) / 3600
            total_err = start_err + end_err + mean_elon_diff

            cost_matrix[i, j] = total_err
            match_info[(i, j)] = {
                "gt_match": gt_name,
                "start_error": start_err,
                "end_error": end_err,
                "elon_error": mean_elon_diff,
                "total_error": total_err
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

    return matches, unmatched_gt, unmatched_pred, matched_gt

def collect_tracking_fronts_by_date(input_dates_plot, cme_dictionary, matches, unmatched, mode):
    """
    Organizes predicted CME fronts by timestamp and tags each as TP or FP for tracking.

    Parameters:
    - input_dates_plot: list of datetime objects to include
    - cme_dictionary: dict of predicted CMEs with CME name as key and:
        - 'times': list of timestamps
        - 'cme_front_pixels': list of (y_coords, x_coords) pairs
    - matches: dict with CME name as key and match details (implies TP)
    - unmatched: list of CME names (implies FP)
    - mode: gt or pred

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
        elif cme_name in unmatched and mode == 'pred':
            code = "FP"
        elif cme_name in unmatched and mode == 'gt':
            code = "FN"
        else:
            if mode == 'gt':
                code = "TP"
            else:
                continue

        for idx, ts in enumerate(times):
            if ts not in fronts_by_date:
                continue

            if idx >= len(fronts):
                continue  # Safety check

            y_coords, x_coords = fronts[idx]

            # Only add if front has actual pixels
            if mode == 'gt':
                if not x_coords or not y_coords:
                    continue
            
            else:
                if len(x_coords) == 0 or len(y_coords) == 0:
                    continue

            fronts_by_date[ts].append({'name':cme_name, 'x_coords':x_coords, 'y_coords':y_coords,'label':code})

    return fronts_by_date

def plot_tracking_grid_gap(
    input_dates_plot,
    input_images,
    fronts_by_date,
    fronts_by_date_gt,
    save_path,
    gap=1,
    segmentation_dictionary=None,
    use_threshold=None,
    img_size=128
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

    fac = 1.5
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

            if segmentation_dictionary is not None:
                seg_mask = segmentation_dictionary.get(dt, np.zeros_like(img))

            ax = axes[j]
            ax.imshow(np.flipud(img), aspect='equal', cmap='gray', vmin=np.nanmedian(img)-fac*np.nanstd(img), vmax=np.nanmedian(img)+fac*np.nanstd(img),extent=(0, img.shape[1], img.shape[0], 0), interpolation='none')
            
            if segmentation_dictionary is not None:
                
                seg_mask = transform.resize(segmentation_dictionary[dt], (img_size, img_size), anti_aliasing=True)
                if use_threshold == True:
                    geo_red = (204/255,44/255,1/255)
                    c_black = mpl_colors.colorConverter.to_rgba('black',alpha=0)
                    cmap_gt = mpl_colors.ListedColormap([c_black,geo_red])
                    ax.imshow(np.flipud(seg_mask), cmap=cmap_gt, alpha=0.3, vmin=0, vmax=1, extent=(0, img.shape[1], img.shape[0], 0), interpolation='none')
                
                if use_threshold == False:
                    color_img = ax.imshow(np.flipud(seg_mask), cmap='jet', alpha=0.3, vmin=0, vmax=1, extent=(0, img.shape[1], img.shape[0], 0), interpolation='none')


            # Tracking overlays: plot fronts
            for front in fronts:
                img_dim = img.shape[0]
                xcoords = front['x_coords']
                ycoords = front['y_coords']

                if img_size != 128:
                    xcoords = np.clip(xcoords*8.0, 0, img_size-1).astype(int)
                    ycoords = np.clip(ycoords*8.0, 0, img_size-1).astype(int)

                label = front['label']
                color = c_purple if label == "TP" else c_pink
                ax.scatter(xcoords, img_dim-ycoords, s=10, color=color, label=label, alpha=0.9, marker='x')

            for front in fronts_gt:
                img_dim = img.shape[0]
                xcoords = front['x_coords']
                ycoords = front['y_coords']

                if img_size != 128:
                    xcoords = np.clip(xcoords*8.0, 0, img_size-1).astype(int)
                    ycoords = np.clip(ycoords*8.0, 0, img_size-1).astype(int)

                label = front['label']
                color = c_orange if label == "FN" else c_tp_gt
                ax.scatter(xcoords, img_dim-ycoords, s=25, color=color, label=label, alpha=0.9, marker='*', edgecolors='face',linewidths=2) 


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
            #ax.text(.01, .99, dt.strftime('%Y-%m-%d %H:%M'), ha='left', va='top', fontsize=8, transform=ax.transAxes)
            
            #ax.set_title(dt.strftime('%Y-%m-%d %H:%M'), fontsize=8)
            ax.axis('off')
            ax.set_frame_on(False)

        if segmentation_dictionary is not None and use_threshold == False:
            fig.colorbar(color_img, ax=axes, location='right', anchor=(0, 0.5),shrink=0.5)
            
        #plt.tight_layout()
        plt.savefig(save_path+ 'tracking_segmentation_' + datetime.datetime.strftime(dt, '%Y%m%d_%H%M%S') + '.jpg', dpi=300, bbox_inches='tight',pad_inches=0.0)
        #plt.show()
        plt.close()

def main(mdl, ml_path, timepairs, best_method, best_threshold, date_str, wp2_path, plotting=False, plot_area=False, use_threshold=True, years_plotting=None, months_plotting=None, data_paths=None, img_size=128):

    filenames_final = []

    for pair_idx, pair in enumerate(timepairs):
        start = pair['start']
        end = pair['end']

        start_datetime = datetime.datetime.strptime(start, '%Y_%m_%d')
        end_datetime = datetime.datetime.strptime(end, '%Y_%m_%d')

        groundtruth_dict = np.load(ml_path+'helcats_dictionary_'+start+'_'+end+'.npy', allow_pickle=True).item()
        prediction_dict = np.load(ml_path+mdl+'/segmentation_dictionary_'+best_method+'_'+str(best_threshold).split('.')[-1]+'_'+start+'_'+end+'.npy', allow_pickle=True).item()

        load_path = ml_path+mdl+'/segmentation_masks_'+best_method+'_'+start+'_'+end+'.npz'

        if plotting and plot_area:
            filenames, pred = load_results(load_path)
            filenames = [name.decode('utf-8') for name in filenames]
            file_times = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S') for name in filenames]
            
            segmentation_dictionary = {}
            for idx, file in enumerate(file_times):
                if use_threshold:
                    segmentation_dictionary[file] = np.where(pred[idx] > best_threshold, 1, 0)
                else:
                    segmentation_dictionary[file] = pred[idx].copy()

            if use_threshold == True:
                path_append = '_areas/'
            
            elif use_threshold == False:
                path_append = '_areas_nothresh/'
            
            else:
                print('Invalid use_threshold value. Must be True or False.')
                sys.exit()

        else:
            filenames, _ = load_results(load_path)
            filenames = [name.decode('utf-8') for name in filenames]
            segmentation_dictionary = None

            path_append = '/'
        
        filenames_final.append(filenames)

        matches, unmatched_gt, unmatched_pred, matched_gt = match_cmes_unique(groundtruth_dict, prediction_dict, overlap_threshold=0.25, elon_threshold=4)
        result = evaluate_matches(matches, unmatched_gt, unmatched_pred)

        with open(wp2_path, 'r') as fil:
            wp2_json = json.load(fil)

        wp2_data = wp2_json['data']     

        wp2_reduced = {}
        wp2_reduced =  defaultdict(lambda: {'CME': str, 'time_start': str, 'PA_min': int, 'PA_max': int, 'SC': str, 'quality':str})

        for entry in wp2_data:
            datetime_entry = datetime.datetime.strptime(entry[1],'%Y-%m-%d %H:%M')
            if (datetime_entry >= start_datetime and datetime_entry <= end_datetime) and entry[2] == 'A' and entry[0] not in list(groundtruth_dict.keys()):
                wp2_reduced[entry[0]] = {
                    'time_start': datetime_entry,
                    'PA_min': entry[4],
                    'PA_max': entry[6],
                    'SC': entry[2],
                    'quality': entry[7]
                }

        unmatched_prediction_dict = {key: value for key, value in prediction_dict.items() if key not in matches.keys()}
        matches_extra, unmatched_gt_extra, unmatched_pred_extra = match_cmes_extra(wp2_reduced, unmatched_prediction_dict, time_threshold=200)

        wp2_all_data = {}
        wp2_all_data =  defaultdict(lambda: {'CME': str, 'time_start': str, 'PA_min': int, 'PA_max': int, 'SC': str, 'quality':str})

        for entry in wp2_data:
            datetime_entry = datetime.datetime.strptime(entry[1],'%Y-%m-%d %H:%M')
            if (datetime_entry >= start_datetime and datetime_entry <= end_datetime) and entry[2] == 'A':
                wp2_all_data[entry[0]] = {
                    'time_start': datetime_entry,
                    'PA_min': entry[4],
                    'PA_max': entry[6],
                    'SC': entry[2],
                    'quality': entry[7]
                }

        num_fair_unmatched = 0
        num_poor_unmatched = 0
        num_good_unmatched = 0

        for entry in unmatched_gt:
            if wp2_all_data[entry]['quality'] == 'fair':
                num_fair_unmatched += 1
            elif wp2_all_data[entry]['quality'] == 'poor':
                num_poor_unmatched += 1
            elif wp2_all_data[entry]['quality'] == 'good':
                num_good_unmatched += 1

        num_fair_matched = 0
        num_poor_matched = 0
        num_good_matched = 0

        for entry in matched_gt:
            if wp2_all_data[entry]['quality'] == 'fair':
                num_fair_matched += 1
            elif wp2_all_data[entry]['quality'] == 'poor':
                num_poor_matched += 1
            elif wp2_all_data[entry]['quality'] == 'good':
                num_good_matched += 1

        with open(ml_path+'results_science/'+date_str+'/operational_tracking_'+start+'_'+end+'_results.txt' , 'w') as f:
            print('Final Results:',file=f)
            print('Model',mdl,file=f)
            print('Start:' + start, file=f)
            print('End:' + end, file=f)
            print('\n',file=f)
            print('MAE Start:', result['mean_start_error'], file=f)
            print('MAE End:', result['mean_end_error'], file=f)
            print('IoU:', result['iou'], file=f)
            print('Precision:', result['precision'], file=f)
            print('Recall:', result['recall'], file=f)
            print('\n',file=f)
            print('True Positives:', result['num_true_positives'], file=f)
            print('False Positives:', result['num_false_positives'],file=f)
            print('False Negatives:', result['num_false_negatives'], file=f)
            print('Total CMEs in WP3', result['total_cmes_gt'],file=f)
            print('\n',file=f)
            print('Quality of all unmatched WP3 CMEs - poor: {}, fair: {}, good: {}'.format(num_poor_unmatched, num_fair_unmatched, num_good_unmatched), file=f)
            print('Quality of all matched WP3 CMEs - poor: {}, fair: {}, good: {}'.format(num_poor_matched, num_fair_matched, num_good_matched), file=f)
            print('\n',file=f)
            print('Extra CMEs in WP2:', len(wp2_reduced),file=f)
            print('Quality of extra CMEs in WP2 - poor: {}, fair: {}, good: {}'.format(
                len([cme for cme in wp2_reduced if wp2_reduced[cme]['quality'] == 'poor']),
                len([cme for cme in wp2_reduced if wp2_reduced[cme]['quality'] == 'fair']),
                len([cme for cme in wp2_reduced if wp2_reduced[cme]['quality'] == 'good'])
            ),file=f)
            print('Extra TPs from WP2:', len(matches_extra),file=f)
            print('Quality of extra TPs from WP2 - poor: {}, fair: {}, good: {}'.format(
                len([cme for cme in matches_extra if wp2_reduced[matches_extra[cme]['gt_match']]['quality'] == 'poor']),
                len([cme for cme in matches_extra if wp2_reduced[matches_extra[cme]['gt_match']]['quality'] == 'fair']),
                len([cme for cme in matches_extra if wp2_reduced[matches_extra[cme]['gt_match']]['quality'] == 'good'])
            ),file=f)
            print('\n',file=f)
            print('Best Method: ' + best_method, file=f)
            print('Best Threshold: ' + str(best_threshold), file=f)

        if plotting:    

            binary_range , scatter_extra= get_binary_range_revised(prediction_dict, groundtruth_dict, matches, unmatched_gt, wp2_reduced, pair)
            binary_range_plot = np.zeros((np.shape(binary_range)[0]*np.shape(binary_range)[1], np.shape(binary_range)[2]))
            for i in range(np.shape(binary_range)[0]):
                for j in range(np.shape(binary_range)[1]):
                    binary_range_plot[i*np.shape(binary_range)[1]+j] = binary_range[i][j]

            date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='ME')
            date_labels = [datetime.datetime.strftime(dt, '%Y/%m') for dt in date_range]

            SMALL_SIZE = 8
            MEDIUM_SIZE = 12
            BIGGER_SIZE = 16

            MARKER_SIZE = 9

            al = 1
            c_black = mpl.colors.colorConverter.to_rgba('#1a1633',alpha=al)
            c_orange= mpl.colors.colorConverter.to_rgba('#ff6100',alpha=al)
            c_pink= mpl.colors.colorConverter.to_rgba('#dc2580',alpha=al)
            c_purple= mpl.colors.colorConverter.to_rgba('#785ef1',alpha=al)

            c_map = mpl.colors.ListedColormap([c_black, c_pink, c_orange, c_purple, 'white'])
            fig, ax = plt.subplots(figsize=(15,5))
            ax.scatter(scatter_extra[:,0], scatter_extra[:,1],color='white',marker='x',s=25)
            ax.imshow(np.flipud(binary_range_plot), aspect='equal', cmap=c_map, interpolation='none', extent = [1,32, 0.5, 12.5])

            ax.set_yticks(range(1,13))
            #ax.set_yticklabels(calendar.month_abbr[1:13])
            ax.set_yticklabels(date_labels)
            ax.set_ylabel('Year/Month', fontsize=BIGGER_SIZE)
            ax.set_xlabel('Day', fontsize=BIGGER_SIZE)
            ax.set_xticks(range(1,32))
            ax.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
            gridlines = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
            for gr in gridlines:
                ax.axhline(gr, color='white', alpha=0.8)


            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            legend_labels = ['False Positive', 'False Negative', 'True Positive', 'Outside Time Range']
            colors = [c_pink, c_orange, c_purple, 'gray']
            patches = [mpl.patches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels)-1)]
            ax.legend(handles=patches, bbox_to_anchor=(0.5, 1.10), loc='upper center', borderaxespad=0., ncol=5, frameon=False, fontsize=BIGGER_SIZE)

            plt.savefig(ml_path+'results_science/'+date_str+'/heatmap_'+start+'_'+end+'_'+mdl+'.jpg', dpi=300, bbox_inches='tight')
            plt.close()

            if years_plotting is None or months_plotting is None:
                print('No years or months to plot. Skipping plotting.')
                continue

            else:
                year = years_plotting[pair_idx]
                for month in months_plotting[pair_idx]:
                    print('Plotting for year: ' + year + ' and month: ' + month)
                    input_dates_plot = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d_%H%M%S') for name in filenames if name.split('/')[-1].startswith(year+month)]

                    img_paths = [data_paths[pair_idx] + tim + '_1bh1A.npy' for tim in input_dates_plot]

                    input_imgs = []

                    for file in img_paths:
                        input_imgs.append(transform.resize(np.load(file), (img_size,img_size)))

                        input_dates_datetime = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S') for name in filenames if name.split('/')[-1]]
                        input_dates_datetime_plot = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S') for name in filenames if name.split('/')[-1].startswith(year+month)]

                    fronts_by_date = collect_tracking_fronts_by_date(input_dates_datetime, prediction_dict, matches, unmatched_pred, 'pred')
                    fronts_by_date_gt = collect_tracking_fronts_by_date(input_dates_datetime, groundtruth_dict, matches, unmatched_gt, 'gt')
                    
                    plot_tracking_grid_gap(input_dates_datetime_plot,
                                    input_imgs,
                                    fronts_by_date,
                                    fronts_by_date_gt,
                                    save_path=ml_path+'results_science/'+date_str+'/'+'operational_tracking_imgs_'+start+'_'+end+'_'+mdl+path_append,
                                    gap=3,
                                    segmentation_dictionary=segmentation_dictionary,
                                    use_threshold=use_threshold,
                                    img_size=img_size)


if __name__ == '__main__':

    config = parse_yml('config_evaluation.yaml')

    mdls_operational = config['mdls_operational']
    ml_path = config['paths']['ml_path']
    wp2_path = config['paths']['wp2_path']
    data_paths = config['paths']['data_paths']

    plotting = config['plotting']
    rdif_path = config['paths']['rdif_path']

    time_pairs = config['time_pairs']
    timepairs = [{'start': time_pairs['start'][i], 'end': time_pairs['end'][i]} for i in range(len(time_pairs['start']))]

    if 'dates_plotting_operational' not in config or not config['dates_plotting_operational']:
        dates_plotting_operational = None
        years_plotting_operational = None
        months_plotting_operational = None
    else:
        dates_plotting_operational = config['dates_plotting_operational']
        years_plotting_operational = list(dates_plotting_operational.keys())
        months_plotting_operational = [dates_plotting_operational[year] for year in years_plotting_operational]

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

    main(mdl=mdls_operational, 
         ml_path=ml_path, 
         timepairs=timepairs, 
         best_method=best_method, 
         best_threshold=best_threshold, 
         date_str=date_str,
         wp2_path=wp2_path,
         plotting=plotting, 
         plot_area=False, 
         use_threshold=True, 
         years_plotting=years_plotting_operational, 
         months_plotting=months_plotting_operational,
         data_paths=data_paths,
         img_size=config['img_size']
         )