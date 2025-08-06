import numpy as np
import os
import datetime
from skimage.morphology import disk
from scipy import ndimage
from skimage.measure import label
from skimage import morphology
import astropy.io.fits as fits
from astropy import wcs
from astropy.wcs import FITSFixedWarning
import warnings
import sys
from PIL import Image
import json
from pycocotools import coco
warnings.simplefilter('ignore', category=FITSFixedWarning)
from collections import defaultdict
import glob
import copy
from collections import namedtuple
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries
from skimage.filters import gaussian
from plantcv import plantcv as pcv

def load_results(path, load_gt=False):

    print('Loading results...')
    results = np.load(path)
    filenames = results['filenames']
    pred = results['masks']
    
    if load_gt:
        gt = results['gt']
        return filenames, pred, gt
    
    else:
        return filenames, pred

def post_processing(pred_final, t=0.45, return_labeled=True):

    # print('Post-processing...')
    
    images_proc_thresh = pred_final.copy().astype(float)

    if return_labeled:
        labeled_clusters_connected = np.zeros_like(images_proc_thresh)

    # Label the connected components in the binary image (using skimage.clusters label)
    for i in range(len(images_proc_thresh)):
        # Pre-processing

        images_proc_thresh[i] = np.where(images_proc_thresh[i] > t, 1, 0)
        images_proc_thresh[i] = morphology.remove_small_objects(images_proc_thresh[i].astype(bool), min_size=64, connectivity=2)
        images_proc_thresh[i] = morphology.remove_small_holes(images_proc_thresh[i].astype(bool), area_threshold=50, connectivity=2)
        
        images_proc_thresh[i] = images_proc_thresh[i].astype(float)
        
        if return_labeled:
            labeled_clusters_connected[i] = label(images_proc_thresh[i])

    if return_labeled:
        return images_proc_thresh, labeled_clusters_connected.astype(np.uint8)
    
    else:
        return images_proc_thresh


def clean_cme_elongation_tracks(
    cme_dictionary,
    pa_tolerance=0.1,
    dip_thresh_deg=0.5,
    dip_thresh_count=3
):
    cme_dictionary_final = copy.deepcopy(cme_dictionary)
    
    for cme_key, cme_data in cme_dictionary_final.items():
        time_list = cme_data['times']
        wcs_list = cme_data['cme_front_wcs']
        pixel_list = cme_data['cme_front_pixels']

        # Collect elongation data per PA across time
        pa_time_series = {}

        for t_idx, (time, wcs_data) in enumerate(zip(time_list, wcs_list)):
            pa_values, elong_values = wcs_data
            for pa, elong in zip(pa_values, elong_values):
                # Match to existing PA bin within tolerance
                matched = False
                for existing_pa in pa_time_series.keys():
                    if abs(existing_pa - pa) <= pa_tolerance:
                        pa_time_series[existing_pa].append((t_idx, elong))
                        matched = True
                        break
                if not matched:
                    pa_time_series[pa] = [(t_idx, elong)]

        # Clean elongation sequences
        cleaned_pa_data = {}
        for pa, series in pa_time_series.items():
            series.sort()  # Sort by time index
            cleaned_series = []
            max_elong = -np.inf
            dip_buffer = []
            discard_mode = False

            for idx, elong in series:
                if elong >= max_elong:
                    if dip_buffer:
                        # Check dip criteria
                        max_dip = max(max_elong - e for _, e in dip_buffer)
                        if len(dip_buffer) <= dip_thresh_count and max_dip <= dip_thresh_deg:
                            # Interpolate over dip_buffer
                            cleaned_series.extend(dip_buffer)
                        # Else: Discard them (do nothing)
                        dip_buffer = []
                    max_elong = elong
                    cleaned_series.append((idx, elong))
                else:
                    dip_buffer.append((idx, elong))
                    if len(dip_buffer) > dip_thresh_count or (max_elong - elong) > dip_thresh_deg:
                        # Too deep or too long dip → discard from here
                        break

            cleaned_pa_data[pa] = dict(cleaned_series)  # map: time_index → elong

        # Rebuild cleaned data
        new_wcs_list = []
        new_pixel_list = []

        for t_idx, (wcs_data, pixel_data) in enumerate(zip(wcs_list, pixel_list)):
            pa_values = wcs_data[0]
            elong_values = wcs_data[1]
            x_values = pixel_data[0]
            y_values = pixel_data[1]

            cleaned_pa = []
            cleaned_elong = []
            cleaned_x = []
            cleaned_y = []

            for pa, elong, x, y in zip(pa_values, elong_values, x_values, y_values):
                for cleaned_pa_key, time_elong_map in cleaned_pa_data.items():
                    if abs(pa - cleaned_pa_key) <= pa_tolerance:
                        if t_idx in time_elong_map:
                            cleaned_pa.append(pa)
                            cleaned_elong.append(elong)
                            cleaned_x.append(x)
                            cleaned_y.append(y)
                        break  # matched

            # Sort by PA again
            sorted_entries = sorted(zip(cleaned_pa, cleaned_elong, cleaned_x, cleaned_y), key=lambda item: item[0])
            if sorted_entries:
                pa_sorted, elong_sorted, x_sorted, y_sorted = zip(*sorted_entries)
                new_wcs_list.append([np.array(pa_sorted), np.array(elong_sorted)])
                new_pixel_list.append([np.array(x_sorted), np.array(y_sorted)])
            else:
                new_wcs_list.append([np.array([]), np.array([])])
                new_pixel_list.append([np.array([]), np.array([])])

        # Replace data in dictionary
        cme_data['cme_front_wcs'] = new_wcs_list
        cme_data['cme_front_pixels'] = new_pixel_list
        cme_dictionary_final[cme_key] = cme_data

    return cme_dictionary_final

def get_fitsfiles(input_days_set, input_dates):
    print('Getting fits files...')
    rundif_paths = ['/media/DATA_DRIVE/stereo_processed/reduced/data/A/' + day + '/science/hi_1/' for day in input_days_set]

    # Import all fts files in ordered list

    fts_files = []

    for path in rundif_paths:
        temp_files = []
        for day in input_dates:
            temp_files.extend([path + file for file in os.listdir(path) if file.endswith('.fts') and day in file])
        fts_files.append(temp_files)

    
    fits_headers = []

    for files in fts_files:
        for file in files:
            with fits.open(file) as hdul:
                fits_headers.append(hdul[0].header)
    
    return fits_headers

def get_outline(image_outlines, fits_headers):
    print('Getting outline...')
    image_outlines_wcs = []
    image_outline_pixels = []

    for i, cent in enumerate(image_outlines):
        current_header = fits_headers[i]
        wcoord = wcs.WCS(current_header)

        temp_coords = []
        temp_pixels = []
        for c in cent:
            xv, yv = c
            thetax, thetay = wcoord.all_pix2world(yv*8, xv*8, 0)
            tx = thetax*np.pi/180
            ty = thetay*np.pi/180

            pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))
            elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))

            temp_coords.append([pa_reg*180/np.pi, elon_reg*180/np.pi])
            temp_pixels.append([xv, yv])

        image_outlines_wcs.append(temp_coords)
        image_outline_pixels.append(temp_pixels)

    return image_outlines_wcs, image_outline_pixels

def order_points(points, ind): 

    ### other solutions for that https://stackoverflow.com/questions/58377015/counterclockwise-sorting-of-x-y-data
    points_new = [ points.pop(ind) ]  # initialize a new list of points with the known first point
    pcurr      = points_new[-1]       # initialize the current point (as the known point)
    while len(points)>0:
        d      = np.linalg.norm(np.array(points) - np.array(pcurr), axis=1)  # distances between pcurr and all other remaining points
        ind    = d.argmin()                   # index of the closest point
        points_new.append( points.pop(ind) )  # append the closest point to points_new
        pcurr  = points_new[-1]               # update the current point

    return points_new


def getwcords_new(header, xv, yv):

    wcoord = wcs.WCS(header)

    thetax, thetay = wcoord.all_pix2world(yv, xv, 0)
    tx = thetax*np.pi/180
    ty = thetay*np.pi/180
    pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))

    elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))


    return elon_reg,pa_reg

def getwcords_pix(header, pa_rad, elon_rad):

    wcoord = wcs.WCS(header)

    tx = np.rad2deg(np.arctan2(-np.sin(elon_rad)*np.sin(pa_rad), np.cos(elon_rad)))
    ty = np.rad2deg(np.arcsin(np.sin(elon_rad)*np.cos(pa_rad)))

    pix_y_from_img, pix_x_from_img = wcoord.all_world2pix(tx,ty, 0)

    return pix_x_from_img,pix_y_from_img

def bin_pa(pas, elongs):

    bin_bin = np.linspace(min(pas)-1,max(pas)+1,10)
    bins = np.digitize(pas,bin_bin)

    A = np.vstack( ( np.digitize(pas,bin_bin), elongs)).T
    average = [np.mean(elongs[bins==i]) for i in np.unique(bins)]
    
    return np.deg2rad(bin_bin[np.unique(bins)]),np.deg2rad(average)

def get_front(image_outlines_wcs, image_outline_pixels, fits_headers, image_areas=None):
    print('Getting front...')
    img_front_wcs = []
    img_front_pixels = []
    
    if image_areas is not None:
        img_front_areas = []

    for out_num, outline in enumerate(image_outlines_wcs):

        temp_coords = []
        temp_pixels = []

        if image_areas is not None:
            temp_areas = []

        if(len(image_outline_pixels[out_num])>0):
        
            indices = copy.deepcopy(image_outline_pixels[out_num])
            for i,idx in enumerate(indices):
                img = np.zeros((128,128))
                idx_arr = np.array(copy.deepcopy(idx))
                
                if len(idx_arr[0]) == 0:
                    continue

                img[idx_arr[0,:],idx_arr[1,:]] = 1.0

                points = [(xx,yy)  for xx,yy in zip(idx_arr[0,:],idx_arr[1,:])]

                area = copy.deepcopy(image_areas[out_num][i])
                header = copy.deepcopy(fits_headers[out_num])

                if(len(points)>3):

                    skeleton = morphology.skeletonize(area)
                    pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton.astype(np.uint8), size=10)
                    pruned_coordinates = np.where(pruned_skeleton == 1)
                    skeleton_x = pruned_coordinates[0]
                    skeleton_y = pruned_coordinates[1]

                    elongs_skeleton, pas_skeleton = getwcords_new(header, skeleton_x*8, skeleton_y*8)

                    center_pa_skeleton_deg = np.rad2deg(pas_skeleton)
                    center_elong_skeleton_deg = np.rad2deg(elongs_skeleton)

                    temp_coords.append([center_pa_skeleton_deg, center_elong_skeleton_deg])
                    temp_pixels.append([np.clip(np.round(skeleton_x,0),0,127).astype(np.uint32), np.clip(np.round(skeleton_y,0),0,127).astype(np.uint32)])

                    if image_areas is not None:
                        temp_areas.append(area)


        img_front_wcs.append(temp_coords)
        img_front_pixels.append(temp_pixels)

        if image_areas is not None:
            img_front_areas.append(temp_areas)   


    if image_areas is not None:
        return img_front_wcs, img_front_pixels, img_front_areas
    else:
        return img_front_wcs, img_front_pixels
    

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

def aggregate_fronts_sequential(fronts, fronts_pixels, max_diff_elon=3, areas=None):

    n = len(fronts)

    if n <= 1:
        if areas is not None:
            return fronts, fronts_pixels, areas
        else:
            return fronts, fronts_pixels

    max_elons = [np.max(f[1]) if len(f[1]) > 0 else -999 for f in fronts]

    sorted_indices = np.argsort(max_elons)[::-1]  # Descending order

    used = set()
    discarded = set()

    groups = []

    for i in sorted_indices:
        if i in used or i in discarded:
            continue
        
        if len(fronts[i][1]) == 0:
            # print(f"Front {i} has no elongation data, skipping.")
            discarded.add(i)
            continue
        # Start a new group with front i
        group = [i]
        used.add(i)

        # Aggregate data for matching
        combined_elon = fronts[i][1].copy()
        combined_pa = fronts[i][0].copy()

        changed = True
        while changed:
            changed = False
            for j in range(n):
                if j in used or j in discarded:
                    continue
                
                if len(fronts[j][1]) == 0:
                    # print(f"Front {j} has no elongation data, skipping.")
                    discarded.add(j)
                    continue
                
                # diff = np.abs(np.max(fronts[j][1]) - np.max(combined_elon))
                diff = np.abs(np.mean(fronts[j][1]) - np.mean(combined_elon))
                if diff <= max_diff_elon:
                    # Add to group
                    group.append(j)
                    used.add(j)
                    combined_elon = np.concatenate((combined_elon, fronts[j][1]))
                    combined_pa = np.concatenate((combined_pa, fronts[j][0]))
                    changed = True

        groups.append(group)

    # Now build the aggregated outputs
    img_fronts_agg = []
    img_fronts_pixels_agg = []
    img_fronts_areas_agg = [] if areas is not None else None

    for group in groups:
        elon_group = []
        pa_group = []
        x_pix_group = []
        y_pix_group = []
        areas_group = []

        for idx in group:
            elon_group.append(fronts[idx][1])
            pa_group.append(fronts[idx][0])
            x_pix_group.append(fronts_pixels[idx][0])
            y_pix_group.append(fronts_pixels[idx][1])
            if areas is not None:
                areas_group.append(areas[idx])

        elon_group = np.concatenate(elon_group)
        pa_group = np.concatenate(pa_group)
        x_pix_group = np.concatenate(x_pix_group)
        y_pix_group = np.concatenate(y_pix_group)

        # Keep only largest elongation per PA
        keep_indices = []
        unique_pas = np.unique(pa_group)
        for pa in unique_pas:
            indices = np.where(pa_group == pa)[0]
            if len(indices) == 1:
                keep_indices.append(indices[0])
            else:
                max_idx = indices[np.argmax(elon_group[indices])]
                keep_indices.append(max_idx)

        keep_indices = np.array(keep_indices)
        elon_group = elon_group[keep_indices]
        pa_group = pa_group[keep_indices]
        x_pix_group = x_pix_group[keep_indices]
        y_pix_group = y_pix_group[keep_indices]

        # Sort by PA
        sort_idx = np.argsort(pa_group)
        elon_group = elon_group[sort_idx]
        pa_group = pa_group[sort_idx]
        x_pix_group = x_pix_group[sort_idx]
        y_pix_group = y_pix_group[sort_idx]

        img_fronts_agg.append([pa_group, elon_group])
        img_fronts_pixels_agg.append([x_pix_group, y_pix_group])

        if areas is not None:
            total_area = np.sum(areas_group, axis=0)
            img_fronts_areas_agg.append(total_area)

    if areas is not None:
        return img_fronts_agg, img_fronts_pixels_agg, img_fronts_areas_agg
    else:
        return img_fronts_agg, img_fronts_pixels_agg
    
def aggregate_fronts(fronts, fronts_pixels, max_diff_elon=3, areas=None):
    """
    Groups CME fronts that likely belong to the same CME.

    Args:
        fronts (list of arrays): List of CME fronts. Each front is an Nx2 array of (elongation, PA) coordinates.
        fronts_pixels (list of arrays): List of pixel coordinates corresponding to the CME fronts.
        max_diff (float): Maximum allowed difference in mean elongation for association.
        areas (list of arrays): List of areas corresponding to the CME fronts (optional).

    Returns:
        list of lists: Grouped fronts, where each group is a list of original front indices.
    """
    n = len(fronts)

    if n <= 1:
        # return [[i] for i in range(n)]  # No aggregation needed
        if areas != None:
            return fronts,fronts_pixels,areas  # No aggregation needed
        else:
            return fronts,fronts_pixels
        
    mean_elongations = [np.mean(front[1]) for front in fronts]
    uf = UnionFind(n)

    for i in range(n):
        diffs = []
        for j in range(n):
            if i == j: # Don't take difference between front and itself
                continue

            diff_elon = abs(mean_elongations[i] - mean_elongations[j])
            if diff_elon >= max_diff_elon:
                continue  # No point checking PA overlap if elongation is too different
            
            diffs.append((j, diff_elon))

            # # Compute PA overlap
            # pa_i = set(fronts[i][0])
            # pa_j = set(fronts[j][0])
            # overlap = len(pa_i & pa_j)
            # pa_overlap_ratio = overlap / min(len(pa_i), len(pa_j))
            # if pa_overlap_ratio <= max_overlap_pa:
            #     diffs.append((j, diff_elon))

        # Find the one with the minimum difference
        if diffs:
            best_match = min(diffs, key=lambda x: x[1])[0]
            uf.union(i, best_match)

    # Group by root parent
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    agg_groups = list(groups.values())
    img_fronts_agg = [[]]*len(agg_groups)
    img_fronts_pixels_agg = [[]]*len(agg_groups)

    if areas != None:
        img_fronts_areas_agg = [[]]*len(agg_groups)

    for group_id,group in enumerate(agg_groups):
        elon_group = []
        pa_group = []
        x_pix_group = []
        y_pix_group = []

        if areas != None:
            areas_group = []

        for index in group:
            elon_group.append(fronts[index][1])
            pa_group.append(fronts[index][0])
            x_pix_group.append(fronts_pixels[index][0])
            y_pix_group.append(fronts_pixels[index][1])

            if areas != None:
                areas_group.append(areas[index])

        elon_group = np.concatenate(elon_group)
        pa_group = np.concatenate(pa_group)
        x_pix_group = np.concatenate(x_pix_group)
        y_pix_group = np.concatenate(y_pix_group)

        # Create a mask that keeps only the largest elongation per PA
        keep_indices = []
        unique_pas = np.unique(pa_group)

        for pa in unique_pas:
            indices = np.where(pa_group == pa)[0]
            if len(indices) == 1:
                keep_indices.append(indices[0])
            else:
                # Pick index of max elongation
                max_idx = indices[np.argmax(elon_group[indices])]
                keep_indices.append(max_idx)

        # Apply the mask to both WCS and pixel data
        keep_indices = np.array(keep_indices)

        pa_group = pa_group[keep_indices]
        elon_group = elon_group[keep_indices]
        x_pix_group = x_pix_group[keep_indices]
        y_pix_group = y_pix_group[keep_indices]

        if areas != None:
            areas_group = np.sum(areas_group,axis=0)

        elon_group = np.array([y for (y,x) in sorted(zip(elon_group,pa_group), key=lambda pair: pair[1])])
        x_pix_group = np.array([y for (y,x) in sorted(zip(x_pix_group,pa_group), key=lambda pair: pair[1])])
        y_pix_group = np.array([y for (y,x) in sorted(zip(y_pix_group,pa_group), key=lambda pair: pair[1])])
        pa_group = np.array(sorted(pa_group))
        
        img_fronts_agg[group_id] = [pa_group, elon_group]
        img_fronts_pixels_agg[group_id] = [x_pix_group, y_pix_group]
        
        if areas != None:
            img_fronts_areas_agg[group_id] = areas_group

    if areas != None:
        return img_fronts_agg, img_fronts_pixels_agg, img_fronts_areas_agg
    else:
        return img_fronts_agg, img_fronts_pixels_agg

def identify_new_cme(associated_fronts, threshold_elon=8):
    """
    Identifies which CME fronts could be the start of a new CME based on mean elongation.

    Args:
        associated_fronts (list): List of fronts (each a [PA, elongation] 2-array).
        threshold_elon (float): Maximum mean elongation for a front to be flagged as a potential new CME.

    Returns:
        list: Binary flags (1 = could be new CME, 0 = not) for each front.
    """
    flags = []
    for front in associated_fronts:
        pa, elon = front
        mean_elon = np.mean(elon)
        if mean_elon < threshold_elon:
            flags.append(1)
        else:
            flags.append(0)
    return flags

def connect_cmes_new(input_imgs, input_dates, img_front_wcs, img_front_pixels, input_dates_obs, labeled_areas=None, max_time_gap=140, max_start_elon=7.3, min_elon_end=12.9, elon_diff_thresholds=[-3.2,6.7], min_total_duration=None):

    previous_date = input_dates[0]
    previous_fronts = []
    cme_counter = -1
    cme_dictionary = {} 

    previous_cme_keys = []  # tracks which CME string each previous front belongs to

    for idx in range(len(input_imgs)):
        matched_previous_fronts = []

        current_date = input_dates[idx]

        delta_t = (current_date - previous_date).total_seconds()/60

        skip_val = 0

        for frnt in img_front_wcs[idx]:
            if len(frnt[0]) == 0 or len(frnt[1]) == 0:
                skip_val += 1
        
        if skip_val == len(img_front_wcs[idx]):
            print(f"Skipping {current_date} due to no fronts detected")
            continue

        if labeled_areas != None:
            current_fronts, current_fronts_pix, current_areas = aggregate_fronts(img_front_wcs[idx], img_front_pixels[idx], max_diff_elon=3, areas=labeled_areas[idx])
        else:
            current_fronts, current_fronts_pix = aggregate_fronts(img_front_wcs[idx], img_front_pixels[idx], max_diff_elon=3)
        
        new_cme_flag = identify_new_cme(current_fronts, threshold_elon=max_start_elon)
        current_cme_keys = []

        #print('delta_t', delta_t)
        if len(current_fronts) > 0 and len(previous_fronts) > 0 and delta_t < max_time_gap:
            #print(f"CME continues from {previous_date} to {current_date}")
            elongation_differences = [[] for _ in range(len(current_fronts))]

            for i, current in enumerate(current_fronts):
                pa_current, elon_current = current
                pa_current = np.round(pa_current).astype(int)
                elon_current = np.array(elon_current)

                for j, previous in enumerate(previous_fronts):
                    pa_previous, elon_previous = previous
                    pa_previous = np.round(pa_previous).astype(int)
                    elon_previous = np.array(elon_previous)

                    # Build common PA grid
                    all_pas = np.union1d(pa_current, pa_previous)
                    elon_current_grid = np.full_like(all_pas, np.nan, dtype=np.float32)
                    elon_previous_grid = np.full_like(all_pas, np.nan, dtype=np.float32)

                    current_map = dict(zip(pa_current, elon_current))
                    previous_map = dict(zip(pa_previous, elon_previous))

                    for k, pa in enumerate(all_pas):
                        if pa in current_map:
                            elon_current_grid[k] = current_map[pa]
                        if pa in previous_map:
                            elon_previous_grid[k] = previous_map[pa]

                    shared_mask = ~np.isnan(elon_current_grid) & ~np.isnan(elon_previous_grid)
                    if np.any(shared_mask):
                        diff_vector = elon_current_grid[shared_mask] - elon_previous_grid[shared_mask]
                        mean_diff = np.mean(diff_vector)
                        abs_mean_diff = np.mean(np.abs(diff_vector))
                        weight = np.sum(shared_mask) / len(all_pas)
                        weighted_diff = abs_mean_diff * (1 - weight)

                        if elon_diff_thresholds[0] <= mean_diff <= elon_diff_thresholds[1]:
                            elongation_differences[i].append((j, weighted_diff))

            best_matches = []
            for diffs in elongation_differences:
                if diffs:
                    best_match = min(diffs, key=lambda x: x[1])  # minimum absolute mean difference
                    best_matches.append(best_match[0])
                else:
                    best_matches.append(None)


            for i, match in enumerate(best_matches):
                if match is not None:
                    #print(f"Front {i} continues")
                    cme_key = previous_cme_keys[match]
                    cme_dictionary[cme_key]['cme_front_wcs'].append(current_fronts[i])
                    cme_dictionary[cme_key]['cme_front_pixels'].append(current_fronts_pix[i])
                    cme_dictionary[cme_key]['times'].append(input_dates[idx])
                    cme_dictionary[cme_key]['times_obs'].append(input_dates_obs[idx])

                    if labeled_areas != None:
                        cme_dictionary[cme_key]['areas'].append(current_areas[i])

                    current_cme_keys.append(cme_key)
                    matched_previous_fronts.append(current_fronts[i])

                elif new_cme_flag[i] == 1:
                    #print(f"Front {i} is a new CME")
                    cme_counter += 1
                    new_key = f"CME_{cme_counter}"

                    if labeled_areas != None:
                        cme_dictionary[new_key] = {
                            'cme_front_wcs': [current_fronts[i]],
                            'cme_front_pixels': [current_fronts_pix[i]],
                            'times': [input_dates[idx]],
                            'times_obs': [input_dates_obs[idx]],
                            'areas': [current_areas[i]]
                        }

                    else:
                        cme_dictionary[new_key] = {
                            'cme_front_wcs': [current_fronts[i]],
                            'cme_front_pixels': [current_fronts_pix[i]],
                            'times': [input_dates[idx]],
                            'times_obs': [input_dates_obs[idx]]
                        }     

                    current_cme_keys.append(new_key)
                    matched_previous_fronts.append(current_fronts[i])

            previous_date = current_date
            previous_fronts = matched_previous_fronts.copy()
            previous_cme_keys = current_cme_keys

        elif len(previous_fronts) == 0:
            #print(f'New sequence at {current_date}')
            for i in range(len(current_fronts)):
                if new_cme_flag[i] == 1:
                    #print(f'Front {i} is new CME')
                    cme_counter += 1
                    new_key = f"CME_{cme_counter}"
                    
                    if labeled_areas != None: 
                        cme_dictionary[new_key] = {
                            'cme_front_wcs': [current_fronts[i]],
                            'cme_front_pixels': [current_fronts_pix[i]],
                            'times': [input_dates[idx]],
                            'times_obs': [input_dates_obs[idx]],
                            'areas': [current_areas[i]]
                        }
                    
                    else:
                        cme_dictionary[new_key] = {
                            'cme_front_wcs': [current_fronts[i]],
                            'cme_front_pixels': [current_fronts_pix[i]],
                            'times': [input_dates[idx]],
                            'times_obs': [input_dates_obs[idx]]
                        }
                                            
                    current_cme_keys.append(new_key)
                    matched_previous_fronts.append(current_fronts[i])

            previous_date = current_date
            previous_fronts = matched_previous_fronts.copy()
            previous_cme_keys = current_cme_keys
        
        else:
            previous_date = current_date
            previous_fronts = []
            current_fronts = []
            previous_cme_keys = []

    if min_total_duration is not None:
        for entry in list(cme_dictionary):
            time_beginning = sorted(list(set(cme_dictionary[entry]['times'])))[0]
            time_end = sorted(list(set(cme_dictionary[entry]['times'])))[-1]
            time_diff = (time_end - time_beginning).total_seconds()/3600 # total duration of CME in hours

            if time_diff < min_total_duration:
                del cme_dictionary[entry]

    for entry in list(cme_dictionary):
        if np.nanmax(cme_dictionary[entry]['cme_front_wcs'][-1][1]) < min_elon_end: #if maximum elongation reached is below threshold
            del cme_dictionary[entry]

    # Merge duplicate timestep entries
    for cme_key, cme_data in cme_dictionary.items():
        merged_data = defaultdict(lambda: {'wcs': [], 'pixels': [], 'areas': []})
        
        for i, time in enumerate(cme_data['times']):
            merged_data[time]['wcs'].append(cme_data['cme_front_wcs'][i])
            merged_data[time]['pixels'].append(cme_data['cme_front_pixels'][i])
            if 'areas' in cme_data:
                merged_data[time]['areas'].append(cme_data['areas'][i])
        
        # Sort times
        sorted_times = sorted(merged_data.keys())
        
        cme_data['times'] = sorted_times
        cme_data['times_obs'] = [cme_data['times_obs'][cme_data['times'].index(t)] for t in sorted_times]
        cme_data['cme_front_wcs'] = []
        cme_data['cme_front_pixels'] = []

        if 'areas' in cme_data:
            cme_data['areas'] = []

        for t in sorted_times:
            cme_data['cme_front_wcs'].append(np.concatenate(merged_data[t]['wcs'], axis=1))
            cme_data['cme_front_pixels'].append(np.concatenate(merged_data[t]['pixels'], axis=1))
            if 'areas' in cme_data:
                cme_data['areas'].append(np.sum(merged_data[t]['areas'], axis=0))

    # Sort by PA value

    for cme_key, cme_data in cme_dictionary.items():
        for i, time in enumerate(cme_data['times']):

            # Extract current WCS and pixel data
            wcs_data = cme_data['cme_front_wcs'][i]
            pixel_data = cme_data['cme_front_pixels'][i]

            # Shape is (2, N) with [PA, elongation]
            pa_values = wcs_data[0]
            elongations = wcs_data[1]

            x_pix_values = pixel_data[0]
            y_pix_values = pixel_data[1]

            elongations = np.array([e for _, e in sorted(zip(pa_values, elongations), key=lambda pair: pair[0])])
            x_pix_values = np.array([x for _, x in sorted(zip(pa_values, x_pix_values), key=lambda pair: pair[0])])
            y_pix_values = np.array([y for _, y in sorted(zip(pa_values, y_pix_values), key=lambda pair: pair[0])])
            pa_values = np.array(sorted(pa_values))

            wcs_data = [pa_values,elongations]
            pixel_data = [x_pix_values, y_pix_values]

            cme_data['cme_front_wcs'][i] = wcs_data
            cme_data['cme_front_pixels'][i] = pixel_data

    cme_names = np.array(['CME_' + str(i) for i in range(len(cme_dictionary))])
    
    cme_dictionary_clean = {}

    for i, entry in enumerate(list(cme_dictionary)):
        cme_dictionary_clean[cme_names[i]] = cme_dictionary[entry].copy()

    return cme_dictionary_clean

def remove_outliers_from_fronts(img_front_wcs, img_fronts_pixels, fits_headers, image_areas=None, window=3, threshold=1.5):
    cleaned_fronts = []
    cleaned_fronts_pixels = []

    if image_areas is not None:
        cleaned_fronts_areas = []

    for img_idx, img_fronts in enumerate(img_front_wcs):
        cleaned_img = []
        cleaned_pixels = []

        if image_areas is not None:
            cleaned_areas = []

        wcoord = wcs.WCS(fits_headers[img_idx])

        for front_idx, front in enumerate(img_fronts):

            pa_array, elon_array = front
            pixel_x_array = np.array(img_fronts_pixels[img_idx][front_idx])[0]
            pixel_y_array = np.array(img_fronts_pixels[img_idx][front_idx])[1]

            pa_array = np.round(pa_array).astype(int)
            elon_array = np.array(elon_array, dtype=np.float32)

            if len(pa_array) < 6:
                # Not enough data to apply outlier detection
                cleaned_img.append(np.array([pa_array, elon_array]))
                cleaned_pixels.append(np.array([pixel_x_array, pixel_y_array]))

                if image_areas is not None:
                    cleaned_areas.append(image_areas[img_idx][front_idx])
                    
                continue

            cleaned_elon = elon_array.copy()
            outlier_mask = np.zeros_like(elon_array, dtype=bool)

            cleaned_pix_arr = np.array([pixel_x_array, pixel_y_array])
            
            for i, pa in enumerate(pa_array):
                # Look for surrounding PAs within ±window
                surrounding_values_left = []
                surrounding_values_right = []

                for offset in range(1, window + 1):
                    neighbor_pa_left = pa - offset
                    neighbor_pa_right = pa + offset

                    if neighbor_pa_left in pa_array:
                        idx_left = np.where(pa_array == neighbor_pa_left)[0][0]
                        surrounding_values_left.append(elon_array[idx_left])      

                    if neighbor_pa_right in pa_array:
                        idx_right = np.where(pa_array == neighbor_pa_right)[0][0]
                        surrounding_values_right.append(elon_array[idx_right])

                if len(surrounding_values_left) >= 2 and len(surrounding_values_right) >= 2:
                    surrounding_mean_left = np.nanmean(surrounding_values_left)
                    surrounding_mean_right = np.nanmean(surrounding_values_right)
                    diff_left = np.abs(elon_array[i] - surrounding_mean_left)
                    diff_right = np.abs(elon_array[i] - surrounding_mean_right)

                    if diff_left > threshold and diff_right > threshold:
                        outlier_mask[i] = True

                elif len(surrounding_values_left) >= 2:
                    surrounding_mean_left = np.nanmean(surrounding_values_left)
                    diff_left = np.abs(elon_array[i] - surrounding_mean_left)

                    if diff_left > threshold:
                        outlier_mask[i] = True
                
                elif len(surrounding_values_right) >= 2:
                    surrounding_mean_right = np.nanmean(surrounding_values_right)
                    diff_right = np.abs(elon_array[i] - surrounding_mean_right)

                    if diff_right > threshold:
                        outlier_mask[i] = True

            # Interpolate over outliers
            if np.any(outlier_mask):
                good_idx = ~outlier_mask

                if np.any(good_idx):
                    cleaned_elon[outlier_mask] = np.interp(
                        pa_array[outlier_mask],
                        pa_array[good_idx],
                        elon_array[good_idx]
                    )
                    cleaned_elon_rad = np.deg2rad(cleaned_elon)
                    pa_array_rad = np.deg2rad(pa_array)

                    tx = np.rad2deg(np.arctan2(-np.sin(cleaned_elon_rad)*np.sin(pa_array_rad), np.cos(cleaned_elon_rad)))
                    ty = np.rad2deg(np.arcsin(np.sin(cleaned_elon_rad)*np.cos(pa_array_rad)))

                    pix_y, pix_x = wcoord.all_world2pix(tx,ty, 0)

                    if np.any(np.round(pix_x/8,0).astype(int) > 128):
                        for remove_ind in reversed(np.where(np.round(pix_x/8,0).astype(int) > 128)[0]):
                            pa_array = np.delete(pa_array, remove_ind)
                            elon_array = np.delete(elon_array, remove_ind)
                            cleaned_elon = np.delete(cleaned_elon, remove_ind)
                            pix_y = np.delete(pix_y, remove_ind)
                            pix_x = np.delete(pix_x, remove_ind)

                    if np.any(np.round(pix_x/8,0).astype(int) < 0):
                        for remove_ind in reversed(np.where(np.round(pix_x/8,0).astype(int) < 0)[0]):
                            pa_array = np.delete(pa_array, remove_ind)
                            elon_array = np.delete(elon_array, remove_ind)
                            cleaned_elon = np.delete(cleaned_elon, remove_ind)
                            pix_y = np.delete(pix_y, remove_ind)
                            pix_x = np.delete(pix_x, remove_ind)

                    if np.any(np.round(pix_y/8,0).astype(int) > 128):
                        for remove_ind in reversed(np.where(np.round(pix_y/8,0).astype(int) > 128)[0]):
                            pa_array = np.delete(pa_array, remove_ind)
                            elon_array = np.delete(elon_array, remove_ind)
                            cleaned_elon = np.delete(cleaned_elon, remove_ind)
                            pix_y = np.delete(pix_y, remove_ind)
                            pix_x = np.delete(pix_x, remove_ind)

                    if np.any(np.round(pix_y/8,0).astype(int) < 0):
                        for remove_ind in reversed(np.where(np.round(pix_y/8,0).astype(int) < 0)[0]):
                            pa_array = np.delete(pa_array, remove_ind)
                            elon_array = np.delete(elon_array, remove_ind)
                            cleaned_elon = np.delete(cleaned_elon, remove_ind)
                            pix_y = np.delete(pix_y, remove_ind)
                            pix_x = np.delete(pix_x, remove_ind)

                    cleaned_pix_arr = np.array([np.round(pix_x/8,0).astype(int), np.round(pix_y/8,0).astype(int)])

                else:
                    # All points in the front are considered outliers, discard front
                    continue

            cleaned_pixels.append(np.array(cleaned_pix_arr))
            cleaned_img.append(np.array([pa_array, cleaned_elon]))

            if image_areas is not None:
                cleaned_areas.append(image_areas[img_idx][front_idx])

        cleaned_fronts.append(cleaned_img)
        cleaned_fronts_pixels.append(cleaned_pixels)

        if image_areas is not None:
            cleaned_fronts_areas.append(cleaned_areas)

    if image_areas is not None:
        return cleaned_fronts, cleaned_fronts_pixels, cleaned_fronts_areas
    
    else:
        return cleaned_fronts,cleaned_fronts_pixels

def generate_helcats_gt_corrected(helcats_path,corrected_helcats_path,fits_path,pair):

    start = datetime.datetime.strptime(pair['start'], '%Y_%m_%d')
    end = datetime.datetime.strptime(pair['end'], '%Y_%m_%d')

    helcats_files = sorted([helcats_path + file for file in os.listdir(helcats_path) if file.endswith('.dat')
                            and datetime.datetime.strptime(file.split('_')[3], '%Y%m%d') >= start
                            and datetime.datetime.strptime(file.split('_')[3], '%Y%m%d') <= end
                            and 'HCME_A__' in file])

    if not os.path.exists(corrected_helcats_path):
        os.makedirs(corrected_helcats_path)

    for file in helcats_files:

        helcats_arr = list(np.genfromtxt(file, delimiter=[5,28,14,11,4],autostrip=True,encoding='utf-8', dtype=None))
        for i in range(len(helcats_arr)):
            helcats_arr[i] = np.array(list(helcats_arr[i]),dtype=object)
            
        helcats_arr = np.array(helcats_arr)
        helcats_times = np.array([datetime.datetime.strptime(timestmp, '%Y-%m-%dT%H:%M:%S.%f') for timestmp in helcats_arr[:,1]])
        helats_times_set = set([timestmp[:10].replace('-', '') for timestmp in helcats_arr[:,1]])
        
        fits_files = []

        for settime in helats_times_set:
            fls = sorted(glob.glob(fits_path + settime + '/*_s4h1A.fts'))
            fits_files.extend(fls)
        
        fits_times = []
        fits_times_end = []
        wcoords = []
        

        for fname in fits_files:
            hdul = fits.open(fname)
            
            hdr_date_obs = datetime.datetime.strptime(hdul[0].header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
            hdr_date_end = datetime.datetime.strptime(hdul[0].header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f')

            fits_times.append(hdr_date_obs)
            fits_times_end.append(hdr_date_end)
            wcoords.append(wcs.WCS(hdul[0].header))

        fits_times = np.array(fits_times)
        corrected_helcats_times = []
        corrected_helcats_times_end = []
        wcoords_corrected = []

        for hctime in helcats_times:
            min_time_ind = np.argmin(np.abs(fits_times-hctime))

            closest_time = fits_times[min_time_ind]
            closest_time_end = fits_times_end[min_time_ind]
            wcoords_corrected.append(wcoords[min_time_ind])

            corrected_helcats_times.append(datetime.datetime.strftime(closest_time, '%Y-%m-%dT%H:%M:%S'))
            corrected_helcats_times_end.append(datetime.datetime.strftime(closest_time_end, '%Y-%m-%dT%H:%M:%S'))

        pix_coordinates_x = np.zeros(len(helcats_times))
        pix_coordinates_y = np.zeros(len(helcats_times))

        for i in range(len(helcats_arr)):
            elon = np.deg2rad(helcats_arr[i][2])
            pa = np.deg2rad(helcats_arr[i][3])

            tx = np.rad2deg(np.arctan2(-np.sin(elon)*np.sin(pa), np.cos(elon)))
            ty = np.rad2deg(np.arcsin(np.sin(elon)*np.cos(pa)))

            pix_y, pix_x = wcoords_corrected[i].all_world2pix(tx,ty, 0)

            pix_coordinates_x[i] = int(np.round(pix_x/8,0))
            pix_coordinates_y[i] = int(np.round(pix_y/8,0))

        corrected_helcats_times_end = np.array(corrected_helcats_times_end).astype('U19')
        corrected_helcats_times = np.array(corrected_helcats_times).astype('U19')

        helcats_arr_corrected = helcats_arr.copy()
        for i in range(len(helcats_arr_corrected)):
            helcats_arr_corrected[i][1] = corrected_helcats_times[i]
        
        helcats_arr_corrected = np.insert(helcats_arr_corrected, 2, corrected_helcats_times_end, axis=1)
        helcats_arr_corrected = np.insert(helcats_arr_corrected, 5, pix_coordinates_x, axis=1)
        helcats_arr_corrected = np.insert(helcats_arr_corrected, 6, pix_coordinates_y, axis=1)

        np.savetxt(corrected_helcats_path+file.split('/')[-1], helcats_arr_corrected, delimiter=",",encoding='utf-8',header='track_no,time_obs,time_end,elon,pa,pix_x,pix_y,sc',fmt=('%i,%s,%s,%.4f,%i,%i,%i,%s'))

def get_helcats_tracks(folder_path, save_path, pair):

    start = datetime.datetime.strptime(pair['start'], '%Y_%m_%d')
    end = datetime.datetime.strptime(pair['end'], '%Y_%m_%d')

    cme_validation_dict = {}

    for filename in sorted(os.listdir(folder_path)):
        helcats_date = datetime.datetime.strptime(filename.split('_')[3], '%Y%m%d')
        if not filename.endswith('.dat') or helcats_date < start or helcats_date > end:
            continue
        
        filepath = os.path.join(folder_path, filename)

        data = np.loadtxt(filepath, 
                          dtype={'names': ('track_no','time_obs','time_end','elon','pa','pix_x','pix_y','sc'),
                                 'formats': ('i', 'U19', 'U19','f','i','i','i','U1')},
                          delimiter=',', encoding='utf-8')

        # Filter out elongation > 19 degrees
        data = data[data['elon'] <= 19]

        # Group by track_no
        tracks = defaultdict(list)
        for row in data:
            tracks[row['track_no']].append(row)

        # Get beginning times per track
        track_start_times = {}
        for track_no, entries in tracks.items():
            sorted_entries = sorted(entries, key=lambda r: r['time_obs'])
            start_time = datetime.datetime.strptime(sorted_entries[0]['time_obs'], '%Y-%m-%dT%H:%M:%S')
            track_start_times[track_no] = start_time

        if not track_start_times:
            continue  # Skip empty or filtered-out files

        # Find median start time
        all_start_times = list(track_start_times.values())
        median_time = sorted(all_start_times)[len(all_start_times)//2]

        # Get track_no that corresponds to median_time
        chosen_track_no = [track_no for track_no, time in track_start_times.items() if time == median_time][0]
        chosen_track = sorted(tracks[chosen_track_no], key=lambda r: r['time_obs'])
        # Format for saving
        cme_key = filename.split('/')[-1][:19]

        # chosen_pa = np.array([entry['pa'] for entry in chosen_track]).T
        # chosen_elon = np.array([entry['elon'] for entry in chosen_track]).T

        cme_validation_dict[cme_key] = {
            'times_obs': [datetime.datetime.strptime(entry['time_obs'], '%Y-%m-%dT%H:%M:%S') for entry in chosen_track],
            'times': [datetime.datetime.strptime(entry['time_end'], '%Y-%m-%dT%H:%M:%S') for entry in chosen_track],
            'cme_front_wcs': [[entry['pa'], entry['elon']] for entry in chosen_track],
            'cme_front_pixels': [[entry['pix_x'], entry['pix_y']] for entry in chosen_track]
        }

    # Save dictionary
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    np.save(save_path, cme_validation_dict)

def get_cmes(labeled_clusters_connected, return_area=False):
    print('Getting CMEs...')
    pad_width = 3
    image_outlines = []

    if return_area:
        areas_connected = []

    for k,label_img in enumerate(labeled_clusters_connected):
        
        outl = []

        if return_area:
            outa = []

        if np.max(label_img) != 0:
            for h in range(1,np.max(label_img)+1):
                
                temp_img = np.where(label_img == h, 1, 0).astype(np.float64)
                temp_img = gaussian(temp_img, sigma=3)
                temp_img = np.where(temp_img > 0.5, 1, 0).astype(np.float64)
                temp_img = np.pad(temp_img, pad_width=pad_width, mode='constant', constant_values=0)
                
                edge_img = find_boundaries(temp_img, mode='inner').astype(np.float64)

                #edge_img = feature.canny(temp_img, sigma=3)
                edge_img = edge_img[pad_width:-pad_width, pad_width:-pad_width]
                edge_ind = np.where(edge_img == 1)

                temp_img = temp_img[pad_width:-pad_width, pad_width:-pad_width]
                outl.append(edge_ind)

                if return_area:
                    outa.append(temp_img)

        image_outlines.append(outl)

        if return_area:
            areas_connected.append(outa)
        
    if return_area:
        return image_outlines, areas_connected
    else:
        return image_outlines
    
def get_ml_gt(annotation_path, filenames):
    #Separates annotations into lists of CMEs
    with open(annotation_path) as f:
        annotation_file = json.load(f)
    
    image_ids_dict = {}
    datetime_dict = {}
    for i in range(len(annotation_file['images'])):
        image_ids_dict[annotation_file['images'][i]['id']] = annotation_file['images'][i]['file_name']

    cme_dict = {}

    for i in range(len(annotation_file['annotations'])):
        fname = image_ids_dict[annotation_file['annotations'][i]['image_id']].split('/')[-1].split('.')[0]
        if fname[:15] in filenames:
            #TODO: Change hard-coded paths
            fits_file = '/media/DATA_DRIVE/stereo_processed/reduced/data/A/' + fname[:8] + '/science/hi_1/'
            fits_hdul = fits.open(fits_file+fname+'.fts')
            date_obs = datetime.datetime.strptime(fits_hdul[0].header['DATE-OBS'][:19], '%Y-%m-%dT%H:%M:%S')
            fits_hdul.close()

            datetime_dict[fname] = date_obs
            
            if fname not in cme_dict.keys():
                cme_dict[fname] = {'annotation_info':[], 'areas':[]}

            cme_dict[fname]['annotation_info'].append(annotation_file['annotations'][i])

            temp_ann = Image.fromarray((np.array(coco.maskUtils.decode(coco.maskUtils.frPyObjects([annotation_file['annotations'][i]['segmentation']], 1024, 1024))[:,:,0])*255).astype(np.uint8)).convert("L")

            temp_ann = temp_ann.resize((128, 128))
            temp_ann = (np.array(temp_ann)/255).astype(np.uint8)

            dilation = True

            if dilation:
                kernel = disk(2)
                n_it = int(2)
                
                temp_ann = ndimage.binary_dilation(temp_ann, structure=kernel, iterations=n_it)

            cme_dict[fname]['areas'].append(temp_ann)

    cme_list = {}
    prev_img_id = -1
    prev_cme_ids = []
    cme_temp_names = {}

    for tim in cme_dict.keys():
        if cme_dict[tim] != ['None']:
            img_id = cme_dict[tim]['annotation_info'][0]['image_id']
            cme_ids = [cme_dict[tim]['annotation_info'][i]['attributes']['id'] for i in range(len(cme_dict[tim]['annotation_info']))]

            if (img_id - prev_img_id > 1) or (prev_img_id == -1):
                cme_temp_names = {}
                for num,id in enumerate(cme_ids):
                    cme = '_'.join(tim.split('_')[0:2]) + '_CME_' + str(id)
                    cme_temp_names[str(id)] = cme

                    cme_list[cme] = {'times':[], 'times_obs':[],'areas': []}

                    cme_dict[tim]['annotation_info'][num].update({'image_time': datetime.datetime.strptime('_'.join(tim.split('_')[0:2]), '%Y%m%d_%H%M%S')})
                    cme_dict[tim]['annotation_info'][num].update({'time_obs': datetime_dict[tim]})

                    cme_list[cme]['times_obs'].append(cme_dict[tim]['annotation_info'][num]['time_obs'])
                    cme_list[cme]['times'].append(cme_dict[tim]['annotation_info'][num]['image_time'])
                    cme_list[cme]['areas'].append(cme_dict[tim]['areas'][num])

            elif img_id - prev_img_id == 1:
                diff_ids_remove = np.setdiff1d(prev_cme_ids, cme_ids)

                for rem_ids in diff_ids_remove:
                    prev_cme_ids.remove(rem_ids)
                    del cme_temp_names[str(rem_ids)]

                diff_ids_add = np.setdiff1d(cme_ids, prev_cme_ids)

                for num, add_ids in enumerate(diff_ids_add):
                    cme = '_'.join(tim.split('_')[0:2]) + '_CME_' + str(add_ids)
                    cme_temp_names[str(add_ids)] = cme


                for num,cme_id in enumerate(cme_ids):
                    if cme_id in prev_cme_ids:
                        cme = cme_temp_names[str(cme_id)]
                        cme_dict[tim]['annotation_info'][num].update({'image_time': datetime.datetime.strptime('_'.join(tim.split('_')[0:2]), '%Y%m%d_%H%M%S')})
                        cme_dict[tim]['annotation_info'][num].update({'time_obs': datetime_dict[tim]})
                        
                        cme_list[cme]['times_obs'].append(cme_dict[tim]['annotation_info'][num]['time_obs'])
                        cme_list[cme]['times'].append(cme_dict[tim]['annotation_info'][num]['image_time'])
                        cme_list[cme]['areas'].append(cme_dict[tim]['areas'][num])
                    else:
                        cme = '_'.join(tim.split('_')[0:2]) + '_CME_' + str(cme_id)
                        cme_list[cme] = {'times':[], 'times_obs':[], 'areas': []}
                        cme_dict[tim]['annotation_info'][num].update({'image_time': datetime.datetime.strptime('_'.join(tim.split('_')[0:2]), '%Y%m%d_%H%M%S')})
                        cme_dict[tim]['annotation_info'][num].update({'time_obs': datetime_dict[tim]})

                        cme_list[cme]['times_obs'].append(cme_dict[tim]['annotation_info'][num]['time_obs'])
                        cme_list[cme]['times'].append(cme_dict[tim]['annotation_info'][num]['image_time'])
                        cme_list[cme]['areas'].append(cme_dict[tim]['areas'][num])

            prev_cme_ids = cme_ids
            prev_img_id = img_id

        else:
            continue

    return cme_list

def evaluate_matches(matches, unmatched_gt, unmatched_pred):
    start_errors = []
    end_errors = []

    
    num_true_positives = len(matches)
    num_false_positives = len(unmatched_pred)
    num_false_negatives = len(unmatched_gt)
    total_cmes = num_true_positives + num_false_negatives

    iou = np.round(num_true_positives/(num_true_positives+num_false_positives+num_false_negatives),2)
    precision = np.round(num_true_positives/(num_true_positives+num_false_positives),2)
    recall = np.round(num_true_positives/(num_true_positives+num_false_negatives),2)

    # === Collect matched stats ===
    for pred_name, match_info in matches.items():
        start_errors.append(match_info["start_error"])
        end_errors.append(match_info["end_error"])


    result = {
        "mean_start_error": np.round(np.mean(start_errors),2),
        "mean_end_error": np.round(np.mean(end_errors),2),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "num_true_positives": num_true_positives,
        "num_false_positives": num_false_positives,
        "num_false_negatives": num_false_negatives,
        "total_cmes_gt": total_cmes,
    }

    return result

def create_dictionaries_operational(load_path, save_path, t=0.90):
    print('Creating dictionaries...')
    filenames, pred = load_results(load_path)
    filenames = [name.decode('utf-8') for name in filenames]

    # Convert dates in input_names to datetime objects
    input_dates = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d_%H%M%S') for name in filenames if name.split('/')[-1].startswith(year)]
    input_days = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d') for name in filenames if name.split('/')[-1].startswith(year)]
    input_days_set = sorted(set(input_days))
    pred = pred[:len(input_dates)]

    images_proc, labeled_clusters_connected = post_processing(pred, t=t)

    image_outlines = get_cmes(labeled_clusters_connected,return_area=False)

    fits_headers = get_fitsfiles(input_days_set, input_dates)

    input_dates_obs = [datetime.datetime.strptime(fits_headers[i]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') for i in range(len(fits_headers))]

    image_outlines_wcs, image_outline_pixels = get_outline(image_outlines, fits_headers)

    img_front_wcs, img_front_pixels = get_front(image_outlines_wcs, image_outline_pixels)

    img_front_wcs_clean, img_front_pixels_clean = remove_outliers_from_fronts(img_front_wcs, img_front_pixels, fits_headers, window=3, threshold=1)

    input_dates_obs = [datetime.datetime.strptime(datetime.datetime.strptime(fits_headers[i]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S'),'%Y%m%d_%H%M%S') for i in range(len(fits_headers))]
    input_dates_datetime = [datetime.datetime.strptime(name[:15], '%Y%m%d_%H%M%S') for name in input_dates]

    cme_dictionary = connect_cmes_new(images_proc, input_dates_datetime, img_front_wcs_clean, img_front_pixels_clean, input_dates_obs)
    
    np.save(save_path, cme_dictionary)


def get_ml_gt_science_all(annotation_path):
    #Separates annotations into lists of CMEs
    with open(annotation_path) as f:
        annotation_file = json.load(f)
    
    image_ids_dict = {}
    datetime_dict = {}
    for i in range(len(annotation_file['images'])):
        image_ids_dict[annotation_file['images'][i]['id']] = annotation_file['images'][i]['file_name']

    cme_dict = {}

    for i in range(len(annotation_file['annotations'])):
        fname = image_ids_dict[annotation_file['annotations'][i]['image_id']].split('/')[-1].split('.')[0]
        #TODO: Change hard-coded paths
        fits_file = '/media/DATA_DRIVE/stereo_processed/reduced/data/A/' + fname[:8] + '/science/hi_1/'
        fits_hdul = fits.open(fits_file+fname+'.fts')
        date_obs = datetime.datetime.strptime(fits_hdul[0].header['DATE-OBS'][:19], '%Y-%m-%dT%H:%M:%S')
        fits_hdul.close()

        datetime_dict[fname] = date_obs
        
        if fname not in cme_dict.keys():
            cme_dict[fname] = {'annotation_info':[], 'areas':[]}

        cme_dict[fname]['annotation_info'].append(annotation_file['annotations'][i])

        temp_ann = Image.fromarray((np.array(coco.maskUtils.decode(coco.maskUtils.frPyObjects([annotation_file['annotations'][i]['segmentation']], 1024, 1024))[:,:,0])*255).astype(np.uint8)).convert("L")

        temp_ann = temp_ann.resize((128, 128))
        temp_ann = (np.array(temp_ann)/255).astype(np.uint8)

        dilation = True

        if dilation:
            kernel = disk(2)
            n_it = int(2)
            
            temp_ann = ndimage.binary_dilation(temp_ann, structure=kernel, iterations=n_it)

        cme_dict[fname]['areas'].append(temp_ann)

    cme_list = {}
    prev_img_id = -1
    prev_cme_ids = []
    cme_temp_names = {}

    for tim in cme_dict.keys():
        if cme_dict[tim] != ['None']:
            img_id = cme_dict[tim]['annotation_info'][0]['image_id']
            cme_ids = [cme_dict[tim]['annotation_info'][i]['attributes']['id'] for i in range(len(cme_dict[tim]['annotation_info']))]

            if (img_id - prev_img_id > 1) or (prev_img_id == -1):
                cme_temp_names = {}
                for num,id in enumerate(cme_ids):
                    cme = '_'.join(tim.split('_')[0:2]) + '_CME_' + str(id)
                    cme_temp_names[str(id)] = cme

                    cme_list[cme] = {'times':[], 'times_obs':[],'areas': []}

                    cme_dict[tim]['annotation_info'][num].update({'image_time': datetime.datetime.strptime('_'.join(tim.split('_')[0:2]), '%Y%m%d_%H%M%S')})
                    cme_dict[tim]['annotation_info'][num].update({'time_obs': datetime_dict[tim]})

                    cme_list[cme]['times_obs'].append(cme_dict[tim]['annotation_info'][num]['time_obs'])
                    cme_list[cme]['times'].append(cme_dict[tim]['annotation_info'][num]['image_time'])
                    cme_list[cme]['areas'].append(cme_dict[tim]['areas'][num])

            elif img_id - prev_img_id == 1:
                diff_ids_remove = np.setdiff1d(prev_cme_ids, cme_ids)

                for rem_ids in diff_ids_remove:
                    prev_cme_ids.remove(rem_ids)
                    del cme_temp_names[str(rem_ids)]

                diff_ids_add = np.setdiff1d(cme_ids, prev_cme_ids)

                for num, add_ids in enumerate(diff_ids_add):
                    cme = '_'.join(tim.split('_')[0:2]) + '_CME_' + str(add_ids)
                    cme_temp_names[str(add_ids)] = cme


                for num,cme_id in enumerate(cme_ids):
                    if cme_id in prev_cme_ids:
                        cme = cme_temp_names[str(cme_id)]
                        cme_dict[tim]['annotation_info'][num].update({'image_time': datetime.datetime.strptime('_'.join(tim.split('_')[0:2]), '%Y%m%d_%H%M%S')})
                        cme_dict[tim]['annotation_info'][num].update({'time_obs': datetime_dict[tim]})
                        
                        cme_list[cme]['times_obs'].append(cme_dict[tim]['annotation_info'][num]['time_obs'])
                        cme_list[cme]['times'].append(cme_dict[tim]['annotation_info'][num]['image_time'])
                        cme_list[cme]['areas'].append(cme_dict[tim]['areas'][num])
                    else:
                        cme = '_'.join(tim.split('_')[0:2]) + '_CME_' + str(cme_id)
                        cme_list[cme] = {'times':[], 'times_obs':[], 'areas': []}
                        cme_dict[tim]['annotation_info'][num].update({'image_time': datetime.datetime.strptime('_'.join(tim.split('_')[0:2]), '%Y%m%d_%H%M%S')})
                        cme_dict[tim]['annotation_info'][num].update({'time_obs': datetime_dict[tim]})

                        cme_list[cme]['times_obs'].append(cme_dict[tim]['annotation_info'][num]['time_obs'])
                        cme_list[cme]['times'].append(cme_dict[tim]['annotation_info'][num]['image_time'])
                        cme_list[cme]['areas'].append(cme_dict[tim]['areas'][num])

            prev_cme_ids = cme_ids
            prev_img_id = img_id

        else:
            continue

    return cme_list

def get_ml_gt_beacon(cme_dict_gt_science, filenames):
    cme_dict_gt_beacon = {}

    for cme_key in cme_dict_gt_science.keys():
        science_days = list(set(sorted([datetime.datetime.strftime(cme_dict_gt_science[cme_key]['times'][i], '%Y%m%d') for i in range(len(cme_dict_gt_science[cme_key]['times']))])))

        beacon_files = []

        for i in range(len(science_days)):
            beacon_files.extend(glob.glob('/media/DATA_DRIVE/stereo_processed/reduced/data/A/'+science_days[i]+'/beacon/hi_1/*.fts'))
        
        science_times = cme_dict_gt_science[cme_key]['times'].copy()
        beacon_times = [datetime.datetime.strptime(f.split('/')[-1][:15], '%Y%m%d_%H%M%S') for f in beacon_files]

        matched_times = []
        matched_times_obs = []
        matched_areas = []

        for j in range(len(science_times)):
            min_diff = np.nanmin([abs((science_times[j] - beacon_time).total_seconds()) for beacon_time in beacon_times])
            min_arg = np.nanargmin([abs((science_times[j] - beacon_time).total_seconds()) for beacon_time in beacon_times])

            if min_diff < 60*10:
                matched_times.append(beacon_times[min_arg])
                matched_times_obs.append(cme_dict_gt_science[cme_key]['times_obs'][j])
                matched_areas.append(cme_dict_gt_science[cme_key]['areas'][j])
            
        if len(matched_times) > 0:
            cme_dict_gt_beacon[cme_key] = {
                'times': matched_times,
                'times_obs': matched_times_obs,
                'areas': matched_areas
            }
        else:
            print(f"No matching beacon times found for CME {cme_key}. Skipping this CME.")
            continue

    cme_tag = defaultdict(list)

    for fname in filenames:
        ftime = datetime.datetime.strptime(fname[:15], '%Y%m%d_%H%M%S')

        for cme_key in cme_dict_gt_beacon.keys():
            
            if ftime in cme_dict_gt_beacon[cme_key]['times']:
                cme_tag[cme_key].append('test')
            else:
                cme_tag[cme_key].append('other')

    for cme_key in cme_tag.keys():

        if 'test' in cme_tag[cme_key]:
            cme_tag[cme_key] = 'test'
        else:
            cme_tag[cme_key] = 'other'

    cme_set_test = set([cme_key for cme_key in cme_tag.keys() if cme_tag[cme_key] == 'test'])

    cme_dict_gt_beacon_test = {cme_key: cme_dict_gt_beacon[cme_key] for cme_key in cme_set_test}

    return cme_dict_gt_beacon_test

def match_cmes_extra(ground_truth_cmes, predicted_cmes, time_threshold=200):
    Range = namedtuple('Range', ['start', 'end'])
    pred_names = list(predicted_cmes.keys())
    gt_names = list(ground_truth_cmes.keys())

    num_pred = len(pred_names)
    num_gt = len(gt_names)

    cost_matrix = np.full((num_pred, num_gt), 1e10)
    match_info = {}


    for i, pred_name in enumerate(pred_names):
        pred_data = predicted_cmes[pred_name]
        pred_times = np.array(pred_data["times_obs"])
        pred_start = pred_times[0]

        for j, gt_name in enumerate(gt_names):
            gt_data = ground_truth_cmes[gt_name]
            gt_start = gt_data['time_start']

            time_diff = abs((pred_start - gt_start).total_seconds()) / 60
            
            if time_diff > time_threshold:
                continue

            cost_matrix[i, j] = time_diff

            match_info[(i, j)] = {
                "gt_match": gt_name
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

def get_binary_range_revised(prediction_dict, groundtruth_dict, matches, unmatched_gt, extra_gt, pair):

    binary_range = []
    max_rows = 4
    active_rows = [[] for _ in range(max_rows)]
    scatter_extra = []
    #value_dict = {'FP': 1, 'FN': 2, 'TP': 3}

    start = pair['start']
    end = pair['end']

    start_datetime = datetime.datetime.strptime(start, '%Y_%m_%d')
    end_datetime = datetime.datetime.strptime(end, '%Y_%m_%d')

    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='ME')
    #date_range = [datetime.datetime.strftime(dt, '%Y_%m_%d') for dt in date_range]

    for i, year_month_day in enumerate(date_range):
        begin_month = datetime.datetime.strptime(datetime.datetime.strftime(year_month_day.to_pydatetime(),'%Y_%m')+ '_01', '%Y_%m_%d')
        end_month = year_month_day.to_pydatetime()+datetime.timedelta(hours=23)+datetime.timedelta(minutes=59)

        # begin_month = datetime.datetime.strptime(year + str(i).zfill(2) + '01', '%Y%m%d')
        # end_month = datetime.datetime.strptime(year + str(i).zfill(2) + str(calendar.monthrange(int(year), i)[-1]), '%Y%m%d')+datetime.timedelta(hours=23)+datetime.timedelta(minutes=59)

        comp_time = pd.date_range(start=begin_month, end=end_month, freq='h').to_pydatetime()
        comp_zeros = np.zeros(len(comp_time))
        plot_temp = [comp_zeros,comp_zeros,comp_zeros,comp_zeros]
        
        for extra_cme in list(extra_gt.keys()):
            start_time = extra_gt[extra_cme]['time_start']
            if (start_time.month == begin_month.month) and (start_time.year == begin_month.year):
                scatter_extra.append([(start_time.day)+start_time.hour/24,(i+1)+1.5/(max_rows+1)])


        for cme in list(prediction_dict.keys())+unmatched_gt:

            if cme in prediction_dict.keys():
                current_start = prediction_dict[cme]['times'][0]
                current_end = prediction_dict[cme]['times'][-1]
                if cme in matches.keys():
                    fill_value = 3
                else:
                    fill_value = 1

            elif cme in unmatched_gt:
                current_start = groundtruth_dict[cme]['times'][0]
                current_end = groundtruth_dict[cme]['times'][-1]

                fill_value = 2

            else:
                continue

            if (current_start.month == begin_month.month or current_end.month == end_month.month) and (current_start.year == begin_month.year or current_end.year == end_month.year):

                if current_start < begin_month:
                    current_start = begin_month
                if current_end > end_month:
                    current_end = end_month

                comp_temp = np.where((comp_time >= current_start) & (comp_time <= current_end), fill_value, 0)

                assigned = False

                for row_idx, active in enumerate(active_rows):
                    # Check for overlap
                    if all(current_start > end or current_end < start for (start, end) in active):
                        # No overlap -> safe to assign
                        plot_temp[row_idx] = np.where(plot_temp[row_idx] == 0, comp_temp, plot_temp[row_idx])
                        active.append((current_start, current_end))
                        assigned = True
                        break

                if not assigned:
                    # Optional: handle case where more than max_rows are required
                    print('Max rows exceeded')
                    pass


        binary_range.append(plot_temp)
    
    shape_arr = []
    for i in range(len(binary_range)):
        shape_arr.append(np.shape(binary_range[i])[1])
    
    max_len = max(shape_arr)

    for i in range(len(binary_range)):
        for j in range(len(binary_range[i])):
            if binary_range[i][j].shape[0] < max_len:
                ap_arr = np.zeros(max_len-binary_range[i][j].shape[0])
                ap_arr[:] = 4
                binary_range[i][j] = np.append(binary_range[i][j], ap_arr)

    return np.array(binary_range, dtype=float), np.array(scatter_extra)

if __name__ == "__main__":

    # Load the saved segmentation .npz predicitons
    ml_path_basic = '/home/mbauer/Code/CME_ML/Model_Train/'
    corrected_helcats_path = '/home/mbauer/Data/HCME_WP3_V06_TE_PROFILES_CORRECTED_CSV/'
    helcats_path = '/home/mbauer/Data/HCME_WP3_V06_TE_PROFILES/'
    fits_path = '/media/DATA_DRIVE/stereoa/secchi/L0/img/hi_1/'

    years = ['2009','2011']
    mdls = ['run_25062025_120013_model_cnn3d']
    method = 'median'
    threshold = 0.55

    for year in years:
        helcats_save_path = ml_path_basic + 'helcats_dictionary_'+year+'.npy'
        generate_new_txt = False

        if generate_new_txt:
            generate_helcats_gt_corrected(helcats_path,corrected_helcats_path,fits_path,year)
            get_helcats_tracks(corrected_helcats_path,helcats_save_path,year)

        for mdl in mdls:
            ml_path = ml_path_basic + mdl + '/'
            
            load_path = ml_path + 'segmentation_masks_'+method+'_'+year+'.npz'
            save_path = ml_path + 'segmentation_dictionary_'+method+'_'+year+'.npy'

            create_dictionaries_operational(load_path, save_path, t=threshold)
