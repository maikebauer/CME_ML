import json
import numpy as np
import cv2
from matplotlib import patches as patches
import pandas as pd
from skimage.morphology import disk, binary_closing, remove_small_objects, binary_dilation
import skimage.filters as filters
import datetime as datetime
import glob
from astropy.io import fits
import os
from skimage.measure import label, regionprops
import time
import pickle
import json
import os
import cv2
import numpy as np

def binary_mask_to_rle_np(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle

################ DEFINE PATHS HERE ################
my_year = 2012
my_month_min = 1
my_month_max = 6
my_day_min = 1

path_rdifs = '/home/mbauer/Data/stereo_processed/running_difference/data/A/'
path = '/home/mbauer/Data/SSW_Data/'
png_path = '/home/mbauer/Data/differences/rundif_jan_jun_2012/'
path_filenames = png_path.split('/')[-2] + '/'
###################################################

with open(path+'ssw_classification.pkl', 'rb') as f:
    date_dict = pickle.load(f)

df = pd.DataFrame(date_dict).T.reset_index()
df.columns = ['date_obs', 'classification_id', 'subject_id', 'sc', 'user_name', 'masks']

df['date_obs'] = pd.to_datetime(df['date_obs'], format='%Y%m%d_%H%M%S')
df['date_short'] = df['date_obs'].dt.strftime('%Y%m%d')
df = df.sort_values(by='date_obs')

custom_df = df[(df['date_obs'].dt.year == my_year) & (df['date_obs'].dt.month <= my_month_max) & (df['date_obs'].dt.month >= my_month_min)].reset_index(drop=True)

# Initialize COCO format dictionary
coco_data = {
    "licenses": [
        {
            "name": "",
            "id": 0,
            "url": ""
        }
    ],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    },
    "categories": [
        {"id": 1, "name": "CME", "supercategory": ""}
    ],
    "images": [],
    "annotations": []
}


# Directory containing the binary masks and images

t = 0.25

ssw_date_prev = None

file_days = pd.date_range(start='2012-06-13', end='2012-06-30').strftime('%Y%m%d').values

# Calculate the start time

start = time.time()

image_id = 1
annotation_id = 1

for fdate in file_days:
    print('Processing date: ', fdate)
    df_temp = custom_df[custom_df['date_short'] == fdate]

    path_ssw_rdifs = path_rdifs + fdate + '/science/hi_1/'
    fits_files = sorted(glob.glob(path_ssw_rdifs + '*.fts'))

    headers = []
    rdif_date_obs = []
    data = []

    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            headers.append(hdul[0].header)
            rdif_date_obs.append(datetime.datetime.strptime(hdul[0].header['DATE-OBS'][:-4], '%Y-%m-%dT%H:%M:%S'))
            data.append(hdul[0].data)
            hdul.close()

    rdif_date_obs = np.array(rdif_date_obs)

    for num, fil in enumerate(fits_files):
        width, height = data[num].shape
        file_name = path_filenames + fil.split('/')[-1].split('.')[0] + '.png'

        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }

        coco_data["images"].append(image_info)

        if len(df_temp) == 0:
            continue

        else:
            df_match = df_temp[df_temp['date_obs'] == rdif_date_obs[num]]

            if len(df_match) == 0:
                continue
            
            else:
                msk = []

                for i in range(len(df_match['masks'].values[0])):
                    polygon = df_match['masks'].values[0][i]
                    pol_mask = np.zeros((1024, 1024), dtype=np.uint8)

                    if len(polygon) != 0:
                        pol_mask = cv2.polylines(pol_mask, polygon, isClosed=False, color=(255), thickness=25)

                    if np.sum(pol_mask) != 0:
                        msk.append(pol_mask)

                msk = np.array(msk)

                msk_sum = np.flipud(np.sum(msk, axis=0))

                msk_gaussian = filters.gaussian(msk_sum, sigma=8)

                msk_sato = filters.sato(msk_gaussian, sigmas=[1,10], black_ridges=False)

                msk_thresh = msk_sato/msk_sato.max() > t
                msk_thresh = remove_small_objects(msk_thresh, 5096)
                msk_thresh = binary_dilation(binary_closing(msk_thresh, disk(20)), disk(4))

                dil_temp = binary_dilation(msk_thresh, disk(14))
                dil_temp = binary_closing(dil_temp, disk(26))
                label_im = label(dil_temp)
                
                regions = regionprops(label_im)

                for region in regions:
                    binary_mask  = np.zeros((height, width), dtype=np.uint8)

                    for coords in region.coords:
                        binary_mask[coords[0], coords[1]] = 1

                    binary_mask = np.where(msk_thresh == 0, 0, binary_mask)

                    #RLE encoding
                    rle = binary_mask_to_rle_np(binary_mask)

                    # Bounding box
                    bbox = [region.bbox[1], region.bbox[0], region.bbox[3] - region.bbox[1], region.bbox[2] - region.bbox[0]]
                    # Area of the region
                    area = np.sum(binary_mask)

                    x, y, w, h = bbox
                    
                    # Area of the contour
                    #area = region.area_bbox
                    
                    # Create annotation entry with RLE and attributes
                    annotation = {
                        "id": int(annotation_id),
                        "image_id": int(image_id),
                        "category_id": 1,  # CME category id
                        "segmentation": rle,  # Use RLE instead of polygon
                        "area": int(area),
                        "bbox": {0: x, 1: y, 2: w, 3: h},
                        "iscrowd": 1,
                        "attributes": {
                            "id": "0",  # Default id
                            "potential": False,  # Default potential
                            "occluded": False  # Default occluded
                        }
                    }
                    
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

    end = time.time()
    length = end - start

    if (int(length/60) % 15 == 0):# & (int(length/60) > 0):
        filename = 'cme_coco_annotations_rle.json'

        try:
            os.remove(filename)
        except OSError:
            pass

        with open(filename, 'w') as f:
            json.dump(coco_data, f)

    image_id = image_id + 1
    
filename = 'cme_coco_annotations_rle.json'

try:
    os.remove(filename)
except OSError:
    pass

with open(filename, 'w') as f:
    json.dump(coco_data, f)