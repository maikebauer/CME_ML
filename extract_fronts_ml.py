import numpy as np
import datetime
from skimage.morphology import disk
from skimage.measure import label
from astropy.wcs import FITSFixedWarning
import warnings
from utils_evaluation import load_results, post_processing, get_fitsfiles, get_outline, get_front, connect_cmes_new, remove_outliers_from_fronts,get_cmes,get_ml_gt, clean_cme_elongation_tracks
warnings.simplefilter('ignore', category=FITSFixedWarning)
from utils import parse_yml

def create_dictionaries(load_path, save_path, save_path_gt, annotation_path, t=0.45):
    print('Creating dictionaries...')
    filenames, pred = load_results(load_path)
    filenames = [name.decode('utf-8') for name in filenames]

    # Convert dates in input_names to datetime objects
    input_dates = [datetime.datetime.strptime(name[:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d_%H%M%S') for name in filenames]
    input_days = [datetime.datetime.strptime(name[:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d') for name in filenames]
    input_days_set = sorted(set(input_days))

    cme_dict_gt = get_ml_gt(annotation_path,input_dates)
    np.save(save_path_gt, cme_dict_gt)
    
    images_proc, labeled_clusters_connected = post_processing(pred, t=t)
    image_outlines, image_areas = get_cmes(labeled_clusters_connected, return_area=True)

    fits_headers = get_fitsfiles(input_days_set, input_dates)

    image_outlines_wcs, image_outline_pixels = get_outline(image_outlines, fits_headers)
    img_front_wcs, img_front_pixels, img_front_areas = get_front(image_outlines_wcs, image_outline_pixels, fits_headers, image_areas=image_areas)
    img_front_wcs_clean, img_front_pixels_clean, img_front_areas_clean = remove_outliers_from_fronts(img_front_wcs, img_front_pixels, fits_headers, img_front_areas, window=3, threshold=1)

    input_dates_obs = [datetime.datetime.strptime(datetime.datetime.strptime(fits_headers[i]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S'),'%Y%m%d_%H%M%S') for i in range(len(fits_headers))]
    input_dates_datetime = [datetime.datetime.strptime(name[:15], '%Y%m%d_%H%M%S') for name in input_dates]

    cme_dictionary = connect_cmes_new(images_proc, input_dates_datetime, img_front_wcs_clean, img_front_pixels_clean, input_dates_obs, img_front_areas_clean)
    # cme_dictionary_final = clean_cme_elongation_tracks(cme_dictionary)

    np.save(save_path, cme_dictionary)

def main(mdls, ml_path, annotation_path, best_threshold, best_method, mode='test'):

    for fold_num in range(0,len(mdls)):
        print(f'Processing model {mdls[fold_num]}...')

        ml_base_path = ml_path + mdls[fold_num] + '/'

        load_path = ml_base_path + 'segmentation_masks_'+best_method+'_'+mode+'.npz'
        save_path = ml_base_path + 'segmentation_dictionary_'+best_method+'_'+mode+'_'+str(best_threshold).split('.')[-1]+'.npy'

        save_path_gt = ml_base_path + 'segmentation_dictionary_gt_'+mode+'.npy'

        create_dictionaries(load_path, save_path, save_path_gt, annotation_path, t=best_threshold)

if __name__ == "__main__":

    config = parse_yml('config_evaluation.yaml')

    mode = config['mode']
    mdls_event_based = config['mdls_event_based']
    ml_path = config['paths']['ml_path']

    methods = config['method']

    annotation_path = config['paths']['annotation_path']

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
        
    main(mdls=mdls_event_based, ml_path=ml_path, annotation_path=annotation_path, best_threshold=best_threshold, best_method=best_method, mode=mode)
        
