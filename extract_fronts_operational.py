import numpy as np
import datetime
from astropy.wcs import FITSFixedWarning
import warnings
warnings.simplefilter('ignore', category=FITSFixedWarning)
from utils_evaluation import load_results, post_processing, get_fitsfiles, get_outline, get_front, connect_cmes_new, remove_outliers_from_fronts,get_cmes,generate_helcats_gt_corrected, get_helcats_tracks, clean_cme_elongation_tracks
from utils import parse_yml


def create_dictionaries(load_path, save_path, t=0.90):
    print('Creating dictionaries...')
    filenames, pred = load_results(load_path)
    filenames = [name.decode('utf-8') for name in filenames]

    # Convert dates in input_names to datetime objects
    # input_dates = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d_%H%M%S') for name in filenames if name.split('/')[-1].startswith(year)]
    # input_days = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d') for name in filenames if name.split('/')[-1].startswith(year)]

    input_dates = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d_%H%M%S') for name in filenames]
    input_days = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d') for name in filenames]

    input_days_set = sorted(set(input_days))
    pred = pred[:len(input_dates)]

    images_proc, labeled_clusters_connected = post_processing(pred, t=t)
    image_outlines, image_areas = get_cmes(labeled_clusters_connected, return_area=True)

    fits_headers = get_fitsfiles(input_days_set, input_dates)

    input_dates_obs = [datetime.datetime.strptime(fits_headers[i]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') for i in range(len(fits_headers))]

    image_outlines_wcs, image_outline_pixels = get_outline(image_outlines, fits_headers)
    img_front_wcs, img_front_pixels, img_front_areas = get_front(image_outlines_wcs, image_outline_pixels, fits_headers, image_areas=image_areas)
    img_front_wcs_clean, img_front_pixels_clean, img_front_areas_clean = remove_outliers_from_fronts(img_front_wcs, img_front_pixels, fits_headers, img_front_areas, window=3, threshold=1)

    input_dates_obs = [datetime.datetime.strptime(datetime.datetime.strptime(fits_headers[i]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S'),'%Y%m%d_%H%M%S') for i in range(len(fits_headers))]
    input_dates_datetime = [datetime.datetime.strptime(name[:15], '%Y%m%d_%H%M%S') for name in input_dates]
    
    cme_dictionary = connect_cmes_new(images_proc, input_dates_datetime, img_front_wcs_clean, img_front_pixels_clean, input_dates_obs, img_front_areas_clean)
    #cme_dictionary = connect_cmes_new(images_proc, input_dates_datetime, img_front_wcs_clean, img_front_pixels_clean, input_dates_obs)
    # cme_dictionary_final = clean_cme_elongation_tracks(cme_dictionary)
    
    np.save(save_path, cme_dictionary)

def main(mdl, ml_path, timepairs, helcats_path, corrected_helcats_path, fits_path, best_method, best_threshold):

    for pair_idx, pair in enumerate(timepairs):
        start = pair['start']
        end = pair['end']

        helcats_save_path = ml_path + 'helcats_dictionary_'+start+'_'+end+'.npy'
        generate_new_txt = False

        if generate_new_txt or corrected_helcats_path is None:
            
            if corrected_helcats_path is None:
                corrected_helcats_path = helcats_path[:-1] + '_CORRECTED_CSV/'

            generate_helcats_gt_corrected(helcats_path,corrected_helcats_path,fits_path,pair)
            get_helcats_tracks(corrected_helcats_path,helcats_save_path,pair)

        mdl_path = ml_path + mdl + '/'
        
        load_path = mdl_path + 'segmentation_masks_'+best_method+'_'+start+'_'+end+'.npz'
        save_path = mdl_path + 'segmentation_dictionary_'+best_method+'_'+str(best_threshold).split('.')[-1]+'_'+start+'_'+end+'.npy'

        create_dictionaries(load_path, save_path, t=best_threshold)

if __name__ == "__main__":

    config = parse_yml('config_evaluation.yaml')

    mdls_operational = config['mdls_operational']
    
    ml_path = config['paths']['ml_path']
    helcats_path = config['paths']['helcats_path']

    try:
        corrected_helcats_path = config['paths']['corrected_helcats_path']
    except KeyError:
        corrected_helcats_path = None
    
    fits_path = config['paths']['fits_path']

    methods = config['method']

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

    main(mdl=mdls_operational, ml_path=ml_path, timepairs=timepairs, helcats_path=helcats_path, corrected_helcats_path=corrected_helcats_path, fits_path=fits_path, best_method=best_method, best_threshold=best_threshold)

