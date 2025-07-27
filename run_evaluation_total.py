from datetime import datetime
from astropy.wcs import FITSFixedWarning
import warnings
warnings.simplefilter('ignore', category=FITSFixedWarning)
from utils import parse_yml
from evaluate_segmentation_science import main as evaluate_segmentation_science_main
from run_test_ml import main as run_test_ml_main
from run_test_operational import main as run_test_operational_main
from extract_fronts_ml import main as extract_fronts_ml_main
from extract_fronts_operational import main as extract_fronts_operational_main
from evaluate_tracking_ml import main as evaluate_tracking_ml_main
from evaluate_tracking_operational import main as evaluate_tracking_operational_main
import shutil
import os

if __name__ == "__main__":

    config = parse_yml('config_evaluation.yaml')

    mode = config['mode']
    methods = config['method']
    plotting = config['plotting']
    mdls_event_based = config['mdls_event_based']
    mdls_operational = config['mdls_operational']

    ml_path = config['paths']['ml_path']
    rdif_path = config['paths']['rdif_path']
    data_paths = config['paths']['data_paths']
    annotation_path = config['paths']['annotation_path']
    helcats_path = config['paths']['helcats_path']
    wp2_path = config['paths']['wp2_path']

    try:
        corrected_helcats_path = config['paths']['corrected_helcats_path']
    except KeyError:
        corrected_helcats_path = None
    
    fits_path = config['paths']['fits_path']

    get_segmentation_masks = config['get_segmentation_masks']

    time_pairs = config['time_pairs']
    timepairs = [{'start': time_pairs['start'][i], 'end': time_pairs['end'][i]} for i in range(len(time_pairs['start']))]
    
    dates_plotting_operational = config['dates_plotting_operational']
    years_plotting_operational = list(dates_plotting_operational.keys())
    months_plotting_operational = [dates_plotting_operational[year] for year in years_plotting_operational]

    date_str = None

    if date_str is not None:
            
        best_segmentation_path = ml_path + 'results_science/'+date_str+'/segmentation_results_science.txt'
    
    else:
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")

        best_segmentation_path = ml_path + 'results_science/'+date_str+'/segmentation_results_science.txt'

    if not os.path.exists(ml_path + 'results_science/'+date_str+ '/'):
        os.makedirs(ml_path + 'results_science/'+date_str+ '/')

    shutil.copy("config_evaluation.yaml", ml_path + "results_science/"+date_str+"/config_evaluation.yaml")

    if get_segmentation_masks:
        run_test_ml_main(mdls=mdls_event_based, ml_path=ml_path, mode=mode)

    evaluate_segmentation_science_main(mdls=mdls_event_based, ml_path=ml_path, date_str=date_str, mode=mode, methods=methods, plotting=plotting, rdif_path=rdif_path, img_size=config["img_size"])

    with open(best_segmentation_path, 'r') as f:
        lines = f.readlines()

    best_method = lines[-2].split('Best Method:')[-1].rstrip().strip(' ')
    best_threshold = float(lines[-1].split('Best Threshold:')[-1].rstrip().strip(' '))

    if get_segmentation_masks:
        run_test_operational_main(mdl=mdls_operational, ml_path=ml_path, timepairs=timepairs, data_paths=data_paths, best_method=best_method)

    extract_fronts_ml_main(mdls=mdls_event_based, ml_path=ml_path, annotation_path=annotation_path, best_threshold=best_threshold, best_method=best_method, mode=mode)
    extract_fronts_operational_main(mdl=mdls_operational, ml_path=ml_path, timepairs=timepairs, helcats_path=helcats_path, corrected_helcats_path=corrected_helcats_path, fits_path=fits_path, best_method=best_method, best_threshold=best_threshold)

    evaluate_tracking_ml_main(mdls=mdls_event_based, ml_path=ml_path, best_method=best_method, best_threshold=best_threshold, date_now=date_str, mode=mode, plotting=plotting, rdif_path=rdif_path, img_size=config["img_size"])

    evaluate_tracking_operational_main(mdl=mdls_operational, 
        ml_path=ml_path, 
        timepairs=timepairs, 
        best_method=best_method, 
        best_threshold=best_threshold, 
        date_now=date_str,
        wp2_path=wp2_path,
        plotting=plotting, 
        plot_area=False, 
        use_threshold=True, 
        years_plotting=years_plotting_operational, 
        months_plotting=months_plotting_operational,
        data_paths=data_paths,
        img_size=config['img_size']
        )
