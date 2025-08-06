# Solar Transient Recognition Using Deep Learning (STRUDL)

Code for the various components of the STRUDL model. A list of Python packages used to generate the results in the STRUDL paper can be found in `requirements.txt`.

The main file is `model_3dconv_onec_slide.py`, which is used to run the model for training.
All model configurations are defined in the `sample_config.yaml` file. It is recommended to clone this file and rename the copy to `config.yaml`, as `sample_config.yaml` only serves as an example.
An explanation of each keyword is given in [Section: Configuration](#configuration).


## Running STRUDL

After installing the required packages, you can run STRUDL from the `CME_ML` directory using the following command:

```
python model_3dconv_onec_slide.py
```

This command will use all the settings defined in the config.yaml file and run the base model as described in the [STRUDL paper](https://arxiv.org/abs/2506.16194). The results will be saved in a new directory within `CME_ML/Model_Train/`,
named `run_ddmmyyyy_HHMMSS_model_3donv_onec_slide/`, where `ddmmyyyy_HHMMSS` corresponds to the start time of the model run. The trained model will be saved as a `.pth` file, and the `config.yaml` file used for the run will be copied into the results folder for reference.

## Evaluating STRUDL

All evaluation settings are defined in the `sample_config_evaluation.yaml` file. As with the training config, it is recommended to copy and rename it to `config_evaluation.yaml`.
All configuration options are explained in [Section: Evaluation Configuration](#evaluation-configuration). 
To reproduce the evaluation results shown in the [STRUDL paper](https://arxiv.org/abs/2506.16194), run:

```
python run_evaluation_total.py
```

## Configuration

**dataset:**

  data_path: "/path/to/data/" _(Path to folder in which .pkl files of ST-A HI-1 running difference images in full 1024x1024 resolution are saved)_

  annotation_path: "/path/to/annotation.json" _(Path to .json file containing the annotations for the ST-A HI-1 images)_

  height: 128 _(desired height for ST-A HI-1 images used for training/validation/testing)_

  width: 128 _(desired width for ST-A HI-1 images used for training/validation/testing)_

  win_size: 16 _(desired length of image series)_

  stride: 2 _(desired stride when splitting image seriws into sequences)_

  quick_run: false _(uses only the first few image sequences as input to check if code is running without errors)_


**model:**

  model_path: "/path/to/code/" _(path to where this repository is saved)_

  name: "cnn3d" _(name of backbone to be used, it is recommended to stick to cnn3d)_

  model_parameters:
  - input_channels: 1
  - output_channels: 1
  - final_layer: Sigmoid _(final layer of model)_

  final_layer: Sigmoid

  device: cuda _(which device to run the model on)_

  seed: 42 _(seed to use for sequence generation if pre-determined sequences are not used)_


**optimizer:**

  name: Adam _(name of optimizer to use)_

  optimizer_parameters:
  - lr: 1.0e-05
  - weight_decay: 0.0


**scheduler:**

  use_scheduler: false

  name: ReduceLROnPlateau _(if use_scheduler is true, which scheduler to use)_

  scheduler_parameters:
  - mode: min
  - factor: 0.1
  - patience: 10


**loss:**

  name: AsymmetricUnifiedFocalLoss _(name of loss function to use)_

  loss_parameters:
  - weight: 0.5
  - delta: 0.6
  - gamma: 0.5


**train:**

  batch_size: 4

  num_workers: 2

  epochs: 150

  include_potential: true _(include CMEs flagged as potential CMEs)_

  include_potential_gt: true _(include CMEs flagged as potential CMEs)_

  data_parallel: true _(train the model on parallel GPUs)_

  shuffle: true _(shuffle the training dataset each epoch)_

  binary_gt: true _(read in ground truth as binary array)_

  threshold_iou: 0.9 _(IoU threshold to use for computing training IoU after each epoch)_

  cross_validation:

  - use_cross_validation: true
  - fold_file: "/path/to/folds_dictionary.npy" _(path of .npy file containing definition of folds to use for cross-validation)_

  - fold_definition: _(which folds to use for what set)_

    - train:

      - fold_1

      - fold_2

      - fold_3

      - fold_4

      - fold_5

      - fold_6

      - fold_7

      - fold_8

    - test:

      - fold_9

    - val:

      - fold_10

        
  data_augmentation: _(which data augmentations to use and which probability to apply them with)_

  - name: ToTensor

  - name: RandomHorizontalFlip

    - p: 0.5

  - name: RandomVerticalFlip

    - p: 0.5

  load_checkpoint:

  - load_model: False _(whether to re-start training from a previous checkpoint)_

  - load_optimizer: False

  - checkpoint_path: "/path/to/checkpoint.pth"

  - updated_lr: 1.0e-05


**evaluate:** _(parameters useed only in evaluation step)_

  include_potential: true

  include_potential_gt: true

  shuffle: false

  binary_gt: true


**test:** _(currently not used, parameters used only in testing step)_

  model_path: "/path/to/model.pth"

  include_potential: true

  include_potential_gt: true

  shuffle: false

  binary_gt: true

  threshold_iou: 0.9

## Evaluation Configuration

mode: test (choose either 'val' for validation, or 'test' for testing)

**method:** _(which methods to use for aggregating the predicted segmentation masks)_
- median
- mean
- max

**mdls_event_based:** _(names of folders containing .pth files to evaluate as part of event-based tracking)_
- run_27062025_150321_model_cnn3d
- run_27062025_150708_model_cnn3d
- run_27062025_171628_model_cnn3d
- run_30062025_105242_model_cnn3d
- run_27062025_154802_model_cnn3d

mdls_operational: run_25062025_120013_model_cnn3d _(name of folder containing .pth file to evaluate for continuous tracking)_

**paths:**
- ml_path: /path/to/Model_Train/ _(path to the Model_Train folder)_

- data_paths: _(path to running difference images for continuous tracking)_
  - /path/to/2009_rdifs/
  - /path/to/2011_rdifs/

- rdif_path: /path/to/event_based_rdifs/ _(path to running difference images for event-based tracking)_
- annotation_path: /path/to/instances_default.json _(path to .json file containing annotations)_
- helcats_path: /path/to/helcats/HCME_WP3_V06_TE_PROFILES/ _(path to HELCATS WP3, aka HIGeoCAT, profiles. Can be downloaded [here](https://www.helcats-fp7.eu/catalogues/wp3_cat.html))_
- corrected_helcats_path: /path/to/time_corrected/helcats/HCME_WP3_V06_TE_PROFILES_CORRECTED_CSV/ _(path to corrected versions of HELCATS WP3 profiles. Delete this keyword for program to create corrected files)_
- fits_path: /path/to/STEREO_A/fits/ _(path to ST-A HI-1 .fits files, with or without data reduction applied)_
- wp2_path: /path/to/helcats/HCME_WP2_V06.json _(path to HELCATS WP2, aka HICAT, profiles. Can be downloaded [here](https://www.helcats-fp7.eu/catalogues/wp2_cat.html))_

**time_pairs:** _(start and end times for continuous evaluation)_
- start:
  - '2009_01_01'
  - '2011_01_01'

- end:
  - '2009_12_31'
  - '2011_12_31'

get_segmentation_masks: True _(get segmentation masks from .pth file. Set to True if evaluation for model is done for the first time)_

plotting: True

**dates_plotting_operational:** _(years and months to be plotted for operational evaluation)_
- '2009':
  - '01'
  - '02'

- '2011':
  - '01'
  - '02'
