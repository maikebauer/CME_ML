import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import coco
import matplotlib.pyplot as plt 
import numpy as np
from torchvision.transforms import v2
import sys
import os
from skimage.morphology import binary_dilation, disk
from numpy.random import default_rng
import cv2
from scipy.ndimage.measurements import label
from scipy import ndimage
from utils import sep_noevent_data, check_diff
import random
from skimage import transform
import datetime
import glob
from collections import defaultdict
from bisect import bisect_left

class RundifSequenceNew(Dataset):
    def __init__(self, data_path, annotation_path, im_transform=None, mode='train', win_size=16, stride=2, width_par=128, cadence_minutes=40, include_potential=True, split_ratios=(0.7, 0.2, 0.1), use_cross_validation=True, fold_definition=None, quick_run=False, seed=42):
        
        self.im_transform = im_transform
        self.mode = mode
        self.width_par = width_par
        self.win_size = win_size
        self.stride = stride
        self.cadence_minutes = cadence_minutes
        self.include_potential = include_potential
        self.quick_run = quick_run
        self.seed = seed

        self.annotation_path = annotation_path
        self.data_path = data_path
        
        self.fold_definition = fold_definition
        self.k_folds = len(fold_definition['train']) + len(fold_definition['val']) + len(fold_definition['test']) if fold_definition else None
        self.split_ratios = split_ratios
        self.use_cross_validation = use_cross_validation

        if self.fold_definition is None and self.use_cross_validation:
            raise ValueError("fold_definition must be defined in config.yaml when use_cross_validation is True.")

        self._rng = default_rng(self.seed)
        self._py_rng = random.Random(self.seed)

        try:
            self.coco_obj = coco.COCO(self.annotation_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file not found at: {self.annotation_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load COCO annotations: {e}")

        self.files = sorted(glob.glob(self.data_path + '*.npy'))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in the directory: {self.data_path}")
        
        self.file_times = np.array([datetime.datetime.strptime(fle.split('/')[-1][:15], '%Y%m%d_%H%M%S') for fle in self.files])

        self.img_ids = np.array(self.coco_obj.getImgIds())

        self.sequences = self._generate_split_indices_from_coco()

        self.img_ids_win = self.sequences[self.mode]

        win_delete = []
        for win_num, win in enumerate(self.img_ids_win):
            if np.all(win == None):
                win_delete.append(win_num)
        
        self.img_ids_win = np.delete(self.img_ids_win, win_delete, axis=0)

        if self.quick_run:
            self.img_ids_win = self.img_ids_win[:10]

        self.img_ids_train_win = self.sequences['train'].copy()
        self.img_ids_val_win = self.sequences['val'].copy()
        self.img_ids_test_win = self.sequences['test'].copy()

    def _get_regular_time_grid(self):
        """
        Builds a regular time grid from the first to the last image timestamp
        with fixed cadence. Returns list of datetime objects.

        Returns:
            time_grid: a regular time grid with cadence=minute_cadence, going from first to last image timestamp.
        """

        # timestamps = []

        # for img in self.coco_obj.imgs.values():
        #     try:
        #         timestamps.append(datetime.datetime.strptime(img["file_name"].split('/')[-1][:15], "%Y%m%d_%H%M%S"))
        #     except ValueError as e:
        #         raise ValueError(f"Timestamp parsing failed for file: {img['file_name']}, error: {e}")

        timestamps = self.file_times.copy()

        start_time = min(timestamps)
        end_time = max(timestamps)

        time_grid = []
        current_time = start_time
        delta = datetime.timedelta(minutes=self.cadence_minutes)

        while current_time <= end_time:
            time_grid.append(current_time)
            current_time += delta

        return time_grid


    def _map_time_to_images_and_cmes(self, time_grid, tolerance_seconds=600):
        """
        Builds a mapping from each time in the regular time grid to the closest matching image_id/set of cme_ids
        in the dataset, within a specified time tolerance (in seconds). If no match is found within
        tolerance, the value will be None/an empty set.

        Args:
            time_grid: list of datetime objects representing a regular timegrid with cadence=minute_cadence.
            tolerance_seconds: max allowed time difference between actual image time and grid time.

        Returns:
            time_to_img_id: dict mapping time (from grid) -> image_id (or None if no close match).
            time_to_cme_id: dict mapping time (from grid) -> cme_ids (or empty set if no close match).
        """
        # Extract actual times and sort
        f_times = self.file_times.copy()
        f_times.sort()

        actual_time_to_img_id = {}
        file_time_to_img_id = {}
            
        for img in self.coco_obj.imgs.values():
            t = datetime.datetime.strptime(img['file_name'].split('/')[-1][:13], '%Y%m%d_%H%M')
            actual_time_to_img_id[t] = img['id']

        for file_time in self.file_times:
            short_time = datetime.datetime.strptime(file_time.strftime('%Y%m%d_%H%M'), '%Y%m%d_%H%M')
            file_time_to_img_id[file_time] = actual_time_to_img_id.get(short_time, None)


        # Map image_id to cme_ids
        img_id_to_cme_id = defaultdict(set)
        for ann in self.coco_obj.anns.values():
            image_id = ann["image_id"]
            cme_id = ann["attributes"]["id"]
            is_potential = ann["attributes"]["potential"]

            if is_potential and not self.include_potential:
                continue
            else:
                img_id_to_cme_id[image_id].add(cme_id)

        # Match each grid time to closest actual time
        time_to_img_id = {}
        time_to_cme_id = {}

        for t in time_grid:
            idx = bisect_left(f_times, t)

            # Find the closest neighbor (either before or after)
            candidates = []
            if idx > 0:
                candidates.append(f_times[idx - 1])
            if idx < len(f_times):
                candidates.append(f_times[idx])

            best_match = None
            min_diff = datetime.timedelta(seconds=tolerance_seconds + 1)

            for c in candidates:
                diff = abs((t - c).total_seconds())
                if diff <= tolerance_seconds and diff < min_diff.total_seconds():
                    best_match = c
                    min_diff = datetime.timedelta(seconds=diff)

            time_to_img_id[t] = file_time_to_img_id[best_match] if best_match else None
            time_to_cme_id[t] = img_id_to_cme_id[file_time_to_img_id[best_match]] if best_match else None

            if time_to_cme_id[t] is None:
                short_time = datetime.datetime.strptime(t.strftime('%Y%m%d_%H%M'), '%Y%m%d_%H%M')
                time_to_cme_id[t] = img_id_to_cme_id[actual_time_to_img_id.get(short_time, None)]


        return time_to_img_id, time_to_cme_id

    def _get_padded_cme_blocks(self, time_to_cme_id, time_grid, pad_steps=20):
        """
        Args:
            time_to_cme_id: dict mapping time to cme_ids (or empty set if no close match).
            time_grid: list of datetime objects representing a regular timegrid with cadence=minute_cadence.
            pad_steps: number of steps to pad before and after each CME block.
        Returns:
            padded_blocks: list of lists, where each inner list contains the padded time steps of a CME block.

        Identifies CME blocks (consecutive time steps with CME activity),
        and pads them with empty steps before and after (if no overlap with other CME blocks).
        """
        blocks = []
        current_block = []
        
        for t in time_grid:
            if time_to_cme_id[t] != set():
                current_block.append(t)
            else:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                    
        if current_block:
            blocks.append(current_block)

        # Pad each block without overlapping with neighbors
        padded_blocks = []
        used_indices = set()
        last_discarded_range = None

        for block_num, block in enumerate(blocks):
            previous_block_end = time_grid.index(padded_blocks[-1][-1]) if block_num > 0 else None
            
            next_block_start = time_grid.index(blocks[block_num + 1][0]) if block_num < len(blocks) - 1 else None
            start_idx = time_grid.index(block[0])
            end_idx = time_grid.index(block[-1])

            pad_steps_before_max =  ((start_idx-previous_block_end)//2)-1 if previous_block_end is not None else pad_steps
            pad_steps_after_max = ((next_block_start-end_idx)//2)-1 if next_block_start is not None else pad_steps

            pad_steps_before = min(pad_steps, pad_steps_before_max)
            pad_steps_after = min(pad_steps, pad_steps_after_max)

            pad_start = max(0, start_idx - pad_steps_before)
            pad_end = min(len(time_grid) - 1, end_idx + pad_steps_after)

            if last_discarded_range is not None:
                discarded_start, discarded_end = last_discarded_range
                if (block[0] - time_grid[discarded_end]).total_seconds() < self.cadence_minutes*60*3.5:
                    # print(f"Extending block {block_num} into discarded block ending at {discarded_end}")
                    pad_start =  discarded_start
                    
                last_discarded_range = None

            padded_range = list(range(pad_start, pad_end + 1))

            if len(padded_range) < self.win_size:
                # print(f"Warning: CME block {block_num} is too short after padding - length is {len(padded_range)} steps.")
                last_discarded_range = (pad_start, pad_end)
                continue
            
            if any(i in used_indices for i in padded_range):
                # print(f"Skipping overlapping block: {block}")
                # Maybe do something else here instead of just skipping if there's overlap?
                # But there should never be overlap
                continue
            
            used_indices.update(padded_range)

            padded_blocks.append([time_grid[i] for i in padded_range])

        return padded_blocks


    def _assign_blocks_to_splits(self, blocks, time_to_img_id):
        """
        Randomly assigns blocks to train, val, and test splits. Converts from timesteps to image_ids.

        Args:
            blocks: list of CME blocks (each block is list of time steps).
            time_to_img_id: dict mapping time to image_id.

        Returns:
            split_index_blocks: dict with keywords 'train', 'val', 'test', each with a list of image_ids.
        """


        total_blocks = len(blocks)
        indices = list(range(total_blocks))
        self._py_rng.shuffle(indices)

        train_end = int(self.split_ratios[0] * total_blocks)
        val_end = train_end + int(self.split_ratios[1] * total_blocks)

        split_blocks = {"train": [], "val": [], "test": []}
        for i in range(total_blocks):
            block = blocks[indices[i]]
            if i < train_end:
                split_blocks["train"].append(block)
            elif i < val_end:
                split_blocks["val"].append(block)
            else:
                split_blocks["test"].append(block)

        split_index_blocks = {}

        for split, blocks in split_blocks.items():
            split_index_blocks[split] = [[time_to_img_id[time_id] for time_id in block] for block in blocks]

        return split_index_blocks

    def _assign_blocks_to_folds(self, blocks, time_to_img_id):
        """
        Randomly assigns blocks to k-folds for cross-validation. Converts from timesteps to image_ids.

        Args:
            blocks: list of CME blocks (each block is list of time steps).
            time_to_img_id: dict mapping time to image_id.

        Returns:
            split_dict: dict with keywords 'train', 'val', 'test', each with a list of image_ids.
        """

        indices = list(range(len(blocks)))
        self._py_rng.shuffle(indices)

        fold_blocks = [[] for _ in range(self.k_folds)]
        for i, idx in enumerate(indices):
            fold_blocks[i % self.k_folds].append(blocks[idx])

        split_blocks = {"train": [], "val": [], "test": []}

        for split in split_blocks:
            for fold_idx in self.fold_definition.get(split, []):
                split_blocks[split].extend(fold_blocks[int(fold_idx.split('_')[-1])-1])

        split_dict = {split: [[time_to_img_id[t] for t in block] for block in blocks] for split, blocks in split_blocks.items()}
        # split_dict_time = {split: [[datetime.datetime.strftime(t, '%Y%m%d_%H%M%S') for t in block] for block in blocks] for split, blocks in split_blocks.items()}
        # np.save("dictionary_enhanced.npy", split_dict_time)
        
        return split_dict

    def _generate_sliding_windows(self, split_indices):
        """
        Generates sliding window sequences of image IDs (of length win_size, with stride) from the time grid.

        Args:
            split_indices: dict with "train", "val", "test" -> list of image_ids

        Returns:
            sequences: dict of split -> list of image_id sequences (arrays)
        """
        sequences = {}

        for split, list_indices in split_indices.items():
            split_sequences = []
            
            for indices in list_indices:
                # Skip if not enough data
                if len(indices) < self.win_size:
                    continue

                index_windows = np.lib.stride_tricks.sliding_window_view(indices, self.win_size)[::self.stride]
                split_sequences.append(index_windows)

            sequences[split] = [item for inner_list in split_sequences for item in inner_list]

        return sequences
    
    def _generate_split_indices_from_coco(self):
        """
        Function to generate train/val/test sequences containing image ids from COCO object.
        """

        time_grid = self._get_regular_time_grid()

        time_to_img_id, time_to_cme_id = self._map_time_to_images_and_cmes(time_grid=time_grid)
        padded_blocks = self._get_padded_cme_blocks(time_to_cme_id, time_grid, pad_steps=10)

        if self.use_cross_validation:
            split_indices = self._assign_blocks_to_folds(padded_blocks, time_to_img_id)
        else:
            split_indices = self._assign_blocks_to_splits(padded_blocks, time_to_img_id)

        sequences = self._generate_sliding_windows(split_indices)

        return sequences#, time_to_img_id, time_to_cme_id, time_grid, split_indices, padded_blocks


    def __getitem__(self, index):
       
        # seed_index = int(index)
        seed_index = int(datetime.datetime.now().timestamp() * 1000)

        GT_all = []
        im_all = []
        
        item_ids = self.img_ids_win[index]
        file_names = []

        for idx in item_ids:
            if idx == None:
                # print(f"Warning: Data gap found in img_ids_win {item_ids}. Inserting empty image.")
                GT_all.append(v2.ToTensor()(np.zeros((self.width_par,self.width_par), dtype=bool)))
                im_all.append(v2.ToTensor()(np.zeros((self.width_par,self.width_par), dtype=np.float32)))
                file_names.append("")

            else:

                img_info = self.coco_obj.loadImgs([idx])[0]
                img_file_name = glob.glob(self.data_path+img_info["file_name"].split('/')[-1].split('.')[0][:16] + '*.npy')

                if len(img_file_name) == 0:
                    print("Error: No file found for image "+ self.data_path+img_info["file_name"].split('/')[-1].split('.')[0][:16])
                    sys.exit()
                if len(img_file_name) > 1:
                    print(f"Warning: Multiple files found for image ID {idx}. Using the first one: {img_file_name[0]}")
                    img_file_name = img_file_name[0].split('/')[-1]
                else:
                    img_file_name = img_file_name[0].split('/')[-1]

                file_names.append(img_file_name)

                height_par = self.width_par

                # Use URL to load image.

                im = np.load(self.data_path+img_file_name)

                if self.width_par != im.shape[0]:
                    im = transform.resize(im, (self.width_par , height_par), anti_aliasing=True, preserve_range=True)

                
                GT = []
                annotations = self.coco_obj.getAnnIds(imgIds=idx)

                if (len(annotations)>0):
                    for a in annotations:
                        
                        ann = self.coco_obj.loadAnns(a)
                        attr_potential = ann[0]['attributes']['potential']

                        if attr_potential:
                            if self.include_potential == True:
                                GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
                            else:
                                GT.append(np.zeros((self.width_par,height_par), dtype=bool))

                        else:
                            GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
                            
                else:
                    GT.append(np.zeros((self.width_par,height_par), dtype=bool))
                
                GT = np.array(GT)
                GT = np.nansum(GT, axis=0)

                if self.width_par != np.shape(GT)[0]:
                    GT = transform.resize(GT, (self.width_par , height_par), order=0)

                dilation = True

                if dilation:
                    kernel = disk(2)
                    n_it = int(self.width_par/64)
                    
                    GT = ndimage.binary_dilation(GT, structure=kernel, iterations=n_it)

                normalize = False

                if normalize:
                    vmin = np.nanmedian(im) - 2.5*np.nanstd(im)
                    vmax = np.nanmedian(im) + 2.5*np.nanstd(im)

                    im[im < vmin] = vmin
                    im[im > vmax] = vmax

                    im = (im - np.nanmin(im))/(np.nanmax(im) - np.nanmin(im))
                    im = np.clip(im, 0, 1)

                torch.manual_seed(seed_index)
                im = self.im_transform(im)

                torch.manual_seed(seed_index)
                GT = self.im_transform(GT) 
                
                GT_all.append(GT)
                im_all.append(im)

        GT_all = np.array(GT_all)
        im_all = np.array(im_all)

        if GT_all.ndim != 4:
            GT_all = GT_all[:, np.newaxis, :, :]
        
        if im_all.ndim != 4:
            im_all = im_all[:, np.newaxis, :, :]

        return {'image':torch.tensor(im_all), 'gt':torch.tensor(GT_all), 'names':file_names}
    
    def __len__(self):
        return len(self.img_ids_win)
    
class RundifSequence_Test(Dataset):
    def __init__(self, data_path,  pair, win_size=16, stride=2, width_par=128):

        start = datetime.datetime.strptime(pair['start'], '%Y_%m_%d')
        end = datetime.datetime.strptime(pair['end'], '%Y_%m_%d')

        self.width_par = width_par
        self.data_path = data_path
        self.win_size = win_size
        self.stride = stride

        # Load filenames whose names are between start and end date
        all_files = np.array(sorted(glob.glob(self.data_path + '*.npy')))
        all_files = np.array([file for file in all_files if datetime.datetime.strptime(file.split('/')[-1][:15], '%Y%m%d_%H%M%S') >= start and datetime.datetime.strptime(file.split('/')[-1][:15], '%Y%m%d_%H%M%S') <= end])
    
        self.files_strided = np.lib.stride_tricks.sliding_window_view(all_files,self.win_size)[::self.stride, :]
        
    def __getitem__(self, index):
       
        seed = int(index)

        im_all = []
        
        file_names = list(self.files_strided[index])

        height_par = self.width_par
        im_all = []

        for file in file_names:
            im = np.load(file)

            if self.width_par != 1024:
                im = transform.resize(im, (self.width_par , height_par), anti_aliasing=True, preserve_range=True)

            im_all.append(im)

        im_all = np.array(im_all)
        return {'image':torch.tensor(im_all).unsqueeze(1), 'names':file_names}
    
    def __len__(self):
        return len(self.files_strided)
    
class RundifSequence_SSW(Dataset):
    def __init__(self, data_path, annotation_path, width_par=128, include_potential=True, include_potential_gt=False, quick_run=False):
        
        rng = default_rng()

        self.width_par = width_par
        self.include_potential = include_potential
        self.include_potential_gt = include_potential_gt
        self.quick_run = quick_run
        self.annotation_path = annotation_path
        self.data_path = data_path

        self.coco_obj = coco.COCO(self.annotation_path)
        
        self.img_ids = self.coco_obj.getImgIds()

        time_list = []

        for a in range(0, len(self.img_ids)):
            time_list.append(datetime.datetime.strptime(self.coco_obj.loadImgs([self.img_ids[a]])[0]['file_name'].split('/')[-1][:15], '%Y%m%d_%H%M%S'))

        time_list = np.array(time_list)

        my_ind = np.where((time_list > datetime.datetime.strptime('2012-05-10 00:00:00', '%Y-%m-%d %H:%M:%S')) & (time_list < datetime.datetime.strptime('2012-05-12 00:00:00', '%Y-%m-%d %H:%M:%S')))[0]+1

        self.img_ids_win = my_ind
        
    def __getitem__(self, index):
       
        seed = int(index)

        GT_all = []
        im_all = []
        
        idx = self.img_ids_win[index]
        file_names = []
        
        img_info = self.coco_obj.loadImgs([idx])[0]
        img_file_name = img_info["file_name"].split('/')[-1].split('.')[0] + '.npy'

        file_names.append(img_file_name)

        height_par = self.width_par

        # Use URL to load image.

        im = np.load(self.data_path+img_file_name)

        if self.width_par != 1024:
            im = transform.resize(im, (self.width_par , height_par), anti_aliasing=True, preserve_range=True)

        GT = []
        annotations = self.coco_obj.getAnnIds(imgIds=idx)

        if (len(annotations)>0):
            for a in annotations:
                
                ann = self.coco_obj.loadAnns(a)
                attr_potential = ann[0]['attributes']['potential']

                if attr_potential:
                    if (self.include_potential_gt == True) or (self.include_potential == True):
                        GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
                    else:
                        GT.append(np.zeros((1024,1024)))

                else:
                    GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
                    
        else:
            GT.append(np.zeros((1024,1024)))
        
        GT = np.array(GT)
        GT = Image.fromarray((np.nansum(GT, axis=0)*255).astype(np.uint8)).convert("L")

        if self.width_par != 1024:
            GT = GT.resize((self.width_par , height_par))

        GT = np.array(GT)/255.0

        dilation = False

        if dilation:
            kernel = disk(2)
            n_it = int(self.width_par/64)
            
            GT = ndimage.binary_dilation(GT, structure=kernel, iterations=n_it)
        
        GT_all.append(GT)
        im_all.append(im)

        GT_all = np.array(GT_all)
        im_all = np.array(im_all)
        
        return {'image':torch.tensor(im_all), 'gt':torch.tensor(GT_all), 'names':file_names}
    
    def __len__(self):
        return len(self.img_ids_win)
    
class RundifSequence(Dataset):
    def __init__(self, data_path, annotation_path, im_transform=None, mode='train', win_size=16, stride=2, width_par=128, include_potential=True, include_potential_gt=True, quick_run=False, cross_validation=False, fold_file=None, fold_definition=None):
        
        rng = default_rng()

        self.im_transform = im_transform
        self.mode = mode
        self.width_par = width_par
        self.include_potential = include_potential
        self.include_potential_gt = include_potential_gt
        self.quick_run = quick_run
        self.annotation_path = annotation_path
        self.data_path = data_path

        self.coco_obj = coco.COCO(self.annotation_path)
        
        self.img_ids = self.coco_obj.getImgIds()

        if cross_validation == False:
            self.annotated = []
            self.events = []

            event_list = []
            time_list = {}
            temp_list = []
            
            for a in range(0, len(self.img_ids)):

                anns = self.coco_obj.getAnnIds(imgIds=[self.img_ids[a]])
                time_list[a] = datetime.datetime.strptime(self.coco_obj.loadImgs([self.img_ids[a]])[0]['file_name'].split('/')[-1][:15], '%Y%m%d_%H%M%S')

                if(len(anns)>0):
                    cats = [self.coco_obj.loadAnns(anns[i]) for i in range(len(anns))]
                    attr_potential = [cats[i][0]['attributes']['potential'] for i in range(len(cats))]

                    if not self.include_potential:
                        if all(attr_potential):
                            if len(temp_list) > 0:
                                event_list.append(temp_list)
                                temp_list = []
                            
                        else:
                            self.annotated.append(a)
                            temp_list.append(a)

                            if a == len(self.img_ids)-1:
                                event_list.append(temp_list)
                    else:
                        self.annotated.append(a)
                        temp_list.append(a)

                        if a == len(self.img_ids)-1:
                            event_list.append(temp_list)
                else:
                    if len(temp_list) > 0:
                        event_list.append(temp_list)

                        temp_list = []

            self.events = event_list

            event_ranges = []

            for i in range(len(self.events)):
                len_set = int((len(self.events[i])))

                if len(self.events[i]) < win_size:
                    len_set = int(win_size - len_set)

                if i == 0:
                    diff = [self.events[i][0] - 0, self.events[i+1][0] - self.events[i][-1]]
                    event_ranges.append(check_diff(diff, len_set, self.events[i], time_list, win_size))

                elif i == len(self.events)-1:
                    diff = [self.events[i][0] - event_ranges[i-1][-1], (self.img_ids[-1]-1) - self.events[i][-1]]

                    if diff[1] == 0:
                        diff[1] = 2

                    event_ranges.append(check_diff(diff, len_set, self.events[i], time_list, win_size))

                else:
                    diff = [self.events[i][0] - event_ranges[i-1][-1], self.events[i+1][0] - self.events[i][-1]]
                    event_ranges.append(check_diff(diff, len_set, self.events[i], time_list, win_size))
                        
            len_train = int(len(event_ranges)*0.7)
            len_test = int(len(event_ranges)*0.2)
            len_val = int(len(event_ranges))-(len_train+len_test)

            all_ind = list(np.arange(0, len(event_ranges)))

            ###### DO NOT CHANGE THIS SEED ######
            seed = 42

            random.seed(seed)
            train_ind = random.sample(all_ind, k=len_train)
            all_ind = list(np.setdiff1d(all_ind, train_ind))

            random.seed(seed)
            test_ind = random.sample(all_ind, k=len_test)
            all_ind = np.setdiff1d(all_ind, test_ind)

            val_ind = all_ind.copy()
            
            set_train = [np.array(event_ranges[i])+1 for i in train_ind]
            set_test = [np.array(event_ranges[i])+1 for i in test_ind]
            set_val = [np.array(event_ranges[i])+1 for i in val_ind]

        else:
            fold_dict = np.load(fold_file, allow_pickle=True).item()
            
            train_folds = fold_definition['train']
            test_folds = fold_definition['test']
            val_folds = fold_definition['val']

            set_train = []
            for f in train_folds:
                set_train.extend(fold_dict[f]['image_ids'])

            set_test = []
            for f in test_folds:
                set_test.extend(fold_dict[f]['image_ids'])

            set_val = []
            for f in val_folds:
                set_val.extend(fold_dict[f]['image_ids'])
            
        self.set_train = set_train
        self.set_test = set_test
        self.set_val = set_val
        
        train_paired_idx = []

        for i in range(len(set_train)):
            train_paired_idx.append(np.lib.stride_tricks.sliding_window_view(set_train[i],win_size)[::stride, :])

        test_paired_idx = []

        for i in range(len(set_test)):
            test_paired_idx.append(np.lib.stride_tricks.sliding_window_view(set_test[i],win_size)[::stride, :])

        val_paired_idx = []

        for i in range(len(set_val)):
            val_paired_idx.append(np.lib.stride_tricks.sliding_window_view(set_val[i],win_size)[::stride, :])        

        self.img_ids = np.array(self.img_ids)

        if mode == 'train':
            self.img_ids_win = [item for inner_list in train_paired_idx for item in inner_list]
        elif mode == 'val':
            self.img_ids_win = [item for inner_list in val_paired_idx for item in inner_list]
        elif mode == 'test':
            self.img_ids_win = [item for inner_list in test_paired_idx for item in inner_list]
        else:
            sys.exit('Invalid mode. Specifiy either train, val, or test.')

        if self.quick_run:
            self.img_ids_win = self.img_ids_win[:10]

        self.img_ids_train_win = [item for inner_list in train_paired_idx for item in inner_list]
        self.img_ids_val_win = [item for inner_list in val_paired_idx for item in inner_list]
        self.img_ids_test_win = [item for inner_list in test_paired_idx for item in inner_list]

    # def transform(self, image, mask):

    def __getitem__(self, index):
       
        #seed = int(index)
        
        seed_index = int(datetime.datetime.now().timestamp() * 1000)

        GT_all = []
        im_all = []
        item_ids = self.img_ids_win[index]
        file_names = []

        for idx in item_ids:
            
            img_info = self.coco_obj.loadImgs([idx])[0]
            #img_file_name = img_info["file_name"].split('/')[-1].split('.')[0] + '.npy'
            img_file_name = glob.glob(self.data_path+img_info["file_name"].split('/')[-1].split('.')[0][:16] + '*.npy')
            
            if len(img_file_name) == 0:
                print("Error: No file found for image "+ self.data_path+img_info["file_name"].split('/')[-1].split('.')[0][:16])
                continue
            if len(img_file_name) > 1:
                print(f"Warning: Multiple files found for image ID {idx}. Using the first one: {img_file_name[0]}")
                img_file_name = img_file_name[0].split('/')[-1]
            else:
                img_file_name = img_file_name[0].split('/')[-1]

            height_par = self.width_par

            # Use URL to load image.

            file_names.append(img_file_name)
            im = np.load(self.data_path+img_file_name)

            if self.width_par != 1024:
                im = transform.resize(im, (self.width_par , height_par), anti_aliasing=True, preserve_range=True)

            GT = []
            annotations = self.coco_obj.getAnnIds(imgIds=idx)

            if (len(annotations)>0):
                for a in annotations:
                    
                    ann = self.coco_obj.loadAnns(a)
                    attr_potential = ann[0]['attributes']['potential']

                    if attr_potential:
                        if (self.include_potential_gt == True) or (self.include_potential == True):
                            GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
                        else:
                            GT.append(np.zeros((1024,1024)))

                    else:
                        GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
                        
            else:
                GT.append(np.zeros((1024,1024)))
            
            GT = np.array(GT)
            GT = Image.fromarray((np.nansum(GT, axis=0)*255).astype(np.uint8)).convert("L")

            if self.width_par != 1024:
                GT = GT.resize((self.width_par , height_par))

            GT = np.array(GT)/255.0

            dilation = True

            if dilation:
                kernel = disk(2)
                n_it = int(self.width_par/64)
                
                GT = ndimage.binary_dilation(GT, structure=kernel, iterations=n_it)

            normalize = False

            if normalize:
                vmin = np.nanmedian(im) - 2.5*np.nanstd(im)
                vmax = np.nanmedian(im) + 2.5*np.nanstd(im)

                im[im < vmin] = vmin
                im[im > vmax] = vmax

                im = (im - np.nanmin(im))/(np.nanmax(im) - np.nanmin(im))
                im = np.clip(im, 0, 1)

            torch.manual_seed(seed_index)
            im = self.im_transform(im)

            torch.manual_seed(seed_index)
            GT = self.im_transform(GT)
            
            GT_all.append(GT)
            im_all.append(im)
            
        GT_all = np.array(GT_all)
        im_all = np.array(im_all)

        return {'image':torch.tensor(im_all), 'gt':torch.tensor(GT_all), 'names':file_names}
    
    def __len__(self):
        return len(self.img_ids_win)

class FlowSet(Dataset):
    def __init__(self, transform=None,width_par=128):
        
        rng = default_rng()

        self.width_par = width_par
        self.transform = transform

        self.coco_obj = coco.COCO("instances_clahe.json")
        
        self.img_ids = self.coco_obj.getImgIds()

        self.annotated = []
        self.events = []

        event_list = []
        temp_list = []

        a_id_prev = 0

        for a in range(0, len(self.img_ids)):
            anns = self.coco_obj.getAnnIds(imgIds=[self.img_ids[a]])

            if(len(anns)>0):

                self.annotated.append(a)

                if (self.img_ids[a] - a_id_prev == 1) or (a_id_prev == 0):

                    temp_list.append(a)

                elif (self.img_ids[a] - a_id_prev) > 1:

                    if len(temp_list) > 0:
                        event_list.append(temp_list)

                    temp_list = []
         
                a_id_prev = self.img_ids[a]

        
        self.events = event_list

        len_train = int(len(self.annotated)*0.7)
        boundary_train = self.annotated[len_train]

        for index, ev in enumerate(self.events):
            if boundary_train in ev:
                index_train = index + 1
                break
            else:
                continue

        train_data = self.events[:index_train]

        len_val = int(len(self.annotated)*0.1)
        boundary_val = self.annotated[len_train+len_val]

        for index, ev in enumerate(self.events):
            if boundary_val in ev:
                index_val = index + 1
                break
            else:
                continue

        win_size = 16
        val_data = self.events[index_train:index_val]

        test_data = self.events[index_val:]

        train_data_cme = []

        for sublist in train_data:
            train_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))
        
        train_data_cme = np.array([element for innerList in train_data_cme for element in innerList])

        val_data_cme = []

        for sublist in val_data:
            val_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        val_data_cme = np.array([element for innerList in val_data_cme for element in innerList])

        test_data_cme = []

        for sublist in test_data:
            test_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        test_data_cme = np.array([element for innerList in test_data_cme for element in innerList])

        train_data = [element for innerList in train_data for element in innerList]
        train_data = np.array(train_data)

        val_data = [element for innerList in val_data for element in innerList]
        val_data = np.array(val_data)

        test_data = [element for innerList in test_data for element in innerList]
        test_data = np.array(test_data)

        train_data_noevent = np.arange(np.nanmin(train_data), np.nanmin(val_data))
        train_data_noevent = np.setdiff1d(train_data_noevent, train_data) 

        val_data_noevent = np.arange(np.nanmin(val_data), np.nanmin(test_data))
        val_data_noevent = np.setdiff1d(val_data_noevent, val_data)

        test_data_noevent = np.arange(np.nanmin(test_data), np.nanmax(self.img_ids))
        test_data_noevent = np.setdiff1d(test_data_noevent, test_data)
        
        train_data_noevent = sep_noevent_data(train_data_noevent)
        val_data_noevent = sep_noevent_data(val_data_noevent)
        test_data_noevent = sep_noevent_data(test_data_noevent)
        train_data_nocme = []

        for sublist in train_data_noevent:
            train_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        train_data_nocme = np.array([element for innerList in train_data_nocme for element in innerList])

        val_data_nocme = []

        for sublist in val_data_noevent:
            val_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        val_data_nocme = np.array([element for innerList in val_data_nocme for element in innerList])

        test_data_nocme = []

        for sublist in test_data_noevent:
            test_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        test_data_nocme = np.array([element for innerList in test_data_nocme for element in innerList])

        not_annotated = np.array(np.setdiff1d(np.arange(0,len(self.img_ids)),np.array(self.annotated)))
        self.not_annotated = not_annotated

        seed = 1997
        np.random.seed(seed)

        train_data = np.concatenate([train_data,np.random.choice(self.not_annotated[(self.not_annotated < boundary_train) & (self.not_annotated > train_data[0])], len(train_data), replace=False)])
        val_data = np.concatenate([val_data,np.random.choice(self.not_annotated[(self.not_annotated < boundary_val) & (self.not_annotated > val_data[0])], len(val_data), replace=False)])
        test_data = np.concatenate([test_data,np.random.choice(self.not_annotated[(self.not_annotated > boundary_val) & (self.not_annotated < test_data[-1])], len(test_data), replace=False)])
        
        train_data_paired = np.concatenate([train_data_cme, train_data_nocme[rng.choice(len(train_data_nocme), size=len(train_data_cme), replace=False)]])
        val_data_paired = np.concatenate([val_data_cme, val_data_nocme[rng.choice(len(val_data_nocme), size=len(val_data_cme), replace=False)]])
        test_data_paired = np.concatenate([test_data_cme, test_data_nocme[rng.choice(len(test_data_nocme), size=len(test_data_cme), replace=False)]])

        self.train_data_idx = sorted(train_data)
        self.val_data_idx = sorted(val_data)
        self.test_data_idx = sorted(test_data)

        self.train_paired_idx = train_data_paired
        self.val_paired_idx = val_data_paired
        self.test_paired_idx = test_data_paired

        all_anns = list(self.annotated) + list(self.not_annotated)
        all_anns = sorted(all_anns)

        self.all_anns = all_anns

    def get_img_and_annotation(self,idxs):
        
        seed = np.random.randint(1000)
        torch.manual_seed(seed)

        GT_all = []
        im_all = []

        for idx in idxs:
            
            img_id   = self.img_ids[idx]
            img_info = self.coco_obj.loadImgs([img_id])[0]
            img_file_name = img_info["file_name"].split('/')[-1]

            if torch.cuda.is_available():
                
                if os.path.isdir('/home/mbauer/Data/'):
                    path = "/home/mbauer/Data/differences_clahe/"
                
                elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
                    path = "/gpfs/data/fs72241/maibauer/differences_clahe/"

                else:
                    raise FileNotFoundError('No folder with differences found. Please check path.')
                    sys.exit()

            else:
                path = "/Volumes/SSD/differences_clahe/"

            height_par = self.width_par

            # Use URL to load image.

            im = np.asarray(Image.open(path+img_file_name).convert("L"))

            if self.width_par != 1024:
                im = cv2.resize(im  , (self.width_par , height_par),interpolation = cv2.INTER_CUBIC)

            GT = []
            annotations = self.coco_obj.getAnnIds(imgIds=img_id)

            if(len(annotations)>0):
                for a in annotations:
                    ann = self.coco_obj.loadAnns(a)
                    GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
            
            else:
                GT.append(np.zeros((1024,1024)))
            
            if self.width_par != 1024:
                for i in range(len(GT)):
                    GT[i] = cv2.resize(GT[i]  , (self.width_par , height_par),interpolation = cv2.INTER_CUBIC)

            dilation = False

            if dilation:
                radius = int(3)
                kernel = disk(radius)

                for i in range(len(GT)):
                    GT[i] = binary_dilation(GT[i], footprint=kernel)

            GT = np.array(GT)
            
            cme_pix = np.any(GT, axis=0)*255
            bg_pix = np.all(GT==0, axis=0)*255
            
            if self.transform:
                torch.manual_seed(seed)
                im = im.astype(np.uint8)
                im = self.transform(im)
                im = np.asarray(im.convert("L"))/255.0
            else:
                im = im/255.0
            if self.transform:
                torch.manual_seed(seed)
                cme_pix = cme_pix.astype(np.uint8)
                cme_pix = self.transform(cme_pix)

                torch.manual_seed(seed)
                bg_pix = bg_pix.astype(np.uint8)
                bg_pix = self.transform(bg_pix)

                cme_pix = np.asarray(cme_pix.convert("L"))/255.0
                bg_pix = np.asarray(bg_pix.convert("L"))/255.0

            else:
                cme_pix = cme_pix/255.0
                bg_pix = bg_pix/255.0
            
            GT = np.concatenate([cme_pix[None, :, :],bg_pix[None, :, :]],0)

            GT_all.append(GT)
            im_all.append(im)

            # make sure to apply same tranform to both
        GT_all = np.array(GT_all)
        im_all = np.array(im_all)

        return {'image':torch.tensor(im_all).unsqueeze(1), 'gt':torch.tensor(GT_all)}

    def __len__(self):
        return len(self.all_anns)

    def __getitem__(self, index):
       
        return self.get_img_and_annotation(index)

class BasicSet(Dataset):
    def __init__(self, transform=None,width_par=128):
        
        rng = default_rng()

        width_par = self.width_par
        self.transform = transform

        self.coco_obj = coco.COCO("instances_clahe.json")
        
        self.img_ids = self.coco_obj.getImgIds()

        self.annotated = []
        self.events = []

        event_list = []
        temp_list = []

        a_id_prev = 0

        for a in range(0, len(self.img_ids)):
            anns = self.coco_obj.getAnnIds(imgIds=[self.img_ids[a]])

            if(len(anns)>0):

                self.annotated.append(a)

                if (self.img_ids[a] - a_id_prev == 1) or (a_id_prev == 0):

                    temp_list.append(a)

                elif (self.img_ids[a] - a_id_prev) > 1:

                    if len(temp_list) > 0:
                        event_list.append(temp_list)

                    temp_list = []
         
                a_id_prev = self.img_ids[a]

        
        self.events = event_list

        len_train = int(len(self.annotated)*0.7)
        boundary_train = self.annotated[len_train]

        for index, ev in enumerate(self.events):
            if boundary_train in ev:
                index_train = index + 1
                break
            else:
                continue

        train_data = self.events[:index_train]

        len_val = int(len(self.annotated)*0.1)
        boundary_val = self.annotated[len_train+len_val]

        for index, ev in enumerate(self.events):
            if boundary_val in ev:
                index_val = index + 1
                break
            else:
                continue
        
        val_data = self.events[index_train:index_val]

        test_data = self.events[index_val:]

        train_data_cme = []

        for sublist in train_data:
            train_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,2))
        
        train_data_cme = np.array([element for innerList in train_data_cme for element in innerList])

        val_data_cme = []

        for sublist in val_data:
            val_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,2))

        val_data_cme = np.array([element for innerList in val_data_cme for element in innerList])

        test_data_cme = []

        for sublist in test_data:
            test_data_cme.append(np.lib.stride_tricks.sliding_window_view(sublist,2))

        test_data_cme = np.array([element for innerList in test_data_cme for element in innerList])

        train_data = [element for innerList in train_data for element in innerList]
        train_data = np.array(train_data)

        val_data = [element for innerList in val_data for element in innerList]
        val_data = np.array(val_data)

        test_data = [element for innerList in test_data for element in innerList]
        test_data = np.array(test_data)

        train_data_noevent = np.arange(np.nanmin(train_data), np.nanmin(val_data))
        train_data_noevent = np.setdiff1d(train_data_noevent, train_data) 

        val_data_noevent = np.arange(np.nanmin(val_data), np.nanmin(test_data))
        val_data_noevent = np.setdiff1d(val_data_noevent, val_data)

        test_data_noevent = np.arange(np.nanmin(test_data), np.nanmax(self.img_ids))
        test_data_noevent = np.setdiff1d(test_data_noevent, test_data)
        
        train_data_noevent = sep_noevent_data(train_data_noevent)
        val_data_noevent = sep_noevent_data(val_data_noevent)
        test_data_noevent = sep_noevent_data(test_data_noevent)

        win_size = 2

        train_data_nocme = []

        for sublist in train_data_noevent:
            train_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        train_data_nocme = np.array([element for innerList in train_data_nocme for element in innerList])

        val_data_nocme = []

        for sublist in val_data_noevent:
            val_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        val_data_nocme = np.array([element for innerList in val_data_nocme for element in innerList])

        test_data_nocme = []

        for sublist in test_data_noevent:
            test_data_nocme.append(np.lib.stride_tricks.sliding_window_view(sublist,win_size))

        test_data_nocme = np.array([element for innerList in test_data_nocme for element in innerList])

        not_annotated = np.array(np.setdiff1d(np.arange(0,len(self.img_ids)),np.array(self.annotated)))
        self.not_annotated = not_annotated

        seed = 1997
        np.random.seed(seed)

        train_data = np.concatenate([train_data,np.random.choice(self.not_annotated[(self.not_annotated < boundary_train) & (self.not_annotated > train_data[0])], len(train_data), replace=False)])
        val_data = np.concatenate([val_data,np.random.choice(self.not_annotated[(self.not_annotated < boundary_val) & (self.not_annotated > val_data[0])], len(val_data), replace=False)])
        test_data = np.concatenate([test_data,np.random.choice(self.not_annotated[(self.not_annotated > boundary_val) & (self.not_annotated < test_data[-1])], len(test_data), replace=False)])
        
        train_data_paired = np.concatenate([train_data_cme, train_data_nocme[rng.choice(len(train_data_nocme), size=len(train_data_cme), replace=False)]])
        val_data_paired = np.concatenate([val_data_cme, val_data_nocme[rng.choice(len(val_data_nocme), size=len(val_data_cme), replace=False)]])
        test_data_paired = np.concatenate([test_data_cme, test_data_nocme[rng.choice(len(test_data_nocme), size=len(test_data_cme), replace=False)]])

        self.train_data_idx = sorted(train_data)
        self.val_data_idx = sorted(val_data)
        self.test_data_idx = sorted(test_data)

        self.train_paired_idx = train_data_paired
        self.val_paired_idx = val_data_paired
        self.test_paired_idx = test_data_paired

        all_anns = list(self.annotated) + list(self.not_annotated)
        all_anns = sorted(all_anns)

        self.all_anns = all_anns

    def get_img_and_annotation(self,idx):

        img_id   = self.img_ids[idx]
        img_info = self.coco_obj.loadImgs([img_id])[0]
        img_file_name = img_info["file_name"].split('/')[-1]

        if torch.cuda.is_available():
            
            if os.path.isdir('/home/mbauer/Data/'):
                path = "/home/mbauer/Data/differences_clahe/"
            
            elif os.path.isdir('/gpfs/data/fs72241/maibauer/'):
                path = "/gpfs/data/fs72241/maibauer/differences_clahe/"

            else:
                raise FileNotFoundError('No folder with differences found. Please check path.')
                sys.exit()

        else:
            path = "/Volumes/SSD/differences_clahe/"

        height_par = self.width_par

        # Use URL to load image.

        im = np.asarray(Image.open(path+img_file_name).convert("L"))

        if self.width_par != 1024:
            im = cv2.resize(im  , (self.width_par , height_par),interpolation = cv2.INTER_CUBIC)

        GT = []
        annotations = self.coco_obj.getAnnIds(imgIds=img_id)

        if(len(annotations)>0):
            for a in annotations:
                ann = self.coco_obj.loadAnns(a)
                GT.append(coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0])
        
        else:
            GT.append(np.zeros((1024,1024)))
        
        if self.width_par != 1024:
            for i in range(len(GT)):
                GT[i] = cv2.resize(GT[i]  , (self.width_par , height_par), interpolation = cv2.INTER_CUBIC)

        dilation = False

        if dilation:
            radius = int(3)
            kernel = disk(radius)

            for i in range(len(GT)):
                GT[i] = binary_dilation(GT[i], footprint=kernel)

        GT = np.array(GT)
        
        cme_pix = np.any(GT, axis=0)*255
        bg_pix = np.all(GT==0, axis=0)*255

        seed = np.random.randint(1000)
        torch.manual_seed(seed)
       
        if self.transform:
            im = im.astype(np.uint8)
            im = self.transform(im)
            im = np.asarray(im.convert("L"))/255.0
        else:
            im = im/255.0

        torch.manual_seed(seed)

        if self.transform:
            cme_pix = cme_pix.astype(np.uint8)
            cme_pix = self.transform(cme_pix)
        
            bg_pix = bg_pix.astype(np.uint8)
            bg_pix = self.transform(bg_pix)

            cme_pix = np.asarray(cme_pix.convert("L"))/255.0
            bg_pix = np.asarray(bg_pix.convert("L"))/255.0

        else:
            cme_pix = cme_pix/255.0
            bg_pix = bg_pix/255.0
        
        GT = np.concatenate([cme_pix[None, :, :],bg_pix[None, :, :]],0)

        # make sure to apply same tranform to both

        return torch.tensor(im).unsqueeze(0), torch.tensor(GT)

    def __len__(self):
        return len(self.all_anns)

    def __getitem__(self, index):
       
        return self.get_img_and_annotation(index)
    
if __name__ == "__main__":
    dataset = RundifSequence(data_path='/media/DATA_DRIVE/differences_pickles/', annotation_path='/home/mbauer/Code/CME_ML/instances_default.json', 
                             im_transform=v2.Compose([v2.ToTensor()]), mode='val', win_size=16, stride=2, width_par=128, include_potential=True, include_potential_gt=True, quick_run=False)
    
    print('length train', len(dataset.img_ids_train_win))
    print('length val', len(dataset.img_ids_val_win))
    print('length test', len(dataset.img_ids_test_win))
    print(len(dataset.img_ids_train_win)+len(dataset.img_ids_val_win)+len(dataset.img_ids_test_win))

    data_loader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=2,
                                                pin_memory=False
                                            )

    for num, data in enumerate(data_loader):
        print('batch', num)
        print('image shape', data['image'].shape)
        print('gt shape', data['gt'].shape)
        print('names', np.shape(data['names']))
        
        sys.exit(0)