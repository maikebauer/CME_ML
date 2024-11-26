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

class RundifSequence(Dataset):
    def __init__(self, data_path, annotation_path, transform=None, mode='train', win_size=16, stride=2, width_par=128, include_potential=True, include_potential_gt=False, quick_run=False):
        
        rng = default_rng()

        self.transform = transform
        self.mode = mode
        self.width_par = width_par
        self.include_potential = include_potential
        self.include_potential_gt = include_potential_gt
        self.quick_run = quick_run
        self.annotation_path = annotation_path
        self.data_path = data_path

        self.coco_obj = coco.COCO(self.annotation_path)
        
        self.img_ids = self.coco_obj.getImgIds()

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
            len_set = int(len(self.events[i])/2)

            if len(self.events[i]) < win_size/2:
                len_set = int(win_size/2 - len_set)

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

        for i, ran in enumerate(set_test):
        
            if 3880 in ran:
                ran = np.delete(ran, np.where(ran == 3880))
            if 3881 in ran:
                ran = np.delete(ran, np.where(ran == 3881))

            if 3882 in ran:
                ran = np.delete(ran, np.where(ran == 3882))

            set_test[i] = ran
            
        set_val = [np.array(event_ranges[i])+1 for i in val_ind]

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

        not_annotated = np.array(np.setdiff1d(np.arange(0,len(self.img_ids)),np.array(self.annotated)))
        self.not_annotated = not_annotated

        all_anns = list(self.annotated) + list(self.not_annotated)
        all_anns = sorted(all_anns)

        self.all_anns = all_anns

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
            #self.img_ids_win = self.img_ids_win[:int(len(self.img_ids_win)/8)]
            self.img_ids_win = self.img_ids_win[:10]

        self.win_train = sorted(set(np.array([item for inner_list in train_paired_idx for item in inner_list]).flatten()))
        self.win_val = sorted(set(np.array([item for inner_list in val_paired_idx for item in inner_list]).flatten()))
        self.win_test = sorted(set(np.array([item for inner_list in test_paired_idx for item in inner_list]).flatten()))

    def __getitem__(self, index):
       
        seed = int(index)

        GT_all = []
        im_all = []
        
        item_ids = self.img_ids_win[index]
        file_names = []

        for idx in item_ids:
            
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
                        if self.include_potential_gt == True:
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

            torch.manual_seed(seed)
            im = self.transform(im)

            torch.manual_seed(seed)
            GT = self.transform(GT)
            
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