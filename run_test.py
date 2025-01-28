import torch
from torch import nn
import numpy as np
import sys
from dataset import RundifSequence_Test
from utils import load_model, parse_yml
from torchvision.transforms import v2

ml_path = '/home/mbauer/Code/CME_ML/Model_Train/run_17012025_161748_model_cnn3d/'
model_path = ml_path + 'model_seg.pth'
config_path = ml_path + 'config.yaml'
device = 'cuda:1'

config = parse_yml(config_path)
batch_size = config['train']['batch_size']
num_workers = config['train']['num_workers']
backbone = config['model']['name']

sigmoid = nn.Sigmoid()

model_seg = load_model(config, 'test', model_path)

model_seg.to(device)

year = '2009'
data_path = '/media/DATA_DRIVE/mbauer/Test_Data/' + year + '/'
win_size = 16
stride = 2
width_par = 128

if __name__ == "__main__":
    dataset = RundifSequence_Test(data_path=data_path,win_size=win_size,stride=stride,width_par=width_par)

    data_loader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )

    test_list_names = np.array([l.tolist() for l in dataset.files_strided])

    pred_save = []

    n_val = len(set(test_list_names.flatten()))

    input_imgs = np.zeros((n_val, width_par, width_par))
    input_names = np.chararray((n_val), itemsize=len(test_list_names[0][0]))

    for val in range(n_val):
        pred_save.append([])

    # model_seg.eval()

    ind_keys = sorted(set(np.array(test_list_names).flatten()))
    ind_set = np.arange(0, len(ind_keys))
    ind_dict = {}

    for A, B in zip(ind_keys, ind_set):
        ind_dict[A] = B

    num_batch = 0

    for num, data in enumerate(data_loader):
        print(f'Processing: {num} of {dataset.__len__()}')
        input_data = data['image'].float().to(device)
        name_data = np.array(data['names'])

        if backbone == 'unetr':
            
            im_concat = torch.permute(input_data, (0, 2, 3, 4, 1))
            pred_comb = model_seg(im_concat)
            input_data = torch.permute(input_data, (0, 2, 1, 3, 4))

        elif backbone == 'cnn3d' or backbone == 'resunetpp' or backbone == 'maike_cnn3d':

            input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
            name_data = np.transpose(name_data)

            pred_comb = model_seg(input_data)

            if config['model']['final_layer'].lower() == 'none':
                pred_comb = sigmoid(pred_comb)
                
        else:
            print('Invalid backbone...')
            sys.exit()

        
        for b in range(pred_comb.shape[0]):
            for k in range(win_size):

                current_ind = ind_dict[test_list_names[num_batch][k]]

                pred_save[current_ind].append(pred_comb[b,0,k,:,:].cpu().detach().numpy())

                if np.all(input_imgs[current_ind]) == 0:
                    input_imgs[current_ind] = input_data[b,0,k,:,:].cpu().detach().numpy()
                    input_names[current_ind] = name_data[b,k]

            num_batch = num_batch + 1

    input_imgs = np.array(input_imgs)

    pred_final = np.zeros((n_val, width_par, width_par))
    pred_final_mean = np.zeros((n_val, width_par, width_par))
    pred_final_median = np.zeros((n_val, width_par, width_par))
    pred_final_max = np.zeros((n_val, width_par, width_par))

    for h, pred_arr in enumerate(pred_save):
        pred_prep = np.zeros((len(pred_arr), width_par, width_par))
        
        for j in range(len(pred_arr)):
            pred_prep[j] = pred_arr[j]

        pred_final[h] = np.nanmedian(pred_prep, axis=0)
        pred_final_mean[h] = np.nanmean(pred_prep, axis=0)
        pred_final_median[h] = np.nanmedian(pred_prep, axis=0)
        pred_final_max[h] = np.nanmax(pred_prep, axis=0)

    np.savez_compressed(ml_path+"segmentation_masks_median_"+year+".npz", masks=pred_final_median.astype(np.float16), filenames=input_names)
    np.savez_compressed(ml_path+"segmentation_masks_mean_"+year+".npz", masks=pred_final_mean.astype(np.float16), filenames=input_names)
    np.savez_compressed(ml_path+"segmentation_masks_max_"+year+".npz", masks=pred_final_max.astype(np.float16), filenames=input_names)