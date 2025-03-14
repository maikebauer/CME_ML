    
import torch
import torch.nn.functional as F
import numpy as np
import sys
import yaml
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from models import UNETR_16, CNN3D, ResUnetPlusPlus, Maike_CNN3D
from losses import AdaptiveWingLoss, DiceBCELoss, AsymmetricFocalLoss, AsymmetricFocalTverskyLoss, AsymmetricUnifiedFocalLoss
from monai.losses.dice import DiceLoss
import os

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for _, data in enumerate(loader):
        batch_size, num_channels, depth, height, width = data['image'].shape
        num_pixels += batch_size * height * width * depth
        mean += data['image'].mean(axis=(0, 2, 3, 4)).sum()
        std += data['image'].std(axis=(0, 2, 3, 4)).sum()
        
    mean /= num_pixels
    std /= num_pixels

    return [mean], [std]


def sep_noevent_data(data_noevent):

    temp_list = []
    data_nocme = []

    ev_id_prev = 0
    
    for i, noev in enumerate(data_noevent):

        if (noev - ev_id_prev == 1) or (ev_id_prev == 0):

            temp_list.append(i)

        elif (noev - ev_id_prev) > 1:

            if len(temp_list) > 0:
                data_nocme.append(temp_list)

            temp_list = []
    
        ev_id_prev = data_noevent[i]
    
    return data_nocme


    
def backward_warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default for PyTorch version < 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output

def check_diff(diff, len_set, evs, time_dict, win_size):
    
    flg_bef = 0
    flg_aft = 0

    max_diff = len_set

    if (diff[0] > max_diff) and (diff[1] > max_diff):
        cs = 1
        ev_ran = np.arange(evs[0]-max_diff,evs[-1]+max_diff)

    elif (diff[0] > max_diff) and (diff[1] <= max_diff):
        cs = 2
        ev_ran = np.arange(evs[0]-max_diff,evs[-1]+diff[1]-1)

    elif (diff[0] <= max_diff) and (diff[1] > max_diff):
        cs = 3
        ev_ran = np.arange(evs[0]-diff[0]+1,evs[-1]+max_diff)

    elif (diff[0] <= max_diff) and (diff[1] <= max_diff):
        cs = 4
        ev_ran = np.arange(evs[0]-diff[0]+1,evs[-1]+diff[1]-1)
    else:
        print('Error')

    min_ind = np.where(ev_ran == evs[0])[0][0]
    max_ind = np.where(ev_ran == evs[-1])[0][0]

    for i in range(min_ind, 0, -1):
        tdiff = time_dict[ev_ran[i]] - time_dict[ev_ran[i-1]]

        if np.abs(tdiff.seconds)/60 > 3.5*40:
            flg_bef = 1
            ev_ran = ev_ran[i:]
            break
    
    for i in range(max_ind, len(ev_ran)-1):
        tdiff = time_dict[ev_ran[i+1]] - time_dict[ev_ran[i]]

        if np.abs(tdiff.seconds)/60 > 3.5*40:
            flg_aft = 1
            ev_ran = ev_ran[:i+1]
            break
    
    if len(ev_ran) < win_size:
        if flg_bef and (cs == 2 or cs == 4):
            print('Error: flg_bef = {}, cs = {}'.format(flg_bef, cs))
            sys.exit()
        elif flg_aft and (cs == 3 or cs == 4):
            print('Error: flg_aft = {}, cs = {}'.format(flg_aft, cs))
            sys.exit()
        elif flg_bef and flg_aft:
            print('Error: flg_bef = {}, flg_aft = {}'.format(flg_bef, flg_aft))
            sys.exit()
        elif flg_bef and (cs == 1 or cs == 3):
            # print('Error: flg_bef = {}, cs = {}'.format(flg_bef, cs))
            # print('Fixing')
            ev_ran = np.arange(ev_ran[0],ev_ran[-1]+max_diff)

            max_ind = np.where(ev_ran == evs[-1])[0][0]
            
            for i in range(max_ind, len(ev_ran)-1):
                tdiff = time_dict[ev_ran[i+1]] - time_dict[ev_ran[i]]

                if np.abs(tdiff.seconds)/60 > 3.5*40:
                    flg_aft = 1
                    ev_ran = ev_ran[:i+1]
                    break
        
        elif flg_aft and (cs == 1 or cs == 2):
            # print('Error: flg_aft = {}, cs = {}'.format(flg_aft, cs))
            # print('Fixing')
            ev_ran = np.arange(ev_ran[0]-2*max_diff,ev_ran[-1])

            min_ind = np.where(ev_ran == evs[0])[0][0]

            for i in range(min_ind, 0, -1):
                tdiff = time_dict[ev_ran[i]] - time_dict[ev_ran[i-1]]

                if np.abs(tdiff.seconds)/60 > 3.5*40:
                    flg_bef = 1
                    ev_ran = ev_ran[i:]
                    break

    return ev_ran

def parse_yml(config_path):
    """
    Parses configuration file.

    @return: Configuration file content
    """
    with open(config_path) as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return content


def image_grid(images):
    """
    Plots a grid of images.
    Args:
        images (list or np.array): List or array of images to plot.
    """
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(16, 16))
    for i in range(len(images)):
        # Start next subplot.
        ax = plt.subplot(8, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.tight_layout()

        if i < 16:
            plt.imshow(images[i], cmap='gray', vmin=np.median(images[i])-np.std(images[i]), vmax=np.median(images[i])+np.std(images[i]))
        else:
            plt.imshow(images[i], cmap='gray')

    return figure


def load_augmentations(config):

    AUGMENTATION_MAP = {
    "ToTensor": v2.RandomHorizontalFlip,
    "RandomHorizontalFlip": v2.RandomHorizontalFlip,
    "RandomVerticalFlip": v2.RandomVerticalFlip,
    "RandomAutocontrast": v2.RandomAutocontrast,
    "RandomEqualize": v2.RandomEqualize,
    "RandomPhotometricDistort": v2.RandomPhotometricDistort,
    "ToImage": v2.ToImage,
    "ToDtype": v2.ToDtype,
    "ToTensor": v2.ToTensor,
    "GaussianBlur": v2.GaussianBlur,
    "ElasticTransform": v2.ElasticTransform,
    }

    TORCH_DTYPES = {
        'float32': torch.float32,
        'float64': torch.float64
    }

    augmentations = []

    for aug in config['train']['data_augmentation']:
        name = aug['name']
        if name in AUGMENTATION_MAP:
            # Get the class
            aug_class = AUGMENTATION_MAP[name]
            # Get the parameters, excluding the 'name' key
            params = {key: value for key, value in aug.items() if key != 'name'}

            if 'dtype' in params:
                params['dtype'] = TORCH_DTYPES[params['dtype']]

            # Instantiate the augmentation with its parameters
            if 'randomize' in params and params['randomize'] > 0:
                del params['randomize']
                augmentations.append(v2.RandomApply([aug_class(**params)]))
            elif 'randomize' in params and params['randomize'] == 0:
                del params['randomize']
                augmentations.append(aug_class(**params))
            else:
                augmentations.append(aug_class(**params))

        else:
            raise ValueError(f"Unknown augmentation: {name}")
        
    return v2.Compose(augmentations)

def load_model(config, mode, test_model=''):

    MODEL_MAP = {
    "unetr": UNETR_16,
    "cnn3d": CNN3D,
    "resunetpp": ResUnetPlusPlus,
    "maike_cnn3d": Maike_CNN3D,
    }

    model_type = config['model']['name']
    model_params = config['model']['model_parameters']
    seed = config['model']['seed']

    if model_type in MODEL_MAP:
        torch.manual_seed(seed)
        model = MODEL_MAP[model_type](**model_params)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    if (mode == 'train') and (config['train']['load_checkpoint']['load_model']):
        checkpoint = torch.load(config['train']['load_checkpoint']['checkpoint_path'], weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    elif (mode == 'test' or mode == 'val'):
        if os.path.exists(test_model):
            checkpoint = torch.load(test_model, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Model file not found: {test_model}")

    return model

def load_optimizer(config, model_params):

    OPTIMIZER_MAP = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    }

    optimizer_type = config['optimizer']['name']
    optimizer_params = config['optimizer']['optimizer_parameters']
    
    if optimizer_type in OPTIMIZER_MAP:
        optimizer = OPTIMIZER_MAP[optimizer_type](model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    if config['train']['load_checkpoint']['load_optimizer']:
        checkpoint = torch.load(config['train']['load_checkpoint']['checkpoint_path'], weights_only=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'updated_lr' in config['train']['load_checkpoint']:
            for g in optimizer.param_groups:
                g['lr'] = config['train']['load_checkpoint']['updated_lr']
    
    return optimizer

def load_scheduler(config, optimizer):

    SCHEDULER_MAP = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

    scheduler_type = config['scheduler']['name']
    scheduler_params = config['scheduler']['scheduler_parameters']
    
    if scheduler_type in SCHEDULER_MAP:
        scheduler = SCHEDULER_MAP[scheduler_type](optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    return scheduler

def load_loss(config):

    LOSS_MAP = {
    "AdaptiveWingLoss": AdaptiveWingLoss,
    "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
    "DiceLoss": DiceLoss,
    "BCELoss": torch.nn.BCELoss,
    "DiceBCELoss": DiceBCELoss,
    "AsymmetricFocalLoss": AsymmetricFocalLoss,
    "AsymmetricFocalTverskyLoss": AsymmetricFocalTverskyLoss,
    "AsymmetricUnifiedFocalLoss": AsymmetricUnifiedFocalLoss,
    }

    loss_type = config['loss']['name']
    loss_params = config['loss']['loss_parameters']
    
    if 'pos_weight' in loss_params:
        loss_params['pos_weight'] = torch.tensor(loss_params['pos_weight']).to(config['model']['device'])

    elif ('bce_params' in loss_params) and ('pos_weight' in loss_params['bce_params']):
        loss_params['bce_params']['pos_weight'] = torch.tensor(loss_params['bce_params']['pos_weight']).to(config['model']['device'])

    if loss_type in LOSS_MAP:
        loss = LOSS_MAP[loss_type](**loss_params)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")

    return loss