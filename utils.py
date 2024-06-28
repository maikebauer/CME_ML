    
import torch
import torch.nn.functional as F
import numpy as np

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

def calc_conv(w,k,s,p):

    sz = ((w-k+2*p)/s).astype(int)+1

    return tuple(sz)

def calc_kernel(inp_size, kernel_size, depth):

    kernels = []
    outputs = np.zeros((depth+1, 3))

    outputs[0] = inp_size

    stride = 2
    padding = 0

    for i in range(1,depth+1):

        ks = outputs[i-1] % kernel_size
        ks[ks == 0] = kernel_size
        kernels.append(tuple(map(int, ks)))

        outputs[i] = calc_conv(outputs[i-1], kernels[i-1], s=stride, p=padding)

    return kernels

def check_diff(diff, len_set, evs):
    
    if (diff[0] > len_set) and (diff[1] > len_set):
        ev_ran = np.arange(evs[0]-len_set,evs[-1]+len_set)

    elif (diff[0] > len_set) and (diff[1] <= len_set):
        ev_ran = np.arange(evs[0]-len_set,evs[-1]+diff[1]-1)

    elif (diff[0] <= len_set) and (diff[1] > len_set):
        ev_ran = np.arange(evs[0]-diff[0]+1,evs[-1]+len_set)

    elif (diff[0] <= len_set) and (diff[1] <= len_set):
        ev_ran = np.arange(evs[0]-diff[0]+1,evs[-1]+diff[1]-1)
    else:
        print('Error')

    return ev_ran