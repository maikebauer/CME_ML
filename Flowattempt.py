import torch 
from torch import nn,optim
import torch.nn.functional as F
import b2s_dataset 
from kornia.geometry.transform import translate,warp_affine
import matplotlib.pyplot as plt 
import cv2 as cv
import numpy as np 
from ESRGAN import *
from kornia.enhance.equalization import equalize_clahe
import cv2 


class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2*in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out

    
device = torch.device("cpu")
if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    device = torch.device("cuda")


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

def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss



def train():
    model = FNet(1).to(device)
    dataset = b2s_dataset.CombinedDataloader(256,512,"../images3",True)
    dataloader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                num_workers=1,
                                                pin_memory=False
                                            )
    
    g_optimizer = optim.Adam(model.parameters(),1e-5)
    pixel_looser= nn.L1Loss(reduction="mean")

    # g_optimizer.load_state_dict(torch.load("flowopti.pth"))
    # model.load_state_dict(torch.load("flowgen.pth"))

    losses = []
    for i in range(0,200):
        losses2 = []
        for data in dataloader:
            LR1 = data["LR1"].to(device) #beacon1
            LR2 = data["LR2"].to(device) #beacon2
            HR1 = data["HR1"].to(device) #science1
            HR2 = data["HR2"].to(device) #science2

            g_optimizer.zero_grad()
            
            #
            #translate science1 to science2 according to shift_arr from header
            tHR1 = translate(HR1.unsqueeze(1),data["tr1"].float().to(device)/2,mode='bilinear',padding_mode='border')

           
      
            #calculates flow between science2 and science1 shifted to science2 with header shifts
            flow = model(HR2.unsqueeze(1),tHR1)
            #calculates difference between science2 and science1 shifted to science2 with header shifts and propagataed with flow
            difference = HR2.unsqueeze(1) - backward_warp(tHR1,flow)
            loss = charbonnier_loss( (difference- difference.min())/(difference.max()-difference.min()))###,
            
            loss.backward()
            g_optimizer.step()
            losses2.append(loss.item())

        # if(i==100):
        #     for g in g_optimizer.param_groups:
        #         g['lr'] = g['lr'] * 0.1


        losses.append(np.array(losses2).sum()/dataset.__len__())
        print(data["tr1"].float()/2,flow[0,0,:,:].mean().item(),flow[0,1,:,:].mean().item())


        flow2 = flow[0].detach().cpu().numpy()
        magnitude = np.sqrt(flow2[0]**2+flow2[1]**2)
        # Sets image hue according to the optical flow  
      
        difference =HR2[0]-tHR1[0][0]

        difference =  equalize_clahe((difference- difference.min())/(difference.max()-difference.min()),100.0,(4,4)).detach().cpu().numpy()
        
        # difference = (difference*255.0).astype(np.uint8)
        # clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))
        # difference = clahe.apply(difference)



        fig,ax = plt.subplots(4,2,figsize=(20, 15))
        ax[0][0].plot(losses)
        ax[0][0].set_yscale('log')
        ax[0][1].plot(losses)
        ax[0][0].set_yscale('log')

        ax[1][0].imshow(data["HR1"][0].detach().cpu().numpy())
        ax[1][1].imshow(data["HR2"][0].detach().cpu().numpy())

        ax[2][0].imshow(magnitude)
        nvec = 25  # Number of vectors to be displayed along each image dimension
        nl, nc = 512,512
        step = max(nl//nvec, nc//nvec)

        y, x = np.mgrid[:nl:step, :nc:step]
        u_ = flow2[0][::step, ::step]
        v_ = flow2[1][::step, ::step]


        ax[2][0].quiver(x, y, u_, v_, color='r', units='dots',angles='xy', scale_units='xy', lw=3)
        ax[2][1].imshow(difference,cmap='twilight')

        ax[3][0].imshow(backward_warp(tHR1,flow)[0][0].detach().cpu().numpy())
        difference2 = HR2[0]-backward_warp(tHR1,flow)[0][0]
        difference2 = equalize_clahe((difference2- difference2.min())/(difference2.max()-difference2.min()),100.0,(4,4)).detach().cpu().numpy()
        ax[3][1].imshow(
                            difference2
                        ,cmap='twilight')
        plt.savefig("flow_test_res/"+str(i)+"_3.png")
        plt.close('all')
        # plt.show()
        # exit()
        torch.save(model.state_dict(), "flowgen.pth")
        torch.save(g_optimizer.state_dict(),"flowopti.pth")


if __name__ == "__main__":
    train()