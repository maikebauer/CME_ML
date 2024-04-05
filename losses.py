import torch
from torch import nn
import torch.nn.functional as F

def charbonnier_loss(delta, gamma=0.45, epsilon=1e-6):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), gamma))
    return loss


class spatial_smoothing_loss(nn.Module):
    #from https://akshay-sharma1995.github.io/files/ml_temp_loss.pdf
    def __init__(self, device):
        super(spatial_smoothing_loss, self).__init__()
        self.eps = 1e-6
        self.device = device

    def forward(self, X):  # X is flow map
        u = X[:, 0:1]
        # Rest of the code
        v = X[:,1:2]
        # print("u",u.size())
        hf1 = torch.tensor([[[[0,0,0],[-1,2,-1],[0,0,0]]]]).type(torch.FloatTensor).to(self.device)
        hf2 = torch.tensor([[[[0,-1,0],[0,2,0],[0,-1,0]]]]).type(torch.FloatTensor).to(self.device)
        hf3 = torch.tensor([[[[-1,0,-1],[0,4,0],[-1,0,-1]]]]).type(torch.FloatTensor).to(self.device)
        # diff = torch.add(X, -Y)
        
        u_hloss = F.conv2d(u,hf1,padding=1,stride=1)
        # print("uhloss",type(u_hloss))
        u_vloss = F.conv2d(u,hf2,padding=1,stride=1)
        u_dloss = F.conv2d(u,hf3,padding=1,stride=1)

        v_hloss = F.conv2d(v,hf1,padding=1,stride=1)
        v_vloss = F.conv2d(v,hf2,padding=1,stride=1)
        v_dloss = F.conv2d(v,hf3,padding=1,stride=1)

        u_hloss = charbonier(u_hloss,self.eps)
        u_vloss = charbonier(u_vloss,self.eps)
        u_dloss = charbonier(u_dloss,self.eps)

        v_hloss = charbonier(v_hloss,self.eps)
        v_vloss = charbonier(v_vloss,self.eps)
        v_dloss = charbonier(v_dloss,self.eps)


        # error = torch.sqrt( diff * diff + self.eps )
        # loss = torch.sum(error) 
        loss = u_hloss + u_vloss + u_dloss + v_hloss + v_vloss + v_dloss
        # print('char_losss',loss)
        return loss
    
def charbonier(x,eps):
	gamma = 0.45
	# print("x.type",type(x))
	loss = x*x + eps*eps
	loss = torch.pow(loss,gamma)
	loss = torch.mean(loss)
	return loss

def miou_loss(pred_im2, real_im2):
    """
    Differentiable mean IOU loss, as defined in Varghese 2021, equation 11.
    """
    
    prod_im2 = pred_im2 * real_im2

    sum_im2 = pred_im2 + real_im2

    loss = torch.sum(torch.abs(prod_im2))/torch.sum(torch.abs(sum_im2 - prod_im2))

    return loss
