import torch
from torch import nn
import torch.nn.functional as F
from monai.losses.dice import DiceLoss
import matplotlib.pyplot as plt
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

def miou_loss(pred, gt):
    """
    Differentiable mean IOU loss, as defined in Varghese 2021, equation 11.
    """
    
    prod_im = pred * gt

    sum_im = pred + gt

    loss = 1 - torch.sum(torch.abs(prod_im))/torch.sum(torch.abs(sum_im - prod_im))

    return loss

def dice_loss(pred, gt):
    """
    Differentiable dice loss.
    """
    
    prod_im = pred * gt

    sum_im = pred + gt

    loss = 1 - torch.sum(2*prod_im+1)/torch.sum(sum_im+1)

    return loss

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class DiceBCELoss(nn.Module):
    def __init__(self, dice_params, bce_params):
        super(DiceBCELoss, self).__init__()

        self.dice_params = dice_params
        self.bce_params = bce_params

        self.dice = DiceLoss(**self.dice_params)
        self.bce = nn.BCEWithLogitsLoss(**self.bce_params)

    def forward(self, pred, target):

        y = target
        y_hat = pred

        loss_dice = self.dice(y_hat, y)
        loss_bce = self.bce(y_hat, y)
        # print('dice',loss_dice)
        # print('bce',loss_bce)
        # ys = torch.nn.Sigmoid()(y_hat)

        # fig, axs = plt.subplots(3, 16, figsize=(20, 5))
        # for b in range(2):
        #     for c in range(16):
        #         axs[0, c].imshow(ys[b, 0, c, :, :].detach().cpu().numpy(), cmap='gray')
        #         axs[0, c].set_title(f'Sigmoid Output {c}')
        #         axs[0, c].axis('off')
        #         axs[1, c].imshow(y_hat[b, 0, c, :, :].detach().cpu().numpy(), cmap='gray')
        #         axs[1, c].set_title(f'Predicted {c}')
        #         axs[1, c].axis('off')
        #         axs[2, c].imshow(y[b, 0, c, :, :].detach().cpu().numpy(), cmap='gray')
        #         axs[2, c].set_title(f'Target {c}')
        #         axs[2, c].axis('off')
        #     plt.show()

        return (loss_dice + loss_bce)/2
    

# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]
    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=2):
        super(AsymmetricFocalLoss, self).__init__()

        self.delta = delta
        self.gamma = gamma

    def forward(self, pred, target):
        """For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
        """
        axis = identify_axis(target.shape)  

        epsilon = 1e-7
        pred = torch.clip(pred, epsilon, 1. - epsilon)
        pred_back = 1 - pred
        target_back = 1 - target

        cross_entropy_fore = -target * torch.log(pred)
        cross_entropy_back = -(target_back) * torch.log(pred_back)
        
        #calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - (pred_back), self.gamma) * cross_entropy_back
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = cross_entropy_fore
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.cat([back_ce, fore_ce],dim=1),dim=1))

        return loss

class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=0.75):
        super(AsymmetricFocalTverskyLoss, self).__init__()

        self.delta = delta
        self.gamma = gamma

    def forward(self, pred, target):
        """This is the implementation for binary segmentation.
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            focal parameter controls degree of down-weighting of easy examples, by default 0.75
        """
        axis = identify_axis(target.shape)  

        epsilon = 1e-7
        pred = torch.clip(pred, epsilon, 1. - epsilon)

        pred_back = 1 - pred
        target_back = 1 - target

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(target * pred, dim=axis)
        fn = torch.sum(target * (1-pred), dim=axis)
        fp = torch.sum((1-target) * pred, dim=axis)
        dice_class_fore = (tp + epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + epsilon)

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(target_back * pred_back, dim=axis)
        fn = torch.sum(target_back * pred, dim=axis)
        fp = torch.sum(target * pred_back, dim=axis)
        dice_class_back = (tp + epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class_back) 
        fore_dice = (1-dice_class_fore) * torch.pow(1-dice_class_fore, -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.cat([back_dice,fore_dice],dim=1))
        return loss

class AsymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(AsymmetricUnifiedFocalLoss, self).__init__()

        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, pred, target):
        """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
        Parameters
        ----------
        weight : float, optional
            represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
        delta : float, optional
            controls weight given to each class, by default 0.6
        gamma : float, optional
            focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
        """
        asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(pred, target)
        asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(pred, target)

        if self.weight is not None:
            loss = (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)  
        else:
            loss = asymmetric_ftl + asymmetric_fl
        
        return loss
    