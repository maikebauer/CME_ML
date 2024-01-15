import torch
from torch import nn,optim
import torch.nn.functional as F
import glob
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pycocotools import coco
import matplotlib.pyplot as plt 
import numpy as np
import time 
from torchvision.transforms import v2
import sys
import model_summed_aug
from matplotlib.colors import ListedColormap
from datetime import datetime
import os

def Kappa_cohen(predictions,groundtruth):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, groundtruth == 1))
     
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, groundtruth == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, groundtruth == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN   = np.sum(np.logical_and(predictions == 0, groundtruth ==  1))
    gt_p = np.sum(groundtruth ==  0)
    gt_r = np.sum(groundtruth ==  1)
        

    observed_accuracy  =   (TP+TN)/groundtruth.shape[0]
    expected_accuracy  =   ((gt_r*TP)/groundtruth.shape[0] + (gt_p*TN)/groundtruth.shape[0])/groundtruth.shape[0]

    return (observed_accuracy - expected_accuracy)/ (1- expected_accuracy +0.000005)

def precision_recall(predictions,groundtruth):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, groundtruth == 1))
     
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, groundtruth == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, groundtruth == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(predictions == 0, groundtruth ==  1))
    precision = TP/ (TP+FP+0.000005)
    recall    = TP/ (TP+FN+0.000005)

    if np.sum(groundtruth) == 0:
        recall = np.nan
        precision = np.nan

    return precision,recall,TP,FP,TN,FN

def IoU(predictions,groundtruth):
   # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, groundtruth == 1))
     
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, groundtruth == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, groundtruth == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN   = np.sum(np.logical_and(predictions == 0, groundtruth ==  1))

    iou =  TP/(TP+FP+FN+0.0000005)

    if np.sum(groundtruth) == 0:
        iou = np.nan

    return  iou


def Accuracy(predictions,groundtruth):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, groundtruth == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, groundtruth == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, groundtruth == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN   = np.sum(np.logical_and(predictions == 0, groundtruth ==  1))

    return  (TP+TN)/(TP+TN+FP+FN+0.0000005)

def confusion_images(predictions,groundtruth):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.logical_and(predictions == 1, groundtruth == 1)

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.logical_and(predictions == 0, groundtruth == 0)
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.logical_and(predictions == 1, groundtruth == 0)
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN   = np.logical_and(predictions == 0, groundtruth ==  1)

    return  TP, TN, FP, FN

def evaluate(pred,gt,img, model_name):

    thresholds = np.round(np.linspace(0.5,0.95,10),2)

    metrics = []
    smax = nn.Softmax2d()

    for num, t in enumerate(thresholds):
        metrics_batch = []
        for p, res in enumerate(pred):
            res = smax(res)
                     
            #compute binary mask for the CME, this is where you can add more processing (removing things behind front...)
            mask = np.zeros(res[0].shape)

            mask[res[0]>t] = 1.0
            #computing metrics
            kapa = Kappa_cohen(mask,gt[p][0]) ## kinda accuracy but for masks with a lot of background and small mask areas
            precision,recall,TP,FP,TN,FN = precision_recall(mask,gt[p][0]) #precision recall at different thresholds for doing a precision recall curve plot
            iou = IoU(mask,gt[p][0]) ## main metric to look at, take the mean over dataset or over thresholds 
            acc = Accuracy(mask,gt[p][0]) #no need to explain
            confusion_matrix = [[TP, FN], [FP, TN]]
            TP, TN, FP, FN = confusion_images(mask,gt[p][0])

            if (np.sum(gt[p][0]) > 0) & (iou > 0.3):

                # print('threshold: ',t)
                # print('precision: ',precision)
                # print('recall: ',recall)
                # print('iou: ',iou)

                hspace = 0
                wspace = 0.01

                fig,ax = plt.subplots(1, figsize=(8,8))
                
                input_data = img[p][0]
                
                TP = np.where(TP == 0, np.nan, TP)
                FP = np.where(FP == 0, np.nan, FP)
                FN = np.where(FN == 0, np.nan, FN)

                cmap_tp = ListedColormap(['#785EF0','red'])
                cmap_fp = ListedColormap(['#DC267F','green'])
                cmap_fn =ListedColormap(['#FE6100','black'])

                al = 0.35
                ax.imshow(input_data, cmap='gray', interpolation='none')
                ax.imshow(TP, alpha=al, cmap=cmap_tp, interpolation='none')
                ax.imshow(FP, alpha=al, cmap=cmap_fp, interpolation='none')
                ax.imshow(FN, alpha=al, cmap=cmap_fn, interpolation='none')
                
                ax.axis("off")
                plt.tight_layout()

                timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                test_path = 'Model_Test/Images/'

                if not os.path.exists(test_path): 
                    os.makedirs(test_path, exist_ok=True) 

                plt.savefig(test_path+model_name+'_t'+'{:.2f}'.format(t)+'_'+timestr+'.png', dpi=300, bbox_inches='tight', pad_inches=0)
                
                #plt.show()                
            #     fig, ax = plt.subplots(1,4, figsize=(8,2), frameon=False)
            #     ax[0].imshow(img[p][0], cmap='gray')
            #     ax[1].imshow(gt[p][0])
            #     ax[2].imshow(res[0])
            #     ax[3].imshow(mask)
            #     ax[0].axis("off")
            #     ax[1].axis("off")
            #     ax[2].axis("off")
            #     ax[3].axis("off")

            #     ax[0].set_aspect("equal")
            #     ax[1].set_aspect("equal")
            #     ax[2].set_aspect("equal")
            #     ax[3].set_aspect("equal")
            #     #plt.subplots_adjust(wspace=wspace, hspace=hspace)
            #     plt.tight_layout()

            #     plt.show()
            metrics_batch.append([kapa,precision,recall,iou,acc])
        
        metrics.append(np.nanmean(metrics_batch, axis=0))

    return metrics


if __name__ == "__main__":
    epoch = sys.argv[1]
    model_summed_aug.test(epoch=epoch)

    