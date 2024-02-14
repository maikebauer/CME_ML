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
import model_torch
from matplotlib.colors import ListedColormap
from datetime import datetime
import os
import csv

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

def evaluate(pred, gt, img, model_name, folder_path):

    thresholds = np.round(np.linspace(0.5,0.95,10),2)
    thresholds = np.append(thresholds, 0.99)

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

            if (np.sum(gt[p][0]) > 0) & (t >= 0.8) & (p == 0):

                fig,ax = plt.subplots(1, figsize=(5,5))
                
                input_data = img[p][0]
                
                TP = np.where(TP == 0, np.nan, TP)
                FP = np.where(FP == 0, np.nan, FP)
                FN = np.where(FN == 0, np.nan, FN)

                cmap_tp = ListedColormap(['#785EF0','violet'])
                cmap_fp = ListedColormap(['#DC267F','pink'])
                cmap_fn = ListedColormap(['#FE6100','orange'])

                al = 0.35
                ax.imshow(input_data, cmap='gray', interpolation='none')
                ax.imshow(TP, alpha=al, cmap=cmap_tp, interpolation='none')
                ax.imshow(FP, alpha=al, cmap=cmap_fp, interpolation='none')
                ax.imshow(FN, alpha=al, cmap=cmap_fn, interpolation='none')
                
                ax.axis("off")
                plt.tight_layout()
                
                im_path = 'Model_Metrics/'+folder_path+'/images/'

                if not os.path.exists(im_path): 
                    os.makedirs(im_path, exist_ok=True) 

                plt.savefig(im_path+model_name+'_test_t'+'{:.2f}'.format(t)+'.png', dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close()
                
            metrics_batch.append([kapa,precision,recall,iou,acc])
        
        metrics.append(np.nanmean(metrics_batch, axis=0))

    return metrics


if __name__ == "__main__":
    epoch = sys.argv[1]
    folder_path = sys.argv[2]
    model_torch.test(epoch=epoch, folder_path=folder_path)

    