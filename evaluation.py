import torch
from torch import nn
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.colors import ListedColormap
import os

def Kappa_cohen(TP, FP, FN, TN, gt_shape, pred_shape):

    gt_neg = TN+FP
    gt_pos = TP+FN
    
    sum_total = (gt_shape[0]*gt_shape[1]+pred_shape[0]*pred_shape[1])
    observed_accuracy  =   (TP+TN)/sum_total
    expected_accuracy  =   ((gt_pos*(TP+FP))/sum_total + (gt_neg*(TN+FN))/sum_total)/sum_total

    return (observed_accuracy - expected_accuracy)/ (1- expected_accuracy)

def precision_recall(TP, FP, FN):

    if (TP+FN == 0):
        recall = np.nan
    else:
        recall    = TP/ (TP+FN)
        
    if (TP+FP == 0):
        precision = np.nan
    else:
        precision = TP/ (TP+FP)

    return precision,recall

def IoU(TP, FP, FN):

    if (TP+FP+FN == 0):
        iou = np.nan
    else:
        iou =  TP/(TP+FP+FN)

    return  iou

def dice(TP, FP, FN):

    if (TP+FP+FN == 0):
        dsc = np.nan
    else:
        dsc = 2*TP/(2*TP+FP+FN)
    
    return dsc

def Accuracy(TP, FP, FN, TN):

    acc = (TP+TN)/(TP+TN+FP+FN)

    return acc

def confusion_images(predictions,groundtruth):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.logical_and(predictions == 1, groundtruth == 1)

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.logical_and(predictions == 0, groundtruth == 0)
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.logical_and(predictions == 1, groundtruth == 0)
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN   = np.logical_and(predictions == 0, groundtruth ==  1)

    return  TP, FP, FN, TN

def confusion_metrics(predictions,groundtruth):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, groundtruth == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, groundtruth == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, groundtruth == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN   = np.sum(np.logical_and(predictions == 0, groundtruth ==  1))

    return  TP, FP, FN, TN

def evaluate_basic(pred, gt, img, model_name, folder_path, data_num, epoch=None):


    #thresholds = np.round(np.arange(0.1,0.99, 0.1),2)
    #thresholds = np.append(thresholds, 0.95)

    smax = nn.Softmax2d()

    im_path = 'Model_Metrics/'+folder_path+'/images/'

    if not os.path.exists(im_path): 
        os.makedirs(im_path, exist_ok=True) 

    #for t in thresholds:
    metrics_batch = []
    metrics_win = []

    for p, res in enumerate(pred):
        for k in range(res.shape[1]):
            #res = smax(res)

            res_win = torch.argmax(res[:,k,:,:], axis=0)
            gt_win = gt[p,1,k,:,:].copy()
            #compute binary mask for the CME, this is where you can add more processing (removing things behind front...)

            mask = torch.clone(res_win)
            #computing metrics
            kapa = Kappa_cohen(mask,gt_win) ## kinda accuracy but for masks with a lot of background and small mask areas
            precision,recall,TP,FP,TN,FN = precision_recall(mask,gt_win) #precision recall at different thresholds for doing a precision recall curve plot
            iou = IoU(mask,gt_win) ## main metric to look at, take the mean over dataset or over thresholds 
            acc = Accuracy(mask,gt_win) #no need to explain
            #confusion_matrix = [[TP, FN], [FP, TN]]

            TP, TN, FP, FN = confusion_images(mask,gt_win)
            metrics_win.append([kapa,precision,recall,iou,acc,epoch])

            if (p == 0) & (k == 0):
                
                fig,ax = plt.subplots(1, figsize=(5,5))
                
                input_data = img[p,0,k,:,:]
                
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

                plt.savefig(im_path+model_name+'_test_p'+'{:.0f}'.format(data_num)+'.png', dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close()
                    
        metrics_batch.append(np.nanmean(metrics_win, axis=0))    
            
    metrics = np.nanmean(metrics_batch, axis=0)

    return metrics

def evaluate_onec_slide(pred, gt, thresh=[0.5]):

    # metrics = []
    metrics_confusion = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    pred = pred.reshape((pred.shape[0]*pred.shape[2],pred.shape[3],pred.shape[4]))
    gt = gt.reshape((gt.shape[0]*gt.shape[2],gt.shape[3],gt.shape[4]))

    for t in thresh:
        # metrics_thresh = []
        metrics_confusion_thresh = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for p, res in enumerate(pred):
            gt_win = gt[p].copy()
            mask_pred = res.copy()

            gt_win[gt_win >= t] = 1
            gt_win[gt_win < t] = 0
            
            mask_pred[mask_pred >= t] = 1
            mask_pred[mask_pred < t] = 0

            TP, FP, FN, TN = confusion_metrics(mask_pred,gt_win)

            # #computing metrics
            # kapa = Kappa_cohen(TP, FP, FN, TN, mask_pred.shape, gt_win.shape) ## kinda accuracy but for masks with a lot of background and small mask areas
            # precision,recall = precision_recall(TP, FP, FN) #precision recall at different thresholds for doing a precision recall curve plot
            # iou = IoU(TP, FP, FN) ## main metric to look at, take the mean over dataset or over thresholds 
            # acc = Accuracy(TP, FP, FN, TN) #no need to explain
            # #confusion_matrix = [[TP, FN], [FP, TN]]

            metrics_confusion_thresh['TP'] += TP
            metrics_confusion_thresh['FP'] += FP
            metrics_confusion_thresh['FN'] += FN
            metrics_confusion_thresh['TN'] += TN

            # metrics_thresh.append([kapa,precision,recall,iou,acc])

        metrics_confusion['TP'].append(metrics_confusion_thresh['TP'])
        metrics_confusion['FP'].append(metrics_confusion_thresh['FP'])
        metrics_confusion['FN'].append(metrics_confusion_thresh['FN'])
        metrics_confusion['TN'].append(metrics_confusion_thresh['TN'])

        # metrics.append(np.nanmean(metrics_thresh, axis=0))

    metrics_confusion = {key: np.nanmean(value) for key, value in metrics_confusion.items()}
    # metrics = np.nanmean(metrics, axis=0)

    return metrics_confusion #, metrics