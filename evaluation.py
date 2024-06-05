import torch
from torch import nn
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.colors import ListedColormap
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
    gt_neg = np.sum(groundtruth ==  0)
    gt_pos = np.sum(groundtruth ==  1)
    
    sum_total = (np.shape(groundtruth)[0]*np.shape(groundtruth)[1]+np.shape(predictions)[0]*np.shape(predictions)[1])
    observed_accuracy  =   (TP+TN)/sum_total
    expected_accuracy  =   ((gt_pos*(TP+FP))/sum_total + (gt_neg*(TN+FN))/sum_total)/sum_total

    return (observed_accuracy - expected_accuracy)/ (1- expected_accuracy)

def precision_recall(predictions,groundtruth):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, groundtruth == 1))
     
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, groundtruth == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, groundtruth == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(predictions == 0, groundtruth ==  1))

    if np.sum(groundtruth) == 0:
        recall = np.nan
        precision = np.nan
    else:
        precision = TP/ (TP+FP)
        recall    = TP/ (TP+FN)

    return precision,recall

def IoU(predictions,groundtruth):
   # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, groundtruth == 1))
     
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, groundtruth == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, groundtruth == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN   = np.sum(np.logical_and(predictions == 0, groundtruth ==  1))

    if np.sum(groundtruth) == 0:
        iou = np.nan
    else:
        iou =  TP/(TP+FP+FN)

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

    acc = (TP+TN)/(TP+TN+FP+FN)

    return  acc

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

def evaluate_onec_slide(pred, gt, img, model_name, folder_path, data_num, epoch=None, thresh=0.1):

    im_path = 'Model_Metrics/'+folder_path+'/images/'

    if not os.path.exists(im_path): 
        os.makedirs(im_path, exist_ok=True) 

    metrics = []

    plot_num = 0

    for p, res in enumerate(pred):

        gt_win = gt[p].copy()
        mask_pred = res.copy()

        #compute binary mask for the CME, this is where you can add more processing (removing things behind front...)
        
        mask_pred[mask_pred >= thresh] = 1
        mask_pred[mask_pred < thresh] = 0

        gt_win[gt_win >= thresh] = 1
        gt_win[gt_win < thresh] = 0
        
        #computing metrics
        kapa = Kappa_cohen(mask_pred,gt_win) ## kinda accuracy but for masks with a lot of background and small mask areas
        precision,recall = precision_recall(mask_pred,gt_win) #precision recall at different thresholds for doing a precision recall curve plot
        iou = IoU(mask_pred,gt_win) ## main metric to look at, take the mean over dataset or over thresholds 
        acc = Accuracy(mask_pred,gt_win) #no need to explain
        #confusion_matrix = [[TP, FN], [FP, TN]]

        TP, TN, FP, FN = confusion_images(mask_pred,gt_win)
        metrics.append([kapa,precision,recall,iou,acc,epoch])
        
        if (gt_win.any()) and (plot_num <= 5):

            fig,ax = plt.subplots(1, figsize=(4,4))
            
            TP = np.where(TP == 0, np.nan, TP)
            FP = np.where(FP == 0, np.nan, FP)
            FN = np.where(FN == 0, np.nan, FN)

            cmap_tp = ListedColormap(['#785EF0','violet'])
            cmap_fp = ListedColormap(['#DC267F','pink'])
            cmap_fn = ListedColormap(['#FE6100','orange'])

            al = 0.35
            ax.imshow(img[p], cmap='gray', interpolation='none')
            ax.imshow(TP, alpha=al, cmap=cmap_tp, interpolation='none')
            ax.imshow(FP, alpha=al, cmap=cmap_fp, interpolation='none')
            ax.imshow(FN, alpha=al, cmap=cmap_fn, interpolation='none')

            ax.axis("off")
            plt.tight_layout()
            plt.savefig(im_path+model_name+'_test_p'+'{:.0f}'.format(p)+'_'+str(data_num)+'.png', dpi=50, bbox_inches='tight', pad_inches=0)
            plt.close()
            plot_num = plot_num + 1
    
    metrics = np.nanmean(metrics, axis=0)

    return metrics

def test_onec_slide(pred, gt, img, model_name, thresh=0.1):

    im_path = 'Model_Test/'+model_name+'/images/'

    if not os.path.exists(im_path): 
        os.makedirs(im_path, exist_ok=True) 

    metrics = []

    plot_num = 0

    for p, res in enumerate(pred):

        gt_win = gt[p].copy()
        mask_pred = res.copy()

        #compute binary mask for the CME, this is where you can add more processing (removing things behind front...)
        
        mask_pred[mask_pred >= thresh] = 1
        mask_pred[mask_pred < thresh] = 0

        gt_win[gt_win >= thresh] = 1
        gt_win[gt_win < thresh] = 0
        
        #computing metrics
        kapa = Kappa_cohen(mask_pred,gt_win) ## kinda accuracy but for masks with a lot of background and small mask areas
        precision,recall = precision_recall(mask_pred,gt_win) #precision recall at different thresholds for doing a precision recall curve plot
        iou = IoU(mask_pred,gt_win) ## main metric to look at, take the mean over dataset or over thresholds 
        acc = Accuracy(mask_pred,gt_win) #no need to explain
        #confusion_matrix = [[TP, FN], [FP, TN]]

        TP, TN, FP, FN = confusion_images(mask_pred,gt_win)
        metrics.append([kapa,precision,recall,iou,acc])
        
        if (gt_win.any()) and (plot_num <= 15):

            fig,ax = plt.subplots(1, figsize=(4,4))
            
            TP = np.where(TP == 0, np.nan, TP)
            FP = np.where(FP == 0, np.nan, FP)
            FN = np.where(FN == 0, np.nan, FN)

            cmap_tp = ListedColormap(['#785EF0','violet'])
            cmap_fp = ListedColormap(['#DC267F','pink'])
            cmap_fn = ListedColormap(['#FE6100','orange'])

            al = 0.35
            ax.imshow(img[p], cmap='gray', interpolation='none')
            ax.imshow(TP, alpha=al, cmap=cmap_tp, interpolation='none')
            ax.imshow(FP, alpha=al, cmap=cmap_fp, interpolation='none')
            ax.imshow(FN, alpha=al, cmap=cmap_fn, interpolation='none')

            ax.axis("off")
            plt.tight_layout()
            plt.savefig(im_path+'model_test_p'+'{:.0f}'.format(p)+'.png', dpi=50, bbox_inches='tight', pad_inches=0)
            plt.close()
            plot_num = plot_num + 1
    
    metrics = np.nanmean(metrics, axis=0)

    return metrics