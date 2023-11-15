import numpy as np 
import matplotlib.pyplot as plt
import torch
import sys

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

    return  TP/(TP+FP+FN+0.0000005)


def Accuracy(predictions,groundtruth):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, groundtruth == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, groundtruth == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, groundtruth == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN   = np.sum(np.logical_and(predictions == 0, groundtruth ==  1))

    return  (TP+TN)/groundtruth.shape[0]



def evaluate(pred,gt):


    thresholds = [0.5,0.55,0.60,0,65,0.7,0.75,0.8,0.85,0.90,0.95]
    ## check if there is a second CME

    metrics = []

    for p, res in enumerate(pred):
        for t in thresholds:

            #compute binary mask for the CME, this is where you can add more processing (removing things behind front...)
            mask = np.zeros(res.shape)

            mask[res>t] = 1.0
            #computing metrics

            kapa = Kappa_cohen(mask,gt[p]) ## kinda accuracy but for masks with a lot of background and small mask areas
            precision,recall,TP,FP,TN,FN = precision_recall(mask,gt[p]) #precision recall at different thresholds for doing a precision recall curve plot
            iou = IoU(mask,gt[p]) ## main metric to look at, take the mean over dataset or over thresholds 
            acc = Accuracy(mask,gt[p]) #no need to explain

            # print('KAPA: ', kapa)
            # print('Precision: ', precision)
            # print('Recall: ', recall)
            # print('IOU: ', iou)
            # print('ACC: ', acc)
            # print('------------------------------')
            
            metrics.append([kapa,precision,recall,iou,acc])


    return metrics





    