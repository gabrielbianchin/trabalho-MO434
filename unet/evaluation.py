import os
import numpy as np
import keras.backend as K
import re

def metric_generator(metric, channel=None, threshold=0.5):
    
    def metrics(y_true, y_pred):
        y_true_flatten = K.flatten(y_true[..., channel])
        y_pred_flatten = K.flatten(y_pred[..., channel])
        y_true_mask = K.cast(y_true_flatten >= threshold,'float64')
        y_true_bg = K.cast(y_true_flatten < threshold,'float64')
        y_pred_mask = K.cast(y_pred_flatten >= threshold,'float64')
        y_pred_bg = K.cast(y_pred_flatten < threshold,'float64')

        tp = K.sum(y_true_mask*y_pred_mask)
        fp = K.sum(y_true_bg*y_pred_mask)
        tn = K.sum(y_true_bg*y_pred_bg)
        fn = K.sum(y_true_mask*y_pred_bg)

        if metric == 'accuracy':
            accuracy = (tp+tn)/K.clip(tp+fp+tn+fn, K.epsilon(), None)
            return accuracy
        elif metric == 'precision':
            precision = tp/K.clip(tp+fp, K.epsilon(), None)
            return precision
        elif metric == 'recall':
            recall = tp/K.clip(tp+fn, K.epsilon(), None)
            return recall
        elif metric == 'iou':
            iou = tp/K.clip(tp+fp+fn, K.epsilon(), None)
            return iou
        elif metric == 'f1':
            precision = tp/K.clip(tp+fp, K.epsilon(), None)
            recall = tp/K.clip(tp+fn, K.epsilon(), None)
            f1 = (2*precision*recall)/K.clip(precision+recall, K.epsilon(), None)
            return f1
        
    return metrics


def define_metrics():
    total_acc = metric_generator('accuracy')
    total_acc.__name__ = 'total_acc'

    total_precision = metric_generator('precision')
    total_precision.__name__ = 'total_precision'

    total_recall = metric_generator('recall')
    total_recall.__name__ = 'total_recall'

    total_iou = metric_generator('iou')
    total_iou.__name__ = 'total_iou'

    total_f1 = metric_generator('f1')
    total_f1.__name__ = 'total_f1'

    crop_acc = metric_generator('accuracy', 0)
    crop_acc.__name__ = 'crop_acc'

    crop_precision = metric_generator('precision', 0)
    crop_precision.__name__ = 'crop_precision'

    crop_recall = metric_generator('recall', 0)
    crop_recall.__name__ = 'crop_recall'

    crop_iou = metric_generator('iou', 0)
    crop_iou.__name__ = 'crop_iou'

    crop_f1 = metric_generator('f1', 0)
    crop_f1.__name__ = 'crop_f1'

    weed_acc = metric_generator('accuracy', 1)
    weed_acc.__name__ = 'weed_acc'

    weed_precision = metric_generator('precision', 1)
    weed_precision.__name__ = 'weed_precision'

    weed_recall = metric_generator('recall', 1)
    weed_recall.__name__ = 'weed_recall'

    weed_iou = metric_generator('iou', 1)
    weed_iou.__name__ = 'weed_iou'

    weed_f1 = metric_generator('f1', 1)
    weed_f1.__name__ = 'weed_f1'

    metrics_list = [total_acc,
                total_precision,
                total_recall,
                total_iou,
                total_f1,
                crop_acc,
                crop_precision,
                crop_recall,
                crop_iou,
                crop_f1,
                weed_acc,
                weed_precision,
                weed_recall,
                weed_iou,
                weed_f1]

    return metrics_list