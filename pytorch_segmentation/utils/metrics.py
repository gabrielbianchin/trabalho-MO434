import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(output, target, num_class):
    _, predict = torch.max(output, 1)
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_classes):
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]



def pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output == target) * (target > 0))
    return pixel_correct, pixel_labeled

def inter_over_union(output, target, num_class):
    output = np.asarray(output) + 1
    target = np.asarray(target) + 1
    output = output * (target > 0)

    intersection = output * (output == target)
    area_inter, _ = np.histogram(intersection, bins=num_class, range=(1, num_class))
    area_pred, _ = np.histogram(output, bins=num_class, range=(1, num_class))
    area_lab, _ = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union

def metrics(y_true, y_pred, channel=None):
    epsilon = 1e-6
    pred = torch.argmax(y_pred, 1)
    pred = (pred==channel).byte()
    pred_inv = (~pred)
    gt = (y_true == channel).byte()
    gt_inv = (~gt)
    tp = (pred & gt).sum().float()
    tn = (pred_inv & gt_inv).sum().float()
    fp = (pred & gt_inv).sum().float()
    fn = (pred_inv & gt).sum().float()
    accuracy  = (tp + tn)/(tp + tn + fp + fn + epsilon)
    precision  = tp/(tp + fp + epsilon)
    recall = tp/(tp + fn + epsilon)
    f1 = 2*(precision*recall)/(precision + recall + epsilon)
    return torch.Tensor([accuracy, precision, recall, f1])

def evaluate_dataloader(model, dataloader):
    metrics_accumulated_soil = torch.zeros((4)).float()
    metrics_accumulated_crop = torch.zeros((4)).float()
    metrics_accumulated_weed = torch.zeros((4)).float()
    start = time.time()

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            y_pred = model(X)
            metrics_accumulated_soil += metrics(y, y_pred, 0)
            metrics_accumulated_crop += metrics(y, y_pred, 1)
            metrics_accumulated_weed += metrics(y, y_pred, 2)

    end = time.time()
    metrics_mean_soil = metrics_accumulated_soil/len(dataloader)
    metrics_mean_crop = metrics_accumulated_crop/len(dataloader)
    metrics_mean_weed = metrics_accumulated_weed/len(dataloader)

    fps = len(dataloader.dataset)/(end - start)
    print("(Soil) Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f"%(metrics_mean_soil[0],
                                                                            metrics_mean_soil[1],
                                                                            metrics_mean_soil[2],
                                                                            metrics_mean_soil[3]))
    
    print("(Crop) Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f"%(metrics_mean_crop[0],
                                                                            metrics_mean_crop[1],
                                                                            metrics_mean_crop[2],
                                                                            metrics_mean_crop[3]))
    
    print("(Weed) Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f"%(metrics_mean_weed[0],
                                                                            metrics_mean_weed[1],
                                                                            metrics_mean_weed[2],
                                                                            metrics_mean_weed[3]))
    print("FPS: %i"%fps)