import os
import numpy as np
import glob
import pickle as pkl
from sklearn import metrics

def compute_IOU(bbox_true, bbox_pred, format='xywh'):
    '''
    IOU 계산
    [cx, cy, w, h] or [x1, y1, x2, y2]
    '''
    if format == 'xywh':
        xmin = np.max([bbox_true[0] - bbox_true[2]/2, bbox_pred[0] - bbox_pred[2]/2])
        xmax = np.min([bbox_true[0] + bbox_true[2]/2, bbox_pred[0] + bbox_pred[2]/2])
        ymin = np.max([bbox_true[1] - bbox_true[3]/2, bbox_pred[1] - bbox_pred[3]/2])
        ymax = np.min([bbox_true[1] + bbox_true[3]/2, bbox_pred[1] + bbox_pred[3]/2])
        w_true = bbox_true[2]
        h_true = bbox_true[3]
        w_pred = bbox_pred[2]
        h_pred = bbox_pred[3]
    elif format == 'x1y1x2y2':
        xmin = np.max([bbox_true[0], bbox_pred[0]])
        xmax = np.min([bbox_true[2], bbox_pred[2]])
        ymin = np.max([bbox_true[1], bbox_pred[1]])
        ymax = np.min([bbox_true[3], bbox_pred[3]])
        w_true = bbox_true[2] - bbox_true[0]
        h_true = bbox_true[3] - bbox_true[1]
        w_pred = bbox_pred[2] - bbox_pred[0]
        h_pred = bbox_pred[3] - bbox_pred[1]
    else:
        raise NameError("Unknown format {}".format(format))
    w_inter = np.max([0, xmax - xmin])
    h_inter = np.max([0, ymax - ymin])
    intersection = w_inter * h_inter
    union = (w_true * h_true + w_pred * h_pred) - intersection

    return intersection/union
