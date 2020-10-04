import numpy as np
# import cv2
import pickle as pkl
import os
import glob
import copy
import pickle as pkl

def bbox_normalize(bbox,W=1280,H=640):
    '''
    bbox값을 0~1사이로 normalize
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[:,0] /= W
    new_bbox[:,1] /= H
    new_bbox[:,2] /= W
    new_bbox[:,3] /= H

    return new_bbox

def bbox_denormalize(bbox,W=1280,H=640):
    '''
    0~1사이의 bbox값을 0~W or H로 denormalize
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[..., 0] *= W
    new_bbox[..., 1] *= H
    new_bbox[..., 2] *= W
    new_bbox[..., 3] *= H

    return new_bbox

# FLow loading code adapted from:
# http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

def load_flow(flow_folder):
    '''
    비디오 key가 주어졌을 때, 해당하는 flow 파일 load하기
    '''
    flow_files = sorted(glob.glob(flow_folder + '*.flo'))
    flows = []
    for file in flow_files:
        flow = read_flo(file)
        flows.append(flow)
    return flows

TAG_FLOAT = 202021.25

def roi_pooling_opencv(boxes,image,size=[5,5]):
    """
    box에서 이미지를 crop하고 bilinear interpolation을 이용하여 resize
    Params:
        image: Input image of shape (1, image_height, image_width, n_channels).
                The shape can be bigger or smaller than tehe
        boxes: ROI of shape (num_boxes, 4) in range of [0,1]
                each row [cx, cy, w, h]
        size: Fixed size [h, w], e.g. [7, 7], for the output slices.
        W, H: width and height or original image
    :Returns:
        4D Tensor (number of regions, slice_height, slice_width, channels)
    """

    w = image.shape[2]
    h = image.shape[1]
    n_channels = image.shape[3]

    xmin = boxes[:,0]-boxes[:,2]/2
    xmax = boxes[:,0]+boxes[:,2]/2
    ymin = boxes[:,1]-boxes[:,3]/2
    ymax = boxes[:,1]+boxes[:,3]/2


    ymin = np.max([0, int(h * ymin)])
    ymax = np.min([h, int(h * ymax)])

    xmin = np.max([0, int(w * xmin)])
    xmax = np.min([w, int(w * xmax)])


    size = (size[0], size[1])
    return np.expand_dims(cv2.resize(image[0,ymin:ymax, xmin:xmax,:], size), axis=0)

    #     # print(boxes)
    #     # raise ValueError('ymin:%d, ymax:%d, xmin:%d, xmax:%d'%(ymin, ymax, xmin, xmax))
        # print('ymin:%d, ymax:%d, xmin:%d, xmax:%d'%(ymin, ymax, xmin, xmax))
    #     # print("boxes: ",boxes)
    #     size = [size[0],size[1], n_channels]
    #     return None #np.expand_dims(np.zeros(size), axis=0)
