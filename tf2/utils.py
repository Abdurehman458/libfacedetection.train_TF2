#-*- coding:utf-8 -*-

from tensorflow.python.ops.gen_math_ops import compare_and_bitpack
import torch
import numpy as np
import tensorflow as tf


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    

    return tf.concat((boxes[:, 0:2] - boxes[:, 2:4]/2,     # xmin, ymin
                     boxes[:, 0:2] + boxes[:, 2:4]/2), 1) 


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:4] + boxes[:, 0:2])/2,  # cx, cy
                     boxes[:, 2:4] - boxes[:, 0:2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    # print(box_a.shape)
    # print(box_b.shape)
    boxa1=box_a[:, 2:4]
    boxa1=tf.expand_dims(boxa1,axis=1)
    boxa1=tf.broadcast_to(boxa1,shape=(A,B,2))
    boxb1=box_b[:, 2:4]
    boxb1=tf.expand_dims(boxb1,axis=0)
    boxb1=tf.broadcast_to(boxb1,shape=(A,B,2))
    # print(boxa1.shape,boxb1.shape)
    max_xy = tf.math.minimum(boxa1,boxb1)
    # max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
    #                    box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))

    boxa1=box_a[:, 0:2]
    boxa1=tf.expand_dims(boxa1,axis=1)
    boxa1=tf.broadcast_to(boxa1,shape=(A,B,2))
    boxb1=box_b[:, 0:2]
    boxb1=tf.expand_dims(boxb1,axis=0)
    boxb1=tf.broadcast_to(boxb1,shape=(A,B,2))
    min_xy = tf.math.maximum(boxa1,boxb1)
    # min_xy = torch.max(box_a[:, 0:2].unsqueeze(1).expand(A, B, 2),
    #                    box_b[:, 0:2].unsqueeze(0).expand(A, B, 2))
    inter = tf.clip_by_value((max_xy - min_xy), clip_value_min=0, clip_value_max=tf.float32.max)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a=tf.expand_dims(((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])),axis=1)
    area_b=tf.expand_dims(((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])),axis=0)
    area_a=tf.broadcast_to(area_a,shape=inter.shape)
    area_b=tf.broadcast_to(area_b,shape=inter.shape)
    # area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    # area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # print("inter",inter.shape)
    # print(area_a.shape,area_b.shape)
    # exit()
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, 0:2], b[:, 0:2])
    rb = np.minimum(a[:, np.newaxis, 2:4], b[:, 2:4])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, 0:2], axis=1)
    area_b = np.prod(b[:, 2:4] - b[:, 0:2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, 0:2], b[:, 0:2])
    rb = np.minimum(a[:, np.newaxis, 2:4], b[:, 2:4])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, 0:2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # print(type(overlaps),overlaps.shape)
    # exit()
    best_prior_overlap = tf.reduce_max(overlaps,axis=1,keepdims=True)
    best_prior_idx = tf.argmax(overlaps,axis=1)
    best_prior_idx = tf.expand_dims(best_prior_idx,axis=-1)
    # print(best_prior_overlap,best_prior_idx)
    # print(x.shape,y.shape)
    # best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    # print(best_prior_overlap,best_prior_idx)
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    # print(valid_gt_idx)
    best_prior_idx_filter = tf.boolean_mask(best_prior_idx, valid_gt_idx)
    # print(best_prior_idx_filter)
    # best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    # print(loc_t)
    if best_prior_idx_filter.shape[0] is not None and best_prior_idx_filter.shape[0] <= 0:
        # loc_t[idx] = 0
        # conf_t[idx] = 0
        # ones = tf.ones((16,5875,4))
        # print(tf.reshape(tf.where(ones[idx]),shape=(idx,5875,4)))
        # ones = tf.tensor_scatter_nd_update(ones,[[idx]],[tf.zeros((5875,4))])
        # loc_t = tf.tensor_scatter_nd_update(loc_t,tf.cast(loc_t[idx,:],dtype="int64"),tf.zeros((idx,5875,4)))
        # print(ones)
        loc_t = tf.tensor_scatter_nd_update(loc_t,[[idx]],[tf.zeros((5875,4))])
        conf_t = tf.tensor_scatter_nd_update(conf_t,[[idx]],[tf.zeros((5875))])
        # print(loc_t)
        # exit()
        # print("exception!!!!!")
        return tf.squeeze(tf.zeros((1, priors.shape[0])),axis=0),loc_t,conf_t

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap = tf.reduce_max(overlaps,axis=0,keepdims=True)
    best_truth_overlap = tf.squeeze(best_truth_overlap,axis=0)
    best_truth_idx = tf.argmax(overlaps,axis=0)
    
    best_prior_idx=tf.squeeze(best_prior_idx,axis=1)
    best_prior_idx_filter=tf.squeeze(best_prior_idx_filter,axis=1)
    best_prior_overlap=tf.squeeze(best_prior_overlap,axis=1)
    # print(best_truth_overlap)
    # print("filter",best_prior_idx_filter)
    # print(tf.where(best_truth_overlap==2))
    # for x in best_prior_idx_filter:
    #     best_truth_overlap = tf.tensor_scatter_nd_update(best_truth_overlap,[[x]],[2])
    # print("IDX",type(best_truth_idx))
    # print("update",tf.ones((best_prior_idx_filter.shape[0]))*2)
    best_truth_overlap = tf.tensor_scatter_nd_update(best_truth_overlap,tf.expand_dims(best_prior_idx_filter,axis=-1),tf.ones((best_prior_idx_filter.shape[0]))*2)
    # print(best_truth_overlap)
    # print(tf.where(best_truth_overlap==2))
    # exit()

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.shape[0]):
        # best_truth_idx = tf.tensor_scatter_nd_update(best_truth_idx, [[best_prior_idx[j].numpy()]],[j])
        best_truth_idx = tf.tensor_scatter_nd_update(best_truth_idx,[[best_prior_idx[j]]],[j])
    matches = tf.gather(truths,best_truth_idx)
    # matches = truths[best_truth_idx]          # Shape: [num_priors,14]
    conf = tf.gather(labels,best_truth_idx)
    # print("conf",conf)
    # print(tf.where(conf==0))
    # exit()
    # conf = labels[best_truth_idx]          # Shape: [num_priors]
    # conf_np=conf.numpy()
    # conf_np[best_truth_overlap < threshold] = 0 
    # print(conf_np,conf_np.shape)
    # print("overlap",best_truth_overlap)
    # print(tf.where(best_truth_overlap < threshold))


    compare = (tf.where(best_truth_overlap < threshold))
    # print("conf",conf)
    # print("compare",compare)
    # print("update",tf.zeros((compare.shape[0])))
    # # for i in compare:
    conf = tf.tensor_scatter_nd_update(conf,compare,tf.zeros((compare.shape[0])))
    # print(conf)
    # exit()
    # print(tf.where(conf>0))
    # confnp =  
    # exit()
    # for i in range (best_truth_overlap.shape[0]):
    #     if best_truth_overlap[i]< threshold:
    #         conf_np[i]=0

    # conf = tf.convert_to_tensor(conf_np)
    # conf = conf_np
    # conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    # loc_t[idx] = loc    # [num_priors,14] encoded offsets to learn
    loc_t = tf.tensor_scatter_nd_update(loc_t,[[idx]],[loc])
    # print(loc_t)
    # loc_tnp=loc_t.numpy()
    # loc_tnp[idx]=loc
    # loc_t = tf.convert_to_tensor(loc_tnp)
    # conf_t[idx] = conf  # [num_priors] top class label for each prior
    conf_t = tf.tensor_scatter_nd_update(conf_t,[[idx]],[conf])
    # print(tf.where(conf_t == 1))
    # conf_tnp=conf_t.numpy()
    # conf_tnp[idx]=conf
    # conf_t = tf.convert_to_tensor(conf_tnp)

    return best_truth_overlap,loc_t,conf_t


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes and landmarks (tensor), Shape: [num_priors, 14]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, 0:2] + matched[:, 2:4])/2 - priors[:, 0:2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:4])
    # match wh / prior wh
    g_wh = (matched[:, 2:4] - matched[:, 0:2]) / priors[:, 2:4]
    g_wh = tf.math.log(g_wh) / variances[1]

    # landmarks
    # g_xy1 = (matched[:, 4:6] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    # g_xy2 = (matched[:, 6:8] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    # g_xy3 = (matched[:, 8:10] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    # g_xy4 = (matched[:, 10:12] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    # g_xy5 = (matched[:, 12:14] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])

    # return target for loss
    # return torch.cat([g_cxcy, g_wh, g_xy1, g_xy2, g_xy3, g_xy4, g_xy5], 1)  # [num_priors,14]
    return tf.concat([g_cxcy, g_wh], 1)  # [num_priors,14]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
        priors[:, 0:2] + loc[:, 0:2] * variances[0] * priors[:, 2:4],
        priors[:, 2:4] * np.exp(loc[:, 2:4] * variances[1])),1)
        # priors[:, 0:2] + loc[:, 4:6] * variances[0] * priors[:, 2:4],
        # priors[:, 0:2] + loc[:, 6:8] * variances[0] * priors[:, 2:4],
        # priors[:, 0:2] + loc[:, 8:10] * variances[0] * priors[:, 2:4],
        # priors[:, 0:2] + loc[:, 10:12] * variances[0] * priors[:, 2:4],
        # priors[:, 0:2] + loc[:, 12:14] * variances[0] * priors[:, 2:4]), 1)
    boxes[:, 0:2] -= boxes[:, 2:4] / 2
    boxes[:, 2:4] += boxes[:, 0:2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    # x_max = x.data.max()
    x_max = tf.reduce_max(x)
    # print("xmax",x_max)
    # print("sum",tf.math.reduce_sum(tf.exp(x-x_max), 1, keepdims=True))
    # print("log",tf.math.log(tf.math.reduce_sum(tf.exp(x-x_max), 1, keepdims=True)))
    return tf.math.log(tf.math.reduce_sum(tf.exp(x-x_max), 1, keepdims=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

