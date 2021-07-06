from numpy.core.numeric import indices
from tensorflow.python.keras.backend import dtype, shape
from tensorflow.python.ops.gen_math_ops import logical_or
from utils import match, log_sum_exp
from eiou import eiou_loss
import tensorflow as tf
import numpy as np

class MultiBoxLoss(object):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label,
                 neg_mining, neg_pos, neg_overlap, encode_target, rect_only):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.rect_only = rect_only
        self.smooth_point = 0.2

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,14)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,15] (last idx is the label).
        """

        loc_data, conf_data, iou_data = predictions
        # print(loc_data)
        # print(conf_data)
        # print(iou_data)
        # exit()
        priors = priors
        num = loc_data.shape[0]
        num_priors = (priors.shape[0])

        # match priors (default boxes) and ground truth boxes
        # loc_t = torch.Tensor(num, num_priors, 4)
        # conf_t = torch.LongTensor(num, num_priors)
        # iou_t = torch.Tensor(num, num_priors)
        # print(iou_t.shape)

        # loc_t = tf.zeros(shape=(num, num_priors, 4),dtype="float32")
        # conf_t = tf.zeros(shape=(num, num_priors),dtype="float32")
        # iou_t = tf.zeros(shape=(num, num_priors),dtype="float32")
        loc_t = tf.zeros(shape=(num, num_priors, 4),dtype="float32")
        conf_t = tf.zeros(shape=(num, num_priors),dtype="float32")
        iou_t = tf.zeros(shape=(num, num_priors),dtype="float32")
        
        for idx in range(num):
            truths = targets[idx][:,0:4]
            labels = targets[idx][:, -1]
            # print("truths",truths)
            # print("labels",labels)
            defaults = priors
            # exit()
            iout,loc_t,conf_t = match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            # print("iou_t",iou_t)
            # print("idx",idx)
            # print("iout",iout)
            iou_t = tf.tensor_scatter_nd_update(iou_t,[[idx]],[iout])
            # iou_t[idx],loc_t,conf_t = match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            # print(iou_t[idx])
            # exit()
        iou_t = tf.reshape(iou_t,(num, num_priors, 1))
        # print(iou_t)
        # exit()
        # a =0
        # for x in range(iou_t.shape[1]):
        #     if iou_t[:,x,:].numpy() > 0:
        #         a +=1
        #         if iou_t[:,x,:].numpy() >= 2:
        #             print(iou_t[:,x,:].numpy(),x)
        # print(a)
        # print(tf.where(iou_t == 2))
        # exit()
        pos = tf.math.greater(conf_t,0)
        # print(tf.where(conf_t == 1))
        # print(tf.where(pos == True))
        # exit()
        # print(conf_t)
        # pos = tf.convert_to_tensor(pos)
        # Localization Loss
        # Shape: [batch,num_priors,4]
        # pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # print("pos",pos)
        # print(pos.shape)
        
        pos_idx = tf.expand_dims(pos,axis=pos.ndim)
        pos_idx = tf.broadcast_to(pos_idx,shape=loc_data.shape)
        
        # loc_t = tf.convert_to_tensor(loc_t)
        # print(loc_t)
        # exit()
        # print(loc_data.shape,loc_t.shape,pos_idx.shape)
        # print(loc_data,loc_t,pos_idx)
        # loc_p = loc_data[pos_idx].view(-1, 4)
        loc_p = tf.reshape(loc_data[pos_idx],[-1,4])
        
        loc_t = tf.reshape(loc_t[pos_idx],[-1,4])
        # print(loc_t)
        # exit()
        # print(loc_t,loc_t.shape)
        
        loss_l = eiou_loss(loc_p[:, 0:4], loc_t[:, 0:4], variance=self.variance, smooth_point=self.smooth_point, reduction='sum')
        # print(loss_l)
        # exit()
        # loss_lm = F.smooth_l1_loss(loc_p[:, 4:14], loc_t[:, 4:14], reduction='sum')

        # IoU diff
        # pos_idx_ = pos.unsqueeze(pos.dim()).expand_as(iou_data)
        # iou_p = iou_data[pos_idx_].view(-1, 1)
        # iou_t = iou_t[pos_idx_].view(-1, 1)
        pos_idx_ = tf.expand_dims(pos,axis=pos.ndim)
        pos_idx_ = tf.broadcast_to(pos_idx_,shape=iou_data.shape)
        iou_p = tf.reshape(iou_data[pos_idx_],[-1,1])
        iou_t = tf.reshape(iou_t[pos_idx_],[-1,1])
        # loss_iou = F.smooth_l1_loss(iou_p, iou_t, reduction='sum')
        huber_loss= tf.keras.losses.Huber(reduction='sum')
        loss_iou = huber_loss(iou_t, iou_p)

        # Compute max conf across batch for hard negative mining
        # batch_conf = conf_data.view(-1, self.num_classes)
        batch_conf = tf.reshape(conf_data,shape=(-1,self.num_classes))
        conf_t_re=tf.reshape(conf_t,[-1,1])
        # conf_t_re=np.reshape(conf_t,[-1,1])
        # print(conf_t_re.shape)
        # print("tf.shape=",tf.shape(conf_t_re)[0])
        # print("tf.range=",tf.range(tf.shape(conf_t_re)[0]))
        # print(conf_t_re[:,0])

        # idx = tf.stack([tf.range(tf.shape(conf_t_re)[0]),conf_t_re[:,0]],axis=1)
        idx = tf.stack([tf.range(tf.shape(conf_t_re)[0]),tf.cast(conf_t_re[:,0],dtype="int32")],axis=1)
        b = tf.gather_nd(batch_conf,idx)
        b = tf.expand_dims(b,axis=-1)
        # print(b)
        # exit()
        loss_c = log_sum_exp(batch_conf) - b
        # print(loss_c)
        # exit()
        # Hard Negative Mining
        # loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        pos_r = tf.reshape(pos,[-1, 1])
        # print("pos_r",pos_r)
        # print(loss_c)
        # print(tf.gather())
        # exit()
        # loss_c_np = loss_c.numpy()
        # loss_c_np[pos_r] = 0
        # print(loss_c.shape,pos_r.shape)
        # print(tf.zeros((pos_r.shape[0],1)))
        loss_c = tf.tensor_scatter_nd_update(loss_c,tf.cast(pos_r,dtype="int32"),tf.zeros((pos_r.shape[0],1)))
        # loss_c = tf.convert_to_tensor(loss_c_np)

        # loss_c = loss_c.view(num, -1)
        loss_c = tf.reshape(loss_c,shape=(num,-1))
        # print(loss_c)
        # exit()
        # _, loss_idx = loss_c.sort(1, descending=True)
        # _, idx_rank = loss_idx.sort(1)
        loss_idx = tf.argsort(loss_c,1,direction="DESCENDING")
        idx_rank = tf.argsort(loss_idx,1)
        num_pos = tf.math.reduce_sum(tf.cast(pos,tf.float32),1,keepdims=True)
        # num_pos = pos.long().sum(1, keepdim=True)
        # num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        num_neg = tf.clip_by_value(self.negpos_ratio*num_pos,clip_value_min=tf.float32.min, clip_value_max=pos.shape[1]-1)
        # neg = idx_rank < num_neg.expand_as(idx_rank)
        neg = idx_rank < tf.broadcast_to(tf.cast(num_neg,tf.int32),shape=idx_rank.shape)

        # Confidence Loss Including Positive and Negative Examples

        # pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        pos_idx = tf.expand_dims(pos,2)
        pos_idx = tf.broadcast_to(pos_idx,shape=conf_data.shape)

        # neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        neg_idx = tf.expand_dims(neg,2)
        neg_idx = tf.broadcast_to(neg_idx,shape=conf_data.shape)
        

        conf_p = conf_data[tf.math.greater(tf.cast((tf.math.logical_or(pos_idx,neg_idx)),tf.float32),0)]
        conf_p = tf.reshape(conf_p,shape=(-1,self.num_classes))
        
        # targets_weighted = conf_t[(pos+neg).gt(0)]
        targets_weighted = conf_t[tf.math.greater(tf.cast((tf.math.logical_or(pos,neg)),tf.float32),0)]
        # targets_weighted = conf_t[np.greater(np.logical_or(pos,neg),0)]
        # targets_weighted = conf_t[np.greater(np.logical_or(pos,neg),0)]
        # print(conf_p.shape,targets_weighted.shape)
        # loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        scc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')
        loss_c = scc(targets_weighted,conf_p)

        # Sum of losses
        # N = max(num_pos.data.sum().float(), 1)
        N = tf.math.reduce_sum(num_pos)
        loss_l /= N
        # loss_lm /= N
        loss_c /= N
        loss_iou /= N
        # return loss_l, loss_lm, loss_c, loss_iou
        return loss_l, loss_c, loss_iou