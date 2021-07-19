import os
from tensorflow.python.keras.optimizer_v2 import adam

from torch.autograd import grad

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.optimizers import SGD, Adam
import argparse
import time
import datetime
import math
import numpy as np
from config import cfg
from multibox_loss import MultiBoxLoss
from prior_box import PriorBox
from yufacedetectnet import YuFaceDetectNet
from yufacedetectnet1 import YuFaceDetectNet1
from data import FaceRectLMDataset, detection_collate
import torch.utils.data as data
import torch
import cv2

img_dim = 320 # only 1024 is supported
rgb_mean =  (0,0,0) #(104, 117, 123) # bgr order
num_classes = 2
batch_size = 16
resume_epoch = 0
max_epoch = 500

lambda_bbox = 1
lambda_iouhead = 1
training_face_rect_dir = "/home/arm/Downloads/liface/libfacedetection.train_TF2/data/WIDER_FACE_rect"


"""#Functional Model"""

# net = YuFaceDetectNet1('train', (320,320,3)) 
# net.load_weights("checkpoints/adam_net.h5")

"""#Subclassed Model"""

net = YuFaceDetectNet('train', (320,320,3))
# net.load_weights("subclass/weights")

net.build((16,320,320,3))
# net.summary()
# print(len(net.trainable_variables))
# exit()
# print("init done!!!")
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=0.0001,
      decay_steps=400000,
      end_learning_rate=0.000001)

# optimizer = SGD(learning_rate=learning_rate_fn)
optimizer = Adam(learning_rate=learning_rate_fn)

criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 3, 0.35, False, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
priors=priorbox.forward()

def train():

    print('Loading Dataset...')
    dataset_rect = FaceRectLMDataset(training_face_rect_dir, img_dim, rgb_mean)
    # img = dataset_rect[0]
    # print(type(dataset_rect))

    for epoch in range(resume_epoch, max_epoch):
        dataset = dataset_rect
        with_landmark = False
        # if with_landmark:
        #     dataset = dataset_landmark

        train_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=detection_collate,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
        )

        #for computing average losses in this epoch
        loss_l_epoch = []
        loss_lm_epoch = []
        loss_c_epoch = []
        loss_iou_epoch = []
        loss_epoch = []

        # the start time
        load_t0 = time.time()
        num_iter_in_epoch = len(train_loader)

        for iter_idx, one_batch_data in enumerate(train_loader):
            images, targets = one_batch_data
            targets1=[]
            images=tf.convert_to_tensor(images)

            for i in range (len(targets)):
                tar=tf.convert_to_tensor(targets[i])
                targets1.append(tar)

            with tf.GradientTape() as tape:
                out = net(images)
                # print(out)
                # exit()
                loss_l, loss_c, loss_iou = criterion.forward(out, priors, targets1)
                loss = lambda_bbox * loss_l + loss_c + lambda_iouhead * loss_iou
                # print(loss)
            gradients=tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients,net.trainable_variables))
            
            # loss, loss_l, loss_c, loss_iou = train_step(images,net,targets1,criterion,optimizer)
            lr=0
            loss_lm=0
            # put losses to lists to average for printing
            loss_l_epoch.append(loss_l)
            loss_lm_epoch.append(loss_lm)
            loss_c_epoch.append(loss_c)
            loss_iou_epoch.append(loss_iou)
            loss_epoch.append(loss)

            # print loss
            if ( iter_idx % 20 == 0 or iter_idx == num_iter_in_epoch - 1):
                print('LM:{} || Epoch:{}/{} || iter: {}/{} || L: {:.2f}({:.2f}) IOU: {:.2f}({:.2f}) LM: {:.2f}({:.2f}) C: {:.2f}({:.2f}) All: {:.2f}({:.2f}) || LR: {:.8f}'.format(
                    with_landmark, epoch, max_epoch, iter_idx, num_iter_in_epoch, 
                    loss_l, np.mean(loss_l_epoch),
                    loss_iou, np.mean(loss_iou_epoch),
                    loss_lm, np.mean(loss_lm_epoch),
                    loss_c, np.mean(loss_c_epoch),
                    loss,  np.mean(loss_epoch), optimizer._decayed_lr('float32').numpy()))
                # print(optimizer._decayed_lr('float32').numpy())
            # exit()
        if (epoch % 5 == 0 and epoch > 0) :
            # net.save_weights("checkpoints/adam_net.h5")
            net.save_weights('subclass/weights',overwrite=True,save_format='tf')
        #the end time
        load_t1 = time.time()
        epoch_time = (load_t1 - load_t0) / 60
        print('Epoch time: {:.2f} minutes; Time left: {:.2f} hours'.format(epoch_time, (epoch_time)*(max_epoch-epoch-1)/60 ))

@tf.function
def train_step(imgs, net,targets1, criterion, optimizer):
    with tf.GradientTape() as tape:
        out = net(imgs)
        loss_l, loss_c, loss_iou = criterion.forward(out, priors, targets1)
        loss = lambda_bbox * loss_l + loss_c + lambda_iouhead * loss_iou
        
    gradients=tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients,net.trainable_variables))
    
    return loss, loss_l, loss_c, loss_iou


if __name__ == '__main__':
    train()
