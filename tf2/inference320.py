#!/usr/bin/python3
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import sys
import argparse
import cv2
import numpy as np
from collections import OrderedDict
import time

sys.path.append(os.getcwd() + '/../../src')

from config import cfg
from prior_box import PriorBox
from nms import nms
from utils import decode
from timer import Timer
from yufacedetectnet import YuFaceDetectNet
from yufacedetectnet1 import YuFaceDetectNet1


parser = argparse.ArgumentParser(description='Face and Landmark Detection')

# parser.add_argument('-m', '--trained_model', default='weights/yunet_final.pth',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('--image_file', default='', type=str, help='the image file to be detected')
parser.add_argument('--confidence_threshold', default=0.50, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
#parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.3, type=float, help='visualization_threshold')
parser.add_argument('--base_layers', default=16, type=int, help='the number of the output of the first layer')
parser.add_argument('--device', default='cuda:0', help='which device the program will run on. cuda:0, cuda:1, ...')
args = parser.parse_args()

vid_path = "facetest.mp4"

@tf.function
def infer(net,img):
    loc, conf, iou = net(img)
    return loc, conf, iou


if __name__ == '__main__':
    img_dim = 320
   
    """### Keras functional model"""

    # net = YuFaceDetectNet1('train', (180,320,3))   # use this model for SAVEDMODEL conversion. Subclassed model converted to SAVEDMODEL format doesnt work well
    # net.load_weights("checkpoints/adam_net.h5")
    
    """for conversion to SAVED MODEL run these along with above 2 lines"""

    # net(tf.random.normal((8,320,320,3)))
    # net.save("saved_model",save_format="tf")
    # tf.saved_model.save(net,"saved_model")
    # exit()
    ##*****************************##

    """###Subclassed Model"""
 
    net = YuFaceDetectNet("test", (320,320,3))
    net.load_weights("subclass/weights")
    
    """#loading Saved model / TRT (TF-TRT conversion) model"""

    # net = tf.saved_model.load("/home/arm/Projects/LibFaceDetection/My work/saved_model")
    # net = tf.saved_model.load("/home/arm/Projects/LibFaceDetection/My work/trt")
    
    # print("Printing net...")
    # print(net.weights)
    # net.summary()


    print('Finished loading model!')
    
    _t = {'forward_pass': Timer(), 'misc': Timer()}

    cap = cv2.VideoCapture(vid_path) #27sec
    # cap = cv2.VideoCapture("filesrc location=\"/path/to/video.mp4\"  decodebin ! videoconvert ! autovideosink",cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture("rtspsrc location=rtsp://admin:abcd1234@10.0.0.236 latency=0 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink",cv2.CAP_GSTREAMER)

    (grabbed, frame) = cap.read()
    im_height1, im_width1, _ = frame.shape

    # used to record the time when we processed last frame 
    prev_frame_time = 0
    # used to record the time at which we processed current frame 
    new_frame_time = 0
    FPS=0
    tot=0
    im_width = img_dim
    im_height = int((im_width*im_height1)/im_width1)
    # im_height = img_dim
    priorbox = PriorBox(cfg, image_size=(im_height,im_width))       
    priors = priorbox.forward()
    # priors = priors.to(device)
    # exit()
    scale = np.array([im_width, im_height, im_width, im_height])
                                # im_width, im_height, im_width, im_height,
                                # im_width, im_height, im_width, im_height,
                                # im_width, im_height ])
    vid_start=time.time()
    while True:
        new_frame_time = time.time() 
        preproc_start = time.time()
        ret, img_raw = cap.read()
        
        if ret:
            img = cv2.resize(img_raw,(im_width, im_height))
            img = tf.convert_to_tensor(img,dtype="float32")
            img = img[tf.newaxis, ...]
            preproc_end = time.time()
            print("preproc:",(preproc_end-preproc_start)*1000)

            _t['forward_pass'].tic()

            loc, conf, iou = infer(net,img)  # forward pass
            print("Inference_Time",_t['forward_pass'].toc()*1000)
            
            _t['misc'].tic()

            prior_data = priors
            boxes = decode(tf.squeeze(loc,axis=0), prior_data, cfg['variance'])
            boxes = boxes * scale

            cls_scores = tf.squeeze(conf,axis=0).numpy()[:, 1]

            """ use these 2 lines if using functional """
            # cls_scores = tf.keras.layers.Softmax(axis=-1)(conf)
            # cls_scores = tf.squeeze(cls_scores,axis=0).numpy()[:, 1]
            
            iou_scores = tf.squeeze(iou,axis=0).numpy()[:, 0]

            # clamp here for the compatibility for ONNX
            _idx = np.where(iou_scores < 0.)
            iou_scores[_idx] = 0.
            _idx = np.where(iou_scores > 1.)
            iou_scores[_idx] = 1.
            scores = np.sqrt(cls_scores * iou_scores)

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            scores = scores[order]
            
            # print('there are', len(boxes), 'candidates')

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            selected_idx = np.array([0,1,2,3,4])
            keep = nms(dets[:,selected_idx], args.nms_threshold)
            dets = dets[keep, :]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]

            # show image
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (int(b[0]*(im_width1/im_width)), int(b[1]*(im_height1/im_height))), 
                (int(b[2]*(im_width1/im_width)), int(b[3]*(im_height1/im_height))), (0, 255, 0), 2)

                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (int(cx*(im_width1/im_width)), int(cy*(im_height1/im_height))),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            total_time=new_frame_time-prev_frame_time
            fps = 1/(new_frame_time-prev_frame_time) 
            prev_frame_time = new_frame_time 
            FPS = FPS+fps
            tot = tot+1
            # converting the fps into integer 
            fps = str(round(fps,1)) 
            
            cv2.putText(img_raw,fps, (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            cv2.namedWindow('res', cv2.WINDOW_NORMAL )
            cv2.imshow('res', img_raw)
            # cv2.resizeWindow('res', im_width, im_height)
            # out.write(img_raw)
            
            print("misc_Time",_t['misc'].toc()*1000)
            print("Total-Time",total_time*1000,"\n")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    vid_end=time.time()
    print("AVG FPS:",FPS/tot)
    print("Total_Process:",vid_end-vid_start)
    # tf.saved_model.save(net,"saved_model")
    # cap.release()
    cv2.destroyAllWindows()
