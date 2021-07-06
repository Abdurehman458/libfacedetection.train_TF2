import os

from tensorflow.python.keras.backend import constant

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Softmax,Flatten,Permute,ReLU,MaxPool2D,Concatenate,BatchNormalization,Reshape
import numpy as np

class ConvRelu(Layer):
    def __init__(self, filters=32, kernel_size=3, stride=1, padding="valid",name="", **kwargs):
        super(ConvRelu, self).__init__()

        self.conv = Conv2D(filters,kernel_size,stride,padding,**kwargs)
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv2(Layer):
    def __init__(self, f1=32, f2=32, k1=3, k2=1, stride=1, **kwargs):
        super(Conv2, self).__init__()

        self.conv1 = ConvRelu(f1,k1,stride,padding="same",**kwargs)
        self.conv2 = ConvRelu(f2,k2,**kwargs)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Conv3(Layer):
    def __init__(self, f1=32, f2=32, f3=32, k1=3, k2=1, k3=3,name="", **kwargs):
        super(Conv3, self).__init__()

        self.conv1 = ConvRelu(f1,k1,padding="same",name=name,**kwargs)
        self.conv2 = ConvRelu(f2,k2,name=name,**kwargs)
        self.conv3 = ConvRelu(f3,k3,padding="same",name=name,**kwargs)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class YuFaceDetectNet(Model):

    def __init__(self,phase,size):
        super(YuFaceDetectNet, self).__init__()
        self.size = size
        self.num_classes = 2
        self.phase = phase
        self.conv2_1 = Conv2(32,16,stride=2,name="conv1")
        self.conv2_2 = Conv2(32,16,name="conv2")
        self.conv3_1 = Conv3(64,32,64,name="conv3")
        self.conv3_2 = Conv3(128,64,128,name="conv4")
        self.conv3_3 = Conv3(256,128,256,name="conv5")
        self.conv3_4 = Conv3(256,256,256,name="conv6")
        self.loc, self.conf, self.iou = self.multibox(self.num_classes)
        # self.cat = Concatenate(axis=1)
        if self.phase == "test":
            self.softmax = tf.keras.layers.Softmax(axis=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        iou_layers = []
        constant = tf.keras.initializers.Constant(value=0.2)
        loc_layers += [Conv2D(3 * 4, kernel_size=3, padding="same",name="loc1" )]
        conf_layers += [Conv2D(3 * num_classes, kernel_size=3, padding="same",name="conf1")]
        iou_layers += [Conv2D(3, kernel_size=3, padding="same",name="iou1" )]

        loc_layers += [Conv2D(2 * 4, kernel_size=3, padding="same",name="loc2")]
        conf_layers += [Conv2D(2 * num_classes, kernel_size=3, padding="same",name="conf2")]
        iou_layers += [Conv2D(2, kernel_size=3, padding="same",name="iou2")]

        loc_layers += [Conv2D(2 * 4, kernel_size=3, padding="same",name="loc3")]
        conf_layers += [Conv2D(2 * num_classes, kernel_size=3, padding="same",name="conf3")]
        iou_layers += [Conv2D(2, kernel_size=3, padding="same", name="iou3")]

        loc_layers += [Conv2D(3 * 4, kernel_size=3, padding="same",name="loc4")]
        conf_layers += [Conv2D(3 * num_classes, kernel_size=3, padding="same",name="conf4")]
        iou_layers += [Conv2D(3, kernel_size=3, padding="same",name="iou4")]

        return loc_layers, conf_layers, iou_layers

    def call(self, x):
        # y = x
        detection_sources = list()
        loc_data = list()
        conf_data = list()
        iou_data = list()
        # x =self._input
        x = self.conv2_1(x)
        x = MaxPool2D(2)(x)

        x = self.conv2_2(x)
        x = MaxPool2D(2)(x)

        x = self.conv3_1(x)
        detection_sources.append(x)
        x = MaxPool2D(2)(x)
        
        x = self.conv3_2(x)
        detection_sources.append(x)
        x = MaxPool2D(2)(x)
        
        x = self.conv3_3(x)
        detection_sources.append(x)
        x = MaxPool2D(2)(x)
        x = self.conv3_4(x)
        # print(x)
        detection_sources.append(x)
        
        for (x, l, c, i) in zip(detection_sources, self.loc, self.conf, self.iou):
            loc_data.append(l(x))
            conf_data.append(c(x))
            iou_data.append(i(x))

        loc_data = tf.concat([tf.reshape(l,(tf.shape(l)[0],-1)) for l in loc_data],axis=1)
        conf_data = tf.concat([tf.reshape(c,(tf.shape(c)[0],-1)) for c in conf_data],axis=1)
        iou_data = tf.concat([tf.reshape(i,(tf.shape(i)[0],-1)) for i in iou_data],axis=1)
        # loc_data = Concatenate(axis=1,name="loc_cat")([Flatten(name="loc_flat"+str(x))(l) for x,l in enumerate(loc_data)])
        # conf_data = Concatenate(axis=1,name="conf_cat")([Flatten(name="conf_flat"+str(x))(c) for x,c in enumerate(conf_data)])
        # iou_data = Concatenate(axis=1,name="iou_cat")([Flatten(name="iou_flat"+str(x))(i) for x,i in enumerate(iou_data)])
        # print(loc_data.shape,conf_data.shape,iou_data.shape)

        if self.phase == "test":
            output = (tf.reshape(loc_data,(tf.shape(loc_data)[0],-1,4)),
                self.softmax(tf.reshape(conf_data,(tf.shape(conf_data)[0],-1,self.num_classes))),
                    tf.reshape(iou_data,(tf.shape(iou_data)[0],-1,1)))
            # print("TEST_Output_Layer",output[0].shape,output[1].shape,output[2].shape)
        
        else:
            output=(tf.reshape(loc_data,(tf.shape(loc_data)[0],-1,4)),
                    tf.reshape(conf_data,(tf.shape(conf_data)[0],-1,self.num_classes)),
                    tf.reshape(iou_data,(tf.shape(iou_data)[0],-1,1)))
            # print("Output_Layer",output[0].shape,output[1].shape,output[2].shape)

        return output

# m = YuFaceDetectNet("train",(320,320,3))
# m(tf.random.normal((8,320,320,3)))
# m.summary()
# tf.saved_model.save
# m.save("modell",save_format="tf")
# m.save_weights('/home/arm/Projects/LibFaceDetection/My work/subclass/',save_format="tf")
# y = YuFaceDetectNet("test",(320,320,3))
# y(tf.random.normal((8,320,320,3)))
# y.load_weights('/home/arm/Projects/LibFaceDetection/My work/subclass/savedmodel')
# print(len(m.layers))
# pb = PriorBox()
# print(pb)
# print(pb.generate_priors().shape)
# from tensorflow.keras import backend as K
# print(K.image_data_format())

#**********************************************************#

# x = tf.keras.Input(shape=(5,5,3))
# # y = ConvRelu(32,3,padding="same",name="convv")(x)
# y = ConvRelu(5,3,name="convv")(x)
# y = Conv3(32,16,6)(y)
# # print(y)
# mm=tf.keras.Model(x,y)

# d=tf.random.normal((1,5,5,3))
# # print(y(d))

# import tensorflow.keras.backend as K

# xx = x = tf.keras.Input(shape=(5,5,3))
# yy = Conv2D(5,3,1,activation='relu')(xx)
# # yy = tf.keras.layers.BatchNormalization()(yy)
# # yy = tf.keras.backend.l2_normalize(yy, axis=3)
# mod = tf.keras.Model(xx,yy)
# # print(mod(d))

# get_1st_layer_output = K.function([mod.layers[0].input],
#                                   [mod.layers[1].output])
# layer_output = get_1st_layer_output(d)
# print(layer_output)