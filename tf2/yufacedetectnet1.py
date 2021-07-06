import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D,Softmax,Flatten,Reshape,ReLU,MaxPool2D,Concatenate,BatchNormalization

def convrelu(x,filters=32, kernel_size=3, stride=1, padding="valid",name="", **kwargs):

    x = Conv2D(filters,kernel_size,stride,name=name,bias_initializer=tf.keras.initializers.Constant(value=0.2),padding=padding,**kwargs)(x)
    x = BatchNormalization(name="BN"+name)(x)
    x = ReLU(name="relu"+name)(x)
    return x

def conv_2layers(x,f1=32, f2=32, k1=3, k2=1, stride=1,name="",**kwargs):

    x = convrelu(x,f1,k1,stride,padding="same",name=name+"1",**kwargs)
    x = convrelu(x,f2,k2,name=name+"2",**kwargs)
    return x

def conv_3layers(x,f1=32, f2=32, f3=32, k1=3, k2=1, k3=3,name="", **kwargs):

    x = convrelu(x,f1,k1,padding="same",name=name+"1",**kwargs)
    x = convrelu(x,f2,k2,name=name+"2",**kwargs)
    x = convrelu(x,f3,k3,padding="same",name=name+"3",**kwargs) 
    return x

def multibox():
    loc_layers = []
    conf_layers = []
    iou_layers = []
    num_classes = 2
    constant = tf.keras.initializers.Constant(value=0.2)
    loc_layers += [Conv2D(3 * 4, kernel_size=3, padding="same",name="loc1",bias_initializer=constant )]
    conf_layers += [Conv2D(3 * num_classes, kernel_size=3, padding="same",name="conf1",bias_initializer=constant)]
    iou_layers += [Conv2D(3, kernel_size=3, padding="same",name="iou1",bias_initializer=constant)]

    loc_layers += [Conv2D(2 * 4, kernel_size=3, padding="same",name="loc2",bias_initializer=constant)]
    conf_layers += [Conv2D(2 * num_classes, kernel_size=3, padding="same",name="conf2",bias_initializer=constant)]
    iou_layers += [Conv2D(2, kernel_size=3, padding="same",name="iou2",bias_initializer=constant)]

    loc_layers += [Conv2D(2 * 4, kernel_size=3, padding="same",name="loc3",bias_initializer=constant)]
    conf_layers += [Conv2D(2 * num_classes, kernel_size=3, padding="same",name="conf3",bias_initializer=constant)]
    iou_layers += [Conv2D(2, kernel_size=3, padding="same",name="iou3",bias_initializer=constant)]

    loc_layers += [Conv2D(3 * 4, kernel_size=3, padding="same",name="loc4",bias_initializer=constant)]
    conf_layers += [Conv2D(3 * num_classes, kernel_size=3, padding="same",name="conf4",bias_initializer=constant)]
    iou_layers += [Conv2D(3, kernel_size=3, padding="same",name="iou4",bias_initializer=constant)]

    return loc_layers, conf_layers, iou_layers


def YuFaceDetectNet1(phase,size):
    
    detection_sources = list()
    loc_data = list()
    conf_data = list()
    iou_data = list()

    inp = tf.keras.Input(shape=size)
    x = conv_2layers(inp,32,16,stride=2,name="m1")
    x = MaxPool2D(2,name="MaxP1")(x)
    x = conv_2layers(x,32,16,name="m2")
    x = MaxPool2D(2,name="MaxP2")(x)
    
    x = conv_3layers(x,64,32,64,name="m3")
    detection_sources.append(x)
    
    x = MaxPool2D(2,name="MaxP3")(x)
    x = conv_3layers(x,128,64,128,name="m4")
    detection_sources.append(x)
    
    x = MaxPool2D(2,name="MaxP4")(x)
    x = conv_3layers(x,256,128,256,name="m5")
    detection_sources.append(x)
    
    x = MaxPool2D(2,name="MaxP5")(x)
    x = conv_3layers(x,256,256,256,name="m6")
    detection_sources.append(x)

    loc, conf, iou = multibox()
    # print(loc[0])
    for (x, l, c, i) in zip(detection_sources, loc, conf, iou):
        loc_data.append(l(x))
        conf_data.append(c(x))
        iou_data.append(i(x))

    # for l in loc_data:
    #     x=Flatten()(l)
    #     print(x) 
    #     exit()
    loc_data = Concatenate(axis=1,name="loc_cat")([Flatten(name="loc_flat"+str(x))(l) for x,l in enumerate(loc_data)])
    conf_data = Concatenate(axis=1,name="conf_cat")([Flatten(name="conf_flat"+str(x))(c) for x,c in enumerate(conf_data)])
    iou_data = Concatenate(axis=1,name="iou_cat")([Flatten(name="iou_flat"+str(x))(i) for x,i in enumerate(iou_data)])
    
    if phase == "test":
        output = (Reshape((-1,4),name="loc")(loc_data),
                Softmax(axis=-1,name="Smax")(Reshape((-1,2),name="conf")(conf_data)),
                Reshape((-1,1),name="iou")(iou_data),
                )
        # print("TEST_Output_Layer",output[0].shape,output[1].shape,output[2].shape)
        
    else:
        output=(Reshape((-1,4),name="loc")(loc_data),
                Reshape((-1,2),name="conf")(conf_data),
                Reshape((-1,1),name="iou")(iou_data))
        # print("Output_Layer",output[0].shape,output[1].shape,output[2].shape)

    model = tf.keras.Model(inputs=inp,outputs=output)
    return model

# m = YuFaceDetectNet1("train",(320,320,3))
# m.build((1,320,320,3))
# m.summary()
# m.save("train.h5")
# m.save_weights("train_weights.h5")

# y = YuFaceDetectNet1("test",(320,320,3))
# y.build((1,320,320,3))
# y(tf.random.normal((8,320,320,3)))
# print(m.layers)
# for i in range(len(m.layers)):
#     print(m.get_layer(index=i).name,m.get_layer(index=i).input_shape)
#     print(y.get_layer(index=i).name,y.get_layer(index=i).input_shape)
#     print('\n')
# exit()
# y.set_weights(m.get_weights())
# y.load_weights("train_weights.h5")
# m(tf.random.normal((8,320,320,3)))
# m.summary()
# m.save("test.h5")

# ann = Sequential()
# x = Conv2D(filters=3,kernel_size=(3,3),input_shape=(10,10,3))
# ann.add(x)

# print(x.get_weights()[0])