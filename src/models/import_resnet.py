"""
These functions imports the pre-trained resnet and builds a model.
"""

#----------------------IMPORTS---------------------------#

import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.applications import ResNet50

#------------------------FUNCTIONS-----------------------#

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)

    return x

def build_resnet50_unet(input_shape, input_dim):
    """input"""
    inputs = Input(input_shape)

    """Pre-trained ResNet50"""
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    """Encoder"""
    # shapes:s
    s1 = resnet.get_layer('input_1').output             ## input_dim
    s2 = resnet.get_layer('conv1_relu').output          ## 256
    s3 = resnet.get_layer('conv2_block3_out').output    ## 128
    s4 = resnet.get_layer('conv3_block4_out').output    ## 64

    """Bridge"""
    b1 = resnet.get_layer('conv4_block6_out').output    ## 32

    """Decoder"""
    d1 = decoder_block(b1, s4, input_dim)                     ## 64
    d2 = decoder_block(d1, s3, input_dim)                     ## 128
    d3 = decoder_block(d2, s2, input_dim)                     ## 256
    d4 = decoder_block(d3, s1, input_dim)                     ## input_dim

    """Output"""
    outputs = Conv2D(4, 1, padding='same', activation='softmax')(d4)    ## changed from 1

    model = Model(inputs, outputs)
    return model


#------------------------EXECUTION-----------------------#

if __name__ == "__main__":
    input_dim = 512
    path = "../../models/segmentation_options/"

    input_shape = (input_dim, input_dim, 3)
    model = build_resnet50_unet(input_shape, input_dim)
    model.summary()
    model.save(f'{path}{input_dim}_resnet50_unet.h5')
    tf.keras.backend.clear_session()