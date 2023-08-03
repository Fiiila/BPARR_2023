# imports
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model

## REFERENCES
"""
mainly made according to describrion in https://arxiv.org/pdf/1505.04597.pdf
something inspired in https://keras.io/examples/vision/oxford_pets_image_segmentation/
something more inspired in https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb

but everything made to work with simple custom dataset
pictures need to be .jpg
masks need to be .png with mask values 1,2,3...important!!! check before
for more than one mask need to repair some functions in train.py and DatasetGenerator.py
made by Filip Jašek  26.7.2021
github: Fiiila 
this project currently in private repository
"""

# function part
def encoder_block(inputs, filters, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', pool_size=(2,2)):
    """typicky konvolucni blok jako jeden blok encoderu
    :param inputs predefined input of the block
    :param filters (integer) the dimensionality of the output space
    :param kernel_size An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
    window. Can be a single integer to specify the same value for all spatial dimensions.
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation)(inputs)
    """jiny zdroj, kde se pouziva separatni aktivace a normalizace
    x = BatchNormalization()(x)
    x = Activation(activation='relu')"""
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation)(x)
    mp = MaxPooling2D(pool_size=pool_size)(x)

    return x, mp


def bridge_block(inputs, filters, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'):
    """bridge between encoder and decoder
    """

    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation)(inputs)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation)(x)

    return x


def decoder_block(inputs, copy, filters, size=(2, 2), kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'):
    """one decoder block

    """
    x = UpSampling2D(size=size)(inputs)
    x = Concatenate()([x, copy])
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation)(x)

    return x

# 16, 32, 64, 128, 256
def build_Unet(input_shape=(256, 256, 3), filters=[64, 128, 256, 512, 1024], model_name='Unet_model'):
    """method that build Unet model
    c* - copy after convolution in encoder block for decoder
    p* - encoder block output after pooling"""
    filters = filters
    inputs = Input(input_shape)

    #ENCODER
    c1, p1 = encoder_block(inputs, filters[0])
    c2, p2 = encoder_block(p1, filters[1])
    c3, p3 = encoder_block(p2, filters[2])
    c4, p4 = encoder_block(p3, filters[3])

    #BRIDGE
    b = bridge_block(p4, filters[4])

    #DECODER
    u1 = decoder_block(b, c4, filters[3])
    u2 = decoder_block(u1, c3, filters[2])
    u3 = decoder_block(u2, c2, filters[1])
    u4 = decoder_block(u3, c1, filters[0])

    outputs = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(u4) #activation='softmax'/'sigmoid'

    model = Model(inputs, outputs, name=model_name)
    return model

if __name__ == "__main__":
    model = build_Unet()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    model.save("model.h5")
