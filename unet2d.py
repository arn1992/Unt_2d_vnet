import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Set some parameters
im_width = 128
im_height = 128
border = 5
path_train = './data/tgs-salt-identification-challenge/train/'
path_test = './data/tgs-salt-identification-challenge/test/'


# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "images"))[2]
    print(ids)
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    z = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, grayscale=True)
        #print(img.shape,"1")
        x_img = img_to_array(img)
        print(x_img.shape,"2")
        x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)
        print(x_img.shape,"3")

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        z[n, ..., 0] = x_img.squeeze()

        if train:

            y[n] = mask / 255

    print('Done!')
    if train:
        return X, y,z
    else:
        return X,z


X, y ,z= get_data(path_train, train=True)
#print(type(X),type(y))
#print(X.shape,y.shape)
#print(X[1])

max_value = float(z.max())
z =z/ max_value
#print("z")

#print(z[1])