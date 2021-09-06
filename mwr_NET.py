import argparse

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from keras import models
from keras.applications.densenet import layers
from keras.layers import Conv2D, Flatten, Dense
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random


from keras_preprocessing.image import ImageDataGenerator
'''
This is a script for create and execute keras model and save it to .h5 and .json
'''

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', default='Data/train/')

    return parser.parse_args()


IMG_WIDTH=200
IMG_HEIGHT=200
img_folder='Data/train/'
def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

def save_model_to_json(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    print('Model saved to json')

def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4)
        ])
    return model

def main(args):

    img_folder = args.input_folder


    # for i in range(5):
    #     file = random.choice(os.listdir(img_folder))
    #     image_path = os.path.join(img_folder, file)
    #     img = mpimg.imread(image_path)
    #
    #     ax = plt.subplot(1, 5, i + 1)
    #     ax.title.set_text(file)
    #     plt.imshow(img)
    img_data, class_name = create_dataset(img_folder)
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

    print(target_val)
    model = create_model()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int, target_val)), np.float32), epochs=20)

    save_model_to_json(model)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    model.save('model_MWR.h5')
    print("Saved model")



if __name__ == '__main__':
    main(parse_arguments())