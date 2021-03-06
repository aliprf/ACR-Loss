from config import DatasetName, W300Conf, DatasetType, LearningConfig, InputDataSize, CofwConf
import tensorflow as tf

import cv2
import os.path
import scipy.io as sio
from cnn import CNNModel
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.integrate import simps
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from skimage.io import imread


class Test:
    def test_model(self, pretrained_model_path, ds_name):
        if ds_name == DatasetName.w300:
            test_annotation_path = W300Conf.test_annotation_path
            test_image_path = W300Conf.test_image_path
        elif ds_name == DatasetName.cofw:
            test_annotation_path = CofwConf.test_annotation_path
            test_image_path = CofwConf.test_image_path

        model = tf.keras.models.load_model(pretrained_model_path)

        for i, file in tqdm(enumerate(os.listdir(test_image_path))):
            # load image and then normalize it
            img = imread(test_image_path + file) / 255.0

            # prediction
            prediction = model.predict(np.expand_dims(img, axis=0))

            # the first dimension is landmark point
            landmark_predicted = prediction
