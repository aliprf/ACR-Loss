from pca_utilities import PCAUtility
from config import DatasetName, DatasetType, W300Conf, InputDataSize, LearningConfig, CofwConf
import tensorflow as tf
import numpy as np
import os
from skimage.transform import resize
import csv
import sys
from PIL import Image
from pathlib import Path
import sqlite3
import cv2
import os.path
from keras import backend as K

from scipy import misc
from scipy.ndimage import gaussian_filter, maximum_filter
from numpy import save, load, asarray
from tqdm import tqdm
import pickle
import PIL.ImageDraw as ImageDraw
import math


class DataUtil:

    def __init__(self, number_of_landmark):
        self.number_of_landmark = number_of_landmark

    def get_asm(self, input, dataset_name, accuracy, alpha=1.0):
        pca_utils = PCAUtility()

        eigenvalues = load('pca_obj/' + dataset_name + pca_utils.eigenvalues_prefix + str(accuracy) + ".npy")
        eigenvectors = load('pca_obj/' + dataset_name + pca_utils.eigenvectors_prefix + str(accuracy) + ".npy")
        meanvector = load('pca_obj/' + dataset_name + pca_utils.meanvector_prefix + str(accuracy) + ".npy")

        b_vector_p = pca_utils.calculate_b_vector(input, True, eigenvalues, eigenvectors, meanvector)
        out = alpha * meanvector + np.dot(eigenvectors, b_vector_p)
        return out

    def create_image_and_labels_name(self, img_path, annotation_path):
        img_filenames = []
        lbls_filenames = []

        for file in os.listdir(img_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                lbl_file = str(file)[:-3] + "npy"  # just name
                if os.path.exists(annotation_path + lbl_file):
                    img_filenames.append(str(file))
                    lbls_filenames.append(lbl_file)

        return np.array(img_filenames), np.array(lbls_filenames)
