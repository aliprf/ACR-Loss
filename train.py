from config import DatasetName, W300Conf, DatasetType, LearningConfig, InputDataSize, CofwConf
from cnn import CNNModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import csv
from skimage.io import imread
import pickle
from tqdm import tqdm
import os
from data_util import DataUtil
from acr_loss import ACRLoss


class Train:
    def __init__(self, arch, dataset_name, save_path, lambda_weight):

        self.lambda_weight = lambda_weight
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.arch = arch
        self.base_lr = 1e-3
        self.max_lr = 5e-3
        if dataset_name == DatasetName.w300:
            self.num_landmark = W300Conf.num_of_landmarks * 2
            self.img_path = W300Conf.train_image
            self.annotation_path = W300Conf.train_annotation
            '''evaluation path:'''
            self.eval_img_path = W300Conf.test_image_path + 'challenging/'
            self.eval_annotation_path = W300Conf.test_annotation_path + 'challenging/'

        if dataset_name == DatasetName.cofw:
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.img_path = CofwConf.train_image
            self.annotation_path = CofwConf.train_annotation
            '''evaluation path:'''
            self.eval_img_path = CofwConf.test_image_path
            self.eval_annotation_path = CofwConf.test_annotation_path

    def train(self, weight_path):
        """
        :param weight_path:
        :return:
        """
        '''create loss'''
        c_loss = ACRLoss()

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        '''create sample generator'''
        x_train_filenames, y_train_filenames = self._create_generators()

        '''making models'''
        model = self.make_model(arch=self.arch, w_path=weight_path)

        '''create train configuration'''
        step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size

        lr = 1e-3
        for epoch in range(LearningConfig.epochs):
            '''calculate Learning rate'''
            optimizer = self._get_optimizer(lr=lr)
            ''''''
            x_train_filenames, y_train_filenames = self._shuffle_data(x_train_filenames, y_train_filenames)
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                images, annotation_gr = self._get_batch_sample(
                    batch_index=batch_index, x_train_filenames=x_train_filenames,
                    y_train_filenames=y_train_filenames)

                phi = self.calculate_adoptive_weight(epoch=epoch, batch_index=batch_index,
                                                     y_train_filenames=y_train_filenames,
                                                     weight_path=weight_path)

                '''convert to tensor'''
                images = tf.cast(images, tf.float32)
                annotation_gr = tf.cast(annotation_gr, tf.float32)
                '''train step'''
                loss_total, loss_low, loss_high = self.train_step(
                    epoch=epoch, step=batch_index,
                    total_steps=step_per_epoch,
                    images=images,
                    model=model,
                    annotation_gr=annotation_gr,
                    phi=phi,
                    lambda_weight=self.lambda_weight,
                    optimizer=optimizer,
                    summary_writer=summary_writer, c_loss=c_loss)

                with summary_writer.as_default():
                    tf.summary.scalar('loss_total', loss_total, step=epoch)
                    tf.summary.scalar('loss_low', loss_low, step=epoch)
                    tf.summary.scalar('loss_high', loss_high, step=epoch)
            '''save weights'''
            model.save(self.save_path + str(epoch) + '_' + self.dataset_name + '.h5')

    # @tf.function
    def train_step(self, epoch, step, total_steps, images, model, annotation_gr, phi,
                   optimizer, summary_writer, c_loss, lambda_weight):
        with tf.GradientTape() as tape:
            '''create annotation_predicted'''
            annotation_predicted = model(images, training=True)
            '''calculate loss'''
            loss_total, loss_low, loss_high = c_loss.acr_loss(x_pr=annotation_predicted,
                                                              x_gt=annotation_gr,
                                                              phi=phi,
                                                              lambda_weight=lambda_weight,
                                                              ds_name=self.dataset_name)
        '''calculate gradient'''
        gradients_of_model = tape.gradient(loss_total, model.trainable_variables)
        '''apply Gradients:'''
        optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps),
                 ' -> : LOSS: ', loss_total,
                 ' -> : loss_low: ', loss_low,
                 ' -> : loss_high: ', loss_high
                 )
        # print('==--==--==--==--==--==--==--==--==--')
        with summary_writer.as_default():
            tf.summary.scalar('loss_total', loss_total, step=epoch)
            tf.summary.scalar('loss_low', loss_low, step=epoch)
            tf.summary.scalar('loss_high', loss_high, step=epoch)
        return loss_total, loss_low, loss_high

    def calculate_adoptive_weight(self, epoch, batch_index, y_train_filenames, weight_path):

        dt_utils = DataUtil(self.num_landmark)
        batch_y = y_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        asm_acc = None
        if 0 <= epoch <= 15:
            asm_acc = 80
        elif 15 < epoch <= 30:
            asm_acc = 85
        elif 30 < epoch <= 70:
            asm_acc = 90
        elif 70 < epoch <= 100:
            asm_acc = 95

        pn_batch = np.array([self._load_and_normalize(self.annotation_path + file_name) for file_name in batch_y])
        pn_batch_asm = np.array([dt_utils.get_asm(input=self._load_and_normalize(self.annotation_path + file_name),
                                                  dataset_name=self.dataset_name, accuracy=asm_acc)
                                 for file_name in batch_y])

        delta = np.array(abs(pn_batch - pn_batch_asm))

        phi = np.array([delta[i] / np.max(delta[i]) for i in range(len(pn_batch))])  # bs * num_lnd
        return phi

    def _get_optimizer(self, lr=1e-1, beta_1=0.9, beta_2=0.999, decay=1e-5):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    def make_model(self, arch, w_path):
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, output_len=self.num_landmark)
        if w_path is not None:
            model.load_weights(w_path)
        return model

    def _shuffle_data(self, filenames, labels):
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        return filenames_shuffled, y_labels_shuffled

    def _create_generators(self, img_path=None, annotation_path=None):
        tf_utils = DataUtil(number_of_landmark=self.num_landmark)
        if img_path is None:
            filenames, labels = tf_utils.create_image_and_labels_name(img_path=self.img_path,
                                                                      annotation_path=self.annotation_path)
        else:
            filenames, labels = tf_utils.create_image_and_labels_name(img_path=img_path,
                                                                      annotation_path=annotation_path)
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        return filenames_shuffled, y_labels_shuffled

    def _get_batch_sample(self, batch_index, x_train_filenames, y_train_filenames, is_eval=False, batch_size=None):
        if is_eval:
            batch_x = x_train_filenames[
                      batch_index * batch_size:(batch_index + 1) * batch_size]
            batch_y = y_train_filenames[
                      batch_index * batch_size:(batch_index + 1) * batch_size]

            img_batch = np.array([imread(self.eval_img_path + file_name) for file_name in batch_x]) / 255.0
            pn_batch = np.array([load(self.eval_annotation_path + file_name) for file_name in
                                 batch_y])
        else:
            img_path = self.img_path
            pn_tr_path = self.annotation_path
            '''create batch data and normalize images'''
            batch_x = x_train_filenames[
                      batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
            batch_y = y_train_filenames[
                      batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
            '''create img and annotations'''
            img_batch = np.array([imread(img_path + file_name) for file_name in batch_x]) / 255.0
            pn_batch = np.array([self._load_and_normalize(pn_tr_path + file_name) for file_name in batch_y])

        return img_batch, pn_batch

    def _load_and_normalize(self, point_path):
        annotation = load(point_path)
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)
        return annotation_norm
