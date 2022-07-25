from config import DatasetName
import tensorflow as tf


class ACRLoss:
    def acr_loss(self, x_pr, x_gt, phi, lambda_weight, ds_name):
        low_map = tf.cast(tf.abs(x_pr - x_gt) <= 1.0, dtype=tf.float32)
        high_map = tf.cast(tf.abs(x_pr - x_gt) > 1.0, dtype=tf.float32)

        '''Big errors'''
        ln_2 = tf.ones_like(x_pr, dtype=tf.float32) * tf.math.log(2.0)
        C = tf.cast(tf.cast(phi, dtype=tf.double) * tf.cast(ln_2, dtype=tf.double) - 1.0, dtype=tf.float32)
        loss_high = 100 * tf.reduce_mean(tf.math.multiply(high_map, (tf.square(x_pr - x_gt) + C)))

        '''Small errors'''
        power = tf.cast(2.0 - phi, tf.dtypes.float32)
        ll = tf.pow(tf.abs(x_pr - x_gt), power)
        loss_low = 100 * tf.reduce_mean(tf.math.multiply(low_map, (lambda_weight * tf.math.log(1.0 + ll))))

        loss_total = loss_low + loss_high

        return loss_total, loss_low, loss_high
