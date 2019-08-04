"""
https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb
"""
import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg


input_data = tf.random_uniform(shape=(8, 224, 224, 3))
output_data = tf.random_uniform(shape=(8, 1000))
predictions, end_dict = vgg.vgg_16(inputs=input_data, num_classes=1000)
loss = slim.losses.mean_squared_error(predictions, output_data)
total_loss = slim.losses.get_total_loss()
optim = tf.train.AdamOptimizer()
train_op = slim.learning.create_train_op(total_loss, optim)
final_loss = slim.learning.train(train_op, number_of_steps=2, logdir="tensorflow_slim_logs", trace_every_n_steps=1)
