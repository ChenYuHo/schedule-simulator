import tensorflow as tf
import tensorflow.keras as k
tf.enable_eager_execution()
print("Tensorflow version: {} Eager execution: {}".format(tf.__version__, tf.executing_eagerly()))
model = k.models.Sequential()
model.add(k.layers.Dense(100, input_shape=(20,)))
model.add(k.layers.Dense(100))
model.add(k.layers.Dense(10))
model.summary()
model.compile(loss="mean_squared_error", optimizer="sgd")
x = tf.random_uniform(shape=(100, 20), maxval=10000)
y = tf.random_uniform(shape=(100, 10), maxval=10000)
print("x shape: {} device: {}".format(x.shape, x.device))
print("y shape: {} device: {}".format(y.shape, y.device))
# This gives us an error for tensorflow 1.13.1 version. Upgrade to 1.14
model.fit(x=x, y=y)

