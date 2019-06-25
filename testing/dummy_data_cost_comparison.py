import tensorflow as tf
import time
import numpy as np
import sys



#tf.enable_eager_execution()
shape = [int(x) for x in sys.argv[1:]]

ops = [tf.ones, tf.zeros,  tf.random_normal, tf.random_uniform,]
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
res = list()
with sess:
    for _ in range(3):
        for op in ops:
            t = time.time()
            r = op(shape=shape)
            sess.run(r)
            res.append("{} cost: {}".format(op.__name__, (time.time() - t)*10**3))

print(shape)
print("\n".join(res))