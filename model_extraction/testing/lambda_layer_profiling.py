import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.client.timeline as timeline
import numpy as np
import json
import time
from model_extraction.keras_utils import clone_layer, get_dummy_input_output, extend_trace


options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
metadata = tf.RunMetadata()
trace = None


def label_layer(x):
    return tf.add(x, tf.constant(0.1), name="Coco")
    # y = tf.timestamp()
    # with tf.control_dependencies([y]):
    #     return tf.identity(x, name="Coco")


def batch_end_callback_function(batch, log):
    print("batch_end_callback_function")
    global trace
    current_trace = json.loads(timeline.Timeline(step_stats=metadata.step_stats).generate_chrome_trace_format())
    if trace is None:
        trace = current_trace
    else:
        extend_trace(trace, current_trace)


def batch_begin_callback_function(batch, log):
    pass


callback = k.callbacks.LambdaCallback(
    on_train_batch_begin=batch_begin_callback_function, on_train_batch_end=batch_end_callback_function,
    on_predict_batch_begin=batch_begin_callback_function, on_predict_batch_end=batch_end_callback_function,
    on_test_batch_begin=batch_begin_callback_function, on_test_batch_end=batch_end_callback_function
)

model = k.models.Sequential()
model.add(k.layers.InputLayer(input_shape=(20, 20, 1)))
model.add(k.layers.Conv2D(8,4, name="Layer0"))
model.add(k.layers.MaxPooling2D(name="Layer1"))
model.add(k.layers.Flatten(name="Layer2"))
model.add(k.layers.Dense(32, name="Layer3"))
model.add(k.layers.Dense(32, name="Layer4"))
model.summary()
model.compile(optimizer="sgd", loss="mean_squared_error", options=options, run_metadata=metadata)
x, y = get_dummy_input_output(model, 100000)


# model.predict(x=x, steps=2, callbacks=[callback])
# model.evaluate(x=x, y=y, steps=2, callbacks=[callback])
model.fit(x=x, y=y, steps_per_epoch=4, callbacks=[callback], epochs=1)
with open("lambda_layer_profiling.chrometrace.json", "w") as file:
    json.dump(trace, file, indent=4)