import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.client.timeline as timeline
import numpy as np
import json
import time
from model_extraction.keras_model_profiler import clone_layer, get_dummy_input_output, extend_trace


options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
metadata = tf.RunMetadata()
trace = None


def label_layer(x):
    return x
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
model.add(k.layers.InputLayer(input_shape=(20,)))
model.add(k.layers.Dense(32))
model.add(k.layers.Lambda(label_layer))
model.summary()
model.compile(optimizer="sgd", loss="mean_squared_error",options=options, run_metadata=metadata)
x, y = get_dummy_input_output(model, 10)


model.predict(x=x, steps=2, callbacks=[callback])
model.evaluate(x=x, y=y, steps=2, callbacks=[callback])
model.fit(x=x, y=y, steps_per_epoch=2, callbacks=[callback], epochs=1)
with open("lambda_layer_profiling_trace.json", "w") as file:
    json.dump(trace, file, indent=4)