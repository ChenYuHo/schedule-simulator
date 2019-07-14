import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.client.timeline as timeline
import numpy as np
import json
import time
from model_extraction.keras_model_profiler import clone_layer
import itertools

def get_trace_duration(trace_dict):
    """
    Get the total trace duration in microseconds
    :param trace_dict: A trace dictionary generated in the chrome trace format
    :return: Time in microseconds
    """
    mn = float("inf")
    mx = 0
    for event in trace_dict["traceEvents"]:
        if "ts" not in event:
            continue
        start_time = event["ts"]
        end_time = start_time + event["dur"] if "dur" in event else 0
        if end_time > mx:
            mx = end_time
        if start_time < mn:
            mn = start_time
    return (mx - mn)


def concatenate_trace(trace_dict1, trace_dict2):
    pass


class RunCost(tf.keras.callbacks.LambdaCallback):
    def __init__(self, run_metadata=None):
        from tensorflow.python.client import timeline
        """
        :param run_metadata: If run metadata is None then the time module is used instead of the tracer
        """
        self.costs = list()
        self.run_metadata = run_metadata
        self.tmp = None
        self.c = 0
        self.trace = None

        def on_batch_begin(batch, logs):
            if not self.run_metadata:
                self.tmp = time.time_ns()

        def on_batch_end(batch, logs):
            if self.run_metadata:
                tm = timeline.Timeline(step_stats=self.run_metadata.step_stats)
                ctf = tm.generate_chrome_trace_format()
                self.c += 1
                trace_dict = json.loads(ctf)
                if self.trace is None:
                    self.trace = trace_dict
                else:
                    self.trace["traceEvents"].extend(trace_dict["traceEvents"])
                tc = get_trace_duration(trace_dict) * 1e3
            else:
                tc = time.time_ns() - self.tmp
            self.costs.append(tc)
        super().__init__(on_train_batch_begin=on_batch_begin, on_train_batch_end=on_batch_end,
                         on_predict_batch_begin=on_batch_begin, on_predict_batch_end=on_batch_end,
                         on_test_batch_begin=on_batch_begin, on_test_batch_end=on_batch_end)


with_pooling = True
with_optimization = False
store_trace = False
for with_pooling, with_optimization in [(True, True), (True, False), (False, True), (False, False)]:
    if with_optimization:
        sess = tf.Session()
    else:
        sess = tf.Session(config=tf.ConfigProto(graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0))))
    with sess:
        model = k.models.Sequential()
        model.add(k.layers.Conv2D(64, 8, padding="same", input_shape=(48, 48, 3)))
        model.add(k.layers.Conv2D(64, 16, padding="same"))
        if with_pooling:
            model.add(k.layers.MaxPooling2D(4))
        model.summary()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        metadata = tf.RunMetadata()
        run_cost = RunCost(metadata)
        model.compile(optimizer="sgd", loss="mean_squared_error", options=options, run_metadata=metadata)
        x = tf.random_uniform(shape=(32, 48, 48, 3))
        output_s = 12 if with_pooling else 48
        y = tf.random_uniform(shape=(32, output_s, output_s, 64))
        for i in range(2):
            model.fit(x=x, y=y, callbacks=[run_cost], steps_per_epoch=10)
        if store_trace:
            with open("opt:{}_pooling:{}.json".format(with_optimization, with_pooling), "w") as f:
                json.dump(run_cost.trace, f, indent=4)
        print(run_cost.costs)
