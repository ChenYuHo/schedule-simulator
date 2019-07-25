import numpy as np
import tensorflow as tf
import time
import json


def dummy_multi_model():
    """
    https://keras.io/getting-started/functional-api-guide/
    :return: A multi input multi output model for testing
    """
    from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, concatenate
    from tensorflow.python.keras.models import Model
    main_input = Input(shape=(100,), dtype='int32', name='main_input')
    x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
    lstm_out = LSTM(32)(x)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
    auxiliary_input = Input(shape=(5,), name='aux_input')
    x = concatenate([lstm_out, auxiliary_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(name="dummy_multi", inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    return model


def dummy_linear_dense_model(units=100, n_of_layers=5):
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense
    model = Sequential(name="dummy_linear_dense")
    model.add(Dense(units, input_shape=(units,)))
    for _ in range(n_of_layers-1):
        model.add(Dense(units))
    return model


def dummy_linear_cnn_model():
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
    model = Sequential(name="dummy_linear_cnn")
    model.add(Conv2D(8, 16, padding="same", input_shape=(256, 256, 1), activation="relu"))
    model.add(Conv2D(16, 16, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 32, padding="same", activation="relu"))
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(Conv2D(64, 64, padding="same", activation="relu"))
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(8, activation="relu"))
    return model


def dummy_2_layers_model():
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Conv2D
    model = Sequential(name="dummy_2_layers")
    model.add(Conv2D(8, 16, padding="same", input_shape=(256, 256, 1), activation="relu"))
    #model.add(Dense(64))
    return model


def get_layer_parents(layer):
    # Note that the keras model structure uses a layer node layer node scheme.
    # A `Node` describes the connectivity between two layers See
    # https://github.com/keras-team/keras/blob/b0bfd5201da2bfced84028bcc5bda05bdfd75af7/keras/engine/base_layer.py#L1178
    parents = set()
    for node in layer._inbound_nodes:
        try:
            for parent in node.inbound_layers:
                parents.add(parent)
        except TypeError:  # In case inbound_layers is a single layer and not iterable
            parents.add(node.inbound_layers)
    return parents


def get_layer_children(layer):
    children = set()
    for node in layer._outbound_nodes:
        children.add(node.outbound_layer)
    return children


def traverse_keras_DFS(model, processing_function, order="post-order", top_to_bottom=True):
    """
    Uses a recursive DFS O(n) to process all layers in a Keras model using the processing_function.
    :param model: Keras model to traverse.
    :param processing_function: The function that will be called on each layer. should only require one argument
    which is the layer being processed.
    :param order: The order of traversal. "pre-order" or "post-order"
    :param top_to_bottom: If false, the traversal considers the end of the model as the root
    """
    if order != "pre-order" and order != "post-order":
        raise Exception("Invalid order '{}' provided.".format(order))

    def traverse(root, visited: set):
        visited.add(root)
        if order == "pre-order":
            processing_function(root)
        if top_to_bottom:
            for child_layer in get_layer_children(root):
                if child_layer not in visited:
                    traverse(root=child_layer, visited=visited)
        else:
            for parent_layer in get_layer_parents(root):
                if parent_layer not in visited:
                    traverse(root=parent_layer, visited=visited)
        if order == "post-order":
            processing_function(root)
    visited = set()
    if top_to_bottom:
        for end_layer in model._input_layers:
            traverse(root=end_layer, visited=visited)
    else:
        for end_layer in model._output_layers:
            traverse(root=end_layer, visited=visited)


def clone_layer(layer):
    """
    Creates a clone of the layer using only its main properties (Does not copy any connections from the clone)
    Better implementation exists ?
    """
    from tensorflow.python.keras.layers import serialize, deserialize
    return deserialize(serialize(layer))


def get_dummy_input_output(model, num_of_samples, use_numpy=False):
    # Keras gives us a list of shapes only in case of multiple inputs / outputs
    model_input_shapes = [model.input_shape] if model.input_shape[0] is None else model.input_shape
    model_output_shapes = [model.output_shape] if model.output_shape[0] is None else model.output_shape
    input_shapes = list()
    for shape in [list(input_shape) for input_shape in model_input_shapes]:
        shape[0] = num_of_samples
        input_shapes.append(shape)
    output_shapes = list()
    for shape in [list(output_shape) for output_shape in model_output_shapes]:
        shape[0] = num_of_samples
        output_shapes.append(shape)
    # Which op takes less time depends on whether you are using gpu or not and whether you are using eager execution or
    # not
    if use_numpy:
        input_data = [np.random.uniform(size=shape) for shape in input_shapes]
        output_data = [np.random.uniform(size=shape) for shape in output_shapes]
    else:
        input_data = [tf.random.uniform(shape=shape, name="data_generation_input") for shape in input_shapes]
        output_data = [tf.random.uniform(shape=shape, name="data_generation_output") for shape in output_shapes]
    return input_data, output_data


def get_trace_costs(trace_dict):
    """
    Duration: The total trace duration in microseconds calculated by looking at the very first op and the last op.
    Total time costs: Adds all the durations of operations disregarding time gaps and parallelization
    :param trace_dict: A trace dictionary generated in the chrome trace format
    :return: Tuple (Duration, Total time costs)
    """
    mn = float("inf")
    mx = 0
    total_cost = 0
    for event in trace_dict["traceEvents"]:
        if event["name"].startswith("data_generation"):
            continue
        if "dur" in event:
            total_cost += event["dur"]
        if "ts" not in event:
            continue
        start_time = event["ts"]
        end_time = start_time + event["dur"] if "dur" in event else 0
        if end_time > mx:
            mx = end_time
        if start_time < mn:
            mn = start_time
    return mx - mn, total_cost


def extend_trace(original_trace_dict, to_be_added_trace_dict, inplace=True):
    # TODO remove this and use join_chrome_traces in utils instead
    if inplace:
        original_trace_dict["traceEvents"].extend(to_be_added_trace_dict["traceEvents"])
    else:
        return original_trace_dict["traceEvents"].copy().extend(to_be_added_trace_dict["traceEvents"])


class RunCost(tf.keras.callbacks.LambdaCallback):
    def __init__(self, run_metadata=None):
        from tensorflow.python.client import timeline
        """
        :param run_metadata: If run metadata is None then the time module is used instead of the tracer
        """
        self.total_time_costs = list()
        self.durations = list()
        self.run_metadata = run_metadata
        self.tmp = None

        def on_batch_begin(batch, logs):
            if not self.run_metadata:
                self.tmp = time.time_ns()

        def on_batch_end(batch, logs):
            if self.run_metadata:
                tm = timeline.Timeline(step_stats=self.run_metadata.step_stats)
                duration, total_costs = get_trace_costs(json.loads(tm.generate_chrome_trace_format()))
                self.durations.append(duration * 1e3)
                self.total_time_costs.append(total_costs * 1e3)
            else:
                tc = time.time_ns() - self.tmp
                self.durations.append(tc)
        super().__init__(on_train_batch_begin=on_batch_begin, on_train_batch_end=on_batch_end,
                         on_predict_batch_begin=on_batch_begin, on_predict_batch_end=on_batch_end,
                         on_test_batch_begin=on_batch_begin, on_test_batch_end=on_batch_end)


def layer_input_output_analysis(model):
    layers = list()

    def process_layer(layer):
        layer_dict = dict()
        layer_dict["name"] = layer.name
        layer_dict["type"] = type(layer).__name__
        layer_dict["input_shape"] = None
        layer_dict["output_shape"] = None
        layer_dict["parameters"] = 0
        layers.append(layer_dict)
    traverse_keras_DFS(model=model, processing_function=process_layer)
    return layers