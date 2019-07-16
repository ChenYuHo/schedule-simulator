"""
A script for getting layer wise timings and other information about any Keras model.
"""

import argparse
import sys
from datetime import datetime
import json
import socket
import os
import time
import tensorflow as tf
import contextlib


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


def get_dummy_input_output(model, num_of_samples):
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


def model_reconstruct_layerwise_costs_profiling(input_model, loss, optimizer, batch_size=32, num_of_batches=8, trials=1,
                                                verbosity=1, device="gpu", use_tracer=True,
                                                skip_untrainable_layers=False):
    """
    The function takes a keras model and attempts to estimate the cost of each layer in the model in terms of
    the fitting evaluating and predicting.
    :param input_model: The model to profile
    :param batch_size: The batch size used for all functions.
    :param num_of_batches: The number of batches to run
    :param trials: The number of layer building & evaluation iterations to do.
    :param device: cpu or gpu ?
    :param loss: The loss function for the model
    :param optimizer: The optimizer used for the model
    :param verbosity:
    0: Suppress all output
    1: Show profiler output
    2: Show tensorflow log messages
    3: Show tensorflow function progress
    :param use_tracer: Whether to use the tensorflow tracer to get the time instead of the time module. Can be more
    accurate to use with gpus however it only stores the information of the last batch.
    :param skip_untrainable_layers: Whether to skip profiling layers that have no trainable parameters
    :return: An (exception,dict) tuple the exception slot is to detect if an exception has occurred and the dict
    contains the information. The dict has key=layer.name and value=dict with
    keys=(forward_pass_cost, gradient_calculation_cost, gradient_application_cost, loss_calculation_cost)
    """
    if tf.executing_eagerly() and use_tracer:
        raise Exception("Cannot use tracer with eager execution.")
    from tensorflow.python.keras.layers import InputLayer
    from tensorflow.python.keras.layers.merge import _Merge
    from tensorflow.python.keras.models import Model
    # Disable annoying tensorflow messages
    if verbosity < 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "999"
        # Supressing deprectation messages is not working
        import tensorflow.python.util.deprecation as deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False
    def log_msg(msg):
        if verbosity >= 1:
            print(msg)
    # Check device
    if device == "cpu":
        # This seems to be the most effective method. Other methods include using the tf.device("/cpu:0")
        # context manager or setting ConfigProto(device_count={'GPU':0}. But both these methods use the GPU a little bit
        # Update: This does not work for eager execution
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        log_msg("Running on CPU")
    elif device == "gpu":
        if not tf.test.is_gpu_available():
            raise Exception("No GPUs are available!. Change --device parameter to use cpu.")
        log_msg("Running on GPU")
    # Produce layer topological order
    topological_layer_order = list()
    traverse_keras_DFS(input_model, topological_layer_order.append, order="post-order", top_to_bottom=True)
    topological_layer_order.reverse()
    # Initialize the costs that we will profile
    accumulative_costs = list()
    global_func_args = dict(verbose=0)
    # This block is the mother of all assumptions. It is the root of all evil
    accumulative_costs.append(dict(name="predict", func="predict",
                                   args=dict(x=None, steps=num_of_batches)))
    accumulative_costs.append(dict(name="evaluate", func="evaluate",
                                   args=dict(x=None, y=None, steps=num_of_batches)))
    accumulative_costs.append(dict(name="fit", func="fit",
                                   args=dict(x=None, y=None, steps_per_epoch=num_of_batches)))
    # Initialize timings dictionary
    timings = dict()
    for original_layer in topological_layer_order:
        if isinstance(original_layer, InputLayer):
            continue
        timings[original_layer.name] = {"Type": type(original_layer).__name__}
        if tf.executing_eagerly():
            timings[original_layer.name]["data_generation_cost"] = list()
        for cost in accumulative_costs:
            if use_tracer:
                timings[original_layer.name][cost["name"]] = dict()
                timings[original_layer.name][cost["name"]]["durations"] = list()
                timings[original_layer.name][cost["name"]]["total_time_costs"] = list()
            else:
                timings[original_layer.name][cost["name"]] = list()
    # Build and profile model layer by layer using the topological order
    log_msg("Start profiling...")
    try:
        for trial in range(trials):
            session_context = contextlib.nullcontext() if tf.executing_eagerly() else tf.Session()
            device_context = tf.device("/{}:0".format(device))
            if not tf.executing_eagerly():
                tf.reset_default_graph()
            log_msg("Trial: {}".format(trial))
            input_layers = list()
            added_layers = dict()
            with device_context, session_context:
                log_msg("Executing eagerly: {}".format(tf.executing_eagerly()))
                for original_layer in topological_layer_order:
                    # Add new layer to cloned network ------------------------------------------------------------------
                    # Get the layer's parents to check what the cloned layer needs to connect to
                    original_parents = get_layer_parents(original_layer)
                    # Remove all connections of this layer by making a clone using properties only
                    current_layer = clone_layer(original_layer)
                    # Restore appropriate connections in cloned network
                    if len(original_parents) == 0:
                        assert isinstance(current_layer, InputLayer)
                        # This is an input layer. It is only used for structuring purposes no need to actually profile
                        # it. Therefore we add it and continue
                        input_layers.append(current_layer)
                        added_layers[current_layer.name] = current_layer
                        continue
                    cloned_parents = set()
                    try:
                        for original_parent in original_parents:
                            cloned_parents.add(added_layers[original_parent.name])
                    except KeyError:
                        raise Exception("Attempting to add layer before one of its parents")
                    assert len(cloned_parents) == len(original_parents)
                    if len(cloned_parents) > 1:
                        # Merge all parents into this layer
                        # https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L328
                        # https://keras.io/layers/merge/
                        if not isinstance(current_layer, _Merge):
                            raise Exception("Non merge layer should not have multiple parents")
                        current_layer([x.output for x in cloned_parents])
                    elif len(cloned_parents) == 1:
                        # Make a connection to the only parent
                        current_layer(cloned_parents.pop().output)
                    added_layers[current_layer.name] = current_layer
                    # Should we go on with profiling ?
                    if skip_untrainable_layers and current_layer.count_params() == 0:
                        log_msg("Skip profiling of {}".format(current_layer.name))
                        continue
                    # Find output layers. Output layers need to be updated with every iteration since we are building
                    # the model from a top to bottom fashion.
                    output_layers = list()
                    for layer in added_layers.values():
                        if layer not in input_layers and len(get_layer_children(layer)) == 0:
                            output_layers.append(layer)
                    # Create and compile model -------------------------------------------------------------------------
                    # Inputs & outputs must be tensors not layers
                    cloned_model = Model(inputs=[x.input for x in input_layers],
                                         outputs=[x.output for x in output_layers])
                    if use_tracer:
                        run_metadata = tf.RunMetadata()
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        cloned_model.compile(optimizer=optimizer, loss=loss, run_metadata=run_metadata,
                                             options=run_options)
                    else:
                        cloned_model.compile(optimizer=optimizer, loss=loss)
                    # Start profiling ----------------------------------------------------------------------------------
                    print_format = "[{}:{:4}/{:<4}] Layer: {:16} {{key:30}}: {{value}}".format(
                        input_model.name, len(cloned_model.layers), len(input_model.layers), current_layer.name)
                    # Generate input output data
                    t = time.time_ns()
                    input_data, output_data = get_dummy_input_output(cloned_model, batch_size)
                    tc = time.time_ns() - t
                    if tf.executing_eagerly():
                        # print(input_data[0].device)
                        timings[current_layer.name]["data_generation_cost"].append(tc)
                        log_msg(print_format.format(key="data_generation_cost", value=tc))
                    for i, cost in enumerate(accumulative_costs):
                        name = cost["name"]
                        func = getattr(cloned_model, cost["func"])
                        args = cost["args"]
                        # Substitute in the input and output data
                        if "x" in args.keys():
                            args["x"] = input_data
                        if "y" in args.keys():
                            args["y"] = output_data
                        # Call the function and record the time
                        rmd = run_metadata if use_tracer else None
                        run_cost = RunCost(run_metadata=rmd)
                        func(**args, **global_func_args, callbacks=[run_cost])
                        v = run_cost.durations if not use_tracer else run_cost.durations, run_cost.total_time_costs
                        log_msg(print_format.format(key=name, value=v))
                        if use_tracer:
                            timings[current_layer.name][name]["durations"].extend(run_cost.durations)
                            timings[current_layer.name][name]["total_time_costs"].extend(run_cost.total_time_costs)
                        else:
                            timings[current_layer.name][name].extend(run_cost.durations)
                    # To confirm that the graph is being reset with each iteration
                    # print(len(sess.graph.get_operations()))
    except BaseException as e:
        return e, timings
    return None, timings


def full_model_layerwise_costs_profiling(input_model, loss, optimizer, batch_size=32, num_of_batches=8, trials=1,
                                         verbosity=1, device="gpu", skip_untrainable_layers=False):
    """
    The function takes a keras model and attempts to estimate the cost of each layer in the model in terms of
    the forward pass and backward pass.
    :return: An (exception,dict) tuple the exception slot is to detect if an exception has occurred and the dict
    contains the information. The dict has key=layer.name and value=dict with
    keys=(forward_pass_cost, gradient_calculation_cost, gradient_application_cost, loss_calculation_cost)
    """



def layer_input_output_profiling(model):
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


def join_reports(list_of_files):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that profiles Keras models and produces layer wise timings.")
    parser.add_argument("model",
                        help="The Keras model to profile. See Keras.Applications documentation for all options."
                             "Dummy options include: [dummy_multi, dummy_linear_dense, dummy_linear_cnn, dummy_2_layers]")
    parser.add_argument("-l", "--loss", default="binary_crossentropy",
                        help="The loss function to use. See Keras.Losses documentation for all options.")
    parser.add_argument("--eager", default=False, action="store_true",
                        help="Whether or not to enable tensorflow eager execution")
    parser.add_argument("-o", "--optimizer", default="sgd",
                        help="The optimizer to use when training. See Keras.Optimizers documentation for all options.")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="The device to use for all operations. If none is specified then it is automated.")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="The batch size used for all functions")
    parser.add_argument("-nb", "--num_of_batches", type=int, default=8,
                        help="The number of batches to run")
    parser.add_argument("-t", "--trials", type=int, default=5,
                        help="The number of layer building & evaluation iterations to do.")
    parser.add_argument("--skip", default=False, action="store_true",
                        help="Whether or not to skip layers with no trainable parameters")
    parser.add_argument("-upt", "--use_python_timing", default=False, action="store_true",
                        help="Whether or not to use the tensorflow tracer to calculate the time costs.")
    parser.add_argument("--out",
                        help="Stream to write the timings to in json format if any. File | stdout | stderr | suppress")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="0: Suppress all output\n"
                             "1: Show profiler output\n"
                             "2: Show tensorflow log messages\n"
                             "3: Show tensorflow function progress")
    args = parser.parse_args()
    if args.eager:
        tf.enable_eager_execution()
    try:
        model_func_name = "{}_model".format(args.model)
        if model_func_name in dir(sys.modules[__name__]):
            module = sys.modules[__name__]
            model = getattr(module, model_func_name)()
        else:
            module = __import__("tensorflow.keras.applications", fromlist=[args.model])
            model = getattr(module, args.model)
            model = model(weights=None, include_top=True)
    except AttributeError:
        raise Exception("'{}' is not a valid dummy or keras model.\n".format(args.model))
    exception, timings = model_reconstruct_layerwise_costs_profiling(input_model=model, batch_size=args.batch_size,
                                                                     num_of_batches=args.num_of_batches, loss=args.loss,
                                                                     optimizer=args.optimizer, device=args.device,
                                                                     verbosity=args.verbosity, trials=args.trials,
                                                                     use_tracer=not args.use_python_timing,
                                                                     skip_untrainable_layers=args.skip)
    if exception is not None:
        if isinstance(exception, KeyboardInterrupt):
            print("Profiling stopped by user. Attempting to write gathered data...")
            exception = None
        else:
            print("Unexpected error occurred. Attempting to write gathered data...")
    if args.out is not None:
        if args.out == "stdout":
            out = sys.__stdout__
        elif args.out == "stderr":
            out = sys.__stderr__
        elif args.out == "suppress":
            out = False
        else:
            out = open(args.out, "w")
    else:
        out = open("{}_{}.timings.json".format(args.model, datetime.now().strftime("%m-%d-%H-%M")), "w")
    report = {"host": socket.gethostname(), "args": args.__dict__, "timings": timings}
    json.dump(report, out, indent=4)
    if exception is not None:
        raise exception
