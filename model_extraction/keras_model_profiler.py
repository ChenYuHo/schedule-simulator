"""
A script for getting layer wise timings of any Keras model. It uses an inefficient approach where the model is built
layer by layer and the timing difference introduced by each layer is assumed to be that layer's cost.

Tested versions:
keras                     2.2.4                         0
keras-applications        1.0.8                      py_0
keras-base                2.2.4                    py37_0
keras-preprocessing       1.1.0                      py_1
tensorflow                1.13.1          mkl_py37h54b294f_0
tensorflow-base           1.13.1          mkl_py37h7ce6ba3_0
tensorflow-estimator      1.13.0                     py_0
"""

import argparse
import sys
from datetime import datetime
import json
import socket
import os
import time


def dummy_multi_model():
    """
    https://keras.io/getting-started/functional-api-guide/
    :return: A multi input multi output model for testing
    """
    import keras
    from keras.layers import Input, Embedding, LSTM, Dense
    from keras.models import Model
    main_input = Input(shape=(100,), dtype='int32', name='main_input')
    x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
    lstm_out = LSTM(32)(x)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
    auxiliary_input = Input(shape=(5,), name='aux_input')
    x = keras.layers.concatenate([lstm_out, auxiliary_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(name="dummy_multi", inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    return model


def dummy_linear_dense_model(units=100, n_of_layers=10):
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential(name="dummy_linear_dense")
    model.add(Dense(units, input_shape=(units,)))
    for _ in range(n_of_layers-1):
        model.add(Dense(units))
    return model


def dummy_linear_cnn_model():
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
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
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D
    model = Sequential(name="dummy_2_layers")
    model.add(Conv2D(8, 16, padding="same", input_shape=(256, 256, 1), activation="relu"))
    model.add(Dense(64))
    return model


def init_models(models, insight=True):
    for i, model_fun in enumerate(models):
        models[i] = model_fun(weights=None, include_top=True)
        if insight:
            print("Model: {}".format(models[i].name))
            print("Inputs: {}".format(models[i]._input_layers))
            print("Outputs: {}".format(models[i]._output_layers))
            print("Num of layers: {}".format(len(models[i].layers)))
            print("Num of nodes: {}".format(len(models[i]._nodes_by_depth)))
            print("Num of parameters: {}".format(models[i].count_params()))


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
        # Note that the Keras model structure uses a layer node layer node scheme.
        # A `Node` describes the connectivity between two layers See
        # https://github.com/keras-team/keras/blob/b0bfd5201da2bfced84028bcc5bda05bdfd75af7/keras/engine/base_layer.py#L1178
        # Regardless, for our purpose, it is only important to know that a layer can have multiple inbound nodes and
        # each node can have multiple inbound layers. Similarly, a layer can have multiple outbound nodes however each
        # node can have one outbound layer only.
        if top_to_bottom:
            for node in root._outbound_nodes:
                child_layer = node.outbound_layer
                if child_layer not in visited:
                    traverse(root=child_layer, visited=visited)
        else:
            for node in root._inbound_nodes:
                for parent_layer in node.inbound_layers:
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
    from keras.layers import serialize, deserialize
    return deserialize(serialize(layer))


def get_dummy_input_output(model, num_of_samples):
    import keras.backend as K
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
    # Which op takes less time depends on whether you are using gpu or not and whether 
    op = K.random_uniform if random else K.ones
    input_data = [op(shape=shape) for shape in input_shapes]
    output_data = [op(shape=shape) for shape in output_shapes]
    return input_data, output_data


def timing_profile(input_model, loss, optimizer, batch_size=32, num_of_batches=8, trials=1, num_of_function_calls=5,
                   full_profiling=False, suppress_negatives=False, log_stream=None, device="gpu", accumulative=True):
    """
    The function takes a model and profiles the cost that each layer contributes to the total training time.
    The function's method is to build the model layer by layer and observe the cost changes each layer introduces. The
    change is the assumed to be that layer's cost.
    The cost can be separated into 4 categories:
    1- Forward pass (Prediction)
    2- Calculate loss
    3- Calculate gradient (Taking the derivative of 1, 2)
    4- Apply the gradient (Taking the calculated gradient and applying it using an optimizer)
    :param input_model: The model to profile
    :param batch_size: The batch size used for all functions.
    :param num_of_batches: The number of batches to run
    :param trials: The number of layer building & evaluation iterations to do.
    :param num_of_function_calls: The number of function calls to make before taking and recording the minimum cost.
    Only the minimum cost of these calls is added to the report.
    :param device: cpu or gpu ?
    :param accumulative: Store the raw costs of each function call do not subtract model L+1 from model L.
    Used for debugging
    :param loss: The loss function for the model
    :param optimizer: The optimizer used for the model
    :param full_profiling: Include step 2 and 4 in profiling
    :param suppress_negatives: Whether we should set a cost to 0 if it is negative.
    :param log_stream: A stream to direct the output of the profiling. Useful to keep track of the current step.
    If none then no logging happens
    :return: An (exception,dict) tuple the exception slot is to detect if an exception has occurred and the dict
    contains the information. The dict has key=layer.name and value=dict with
    keys=(forward_pass_cost, gradient_calculation_cost, gradient_application_cost, loss_calculation_cost)
    """
    import keras
    from keras.layers import InputLayer
    from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
    from keras.layers.merge import _Merge
    from keras.models import Model


    ignored_layer_types = []
    # ignored_layer_types = [_Pooling1D, _Pooling2D, _Pooling3D]  # Uncomment to ignore pooling layers
    # Check device
    if device == "cpu":
        # This seems to be the most effective method. Other methods include using the tf.device("/cpu:0")
        # context manager or setting ConfigProto(device_count={'GPU':0}. But both these methods use the GPU a little bit
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if log_stream:
            log_stream.write("Running on CPU\n")
    elif device == "gpu":
        import tensorflow as tf
        if not tf.test.is_gpu_available():
            raise Exception("No GPUs are available!. Change --device parameter to use cpu.")
        if log_stream:
            log_stream.write("Running on GPU\n")
    # Produce layer topological order
    topological_layer_order = list()
    traverse_keras_DFS(input_model, topological_layer_order.append, order="post-order", top_to_bottom=True)
    topological_layer_order.reverse()
    # Initialize the costs that we will profile
    accumulative_costs = list()
    global_func_args = dict(verbose=0)
    # This block is the mother of all assumptions. It is the root of all evil
    if full_profiling:
        accumulative_costs.append(dict(name="predict", func="predict", batch_size=batch_size,
                                       args=dict(x=None, steps=num_of_batches)))
        accumulative_costs.append(dict(name="evaluate", func="evaluate", batch_size=batch_size,
                                       args=dict(x=None, y=None, steps=num_of_batches)))
        accumulative_costs.append(dict(name="fit", func="fit", batch_size=batch_size,
                                       args=dict(x=None, y=None, steps_per_epoch=num_of_batches)))
        # accumulative_costs.append(dict(name="gradient_calculation_cost", func="fit", batch_size=batch_size,
        #                                args=dict(x=None, y=None, steps_per_epoch=num_of_batches)))
        # accumulative_costs.append(dict(name="gradient_application_cost", func="fit", batch_size=1,
        #                                args=dict(x=None, y=None, steps_per_epoch=num_of_batches*batch_size)))
    else:
        accumulative_costs.append(dict(name="forward_pass_cost", func="evaluate", batch_size=batch_size,
                                       args=dict(x=None, y=None, steps=num_of_batches)))
        accumulative_costs.append(dict(name="backward_pass_cost", func="fit", batch_size=batch_size,
                                       args=dict(x=None, y=None, steps_per_epoch=num_of_batches)))
    # Initialize timings dictionary
    timings = dict()
    for original_layer in topological_layer_order:
        if isinstance(original_layer, InputLayer):
            continue
        timings[original_layer.name] = {"Type": type(original_layer).__name__}
        for cost in accumulative_costs:
            timings[original_layer.name][cost["name"]] = list()
    # Build and profile model layer by layer using the topological order
    if log_stream:
        log_stream.write("Start profiling...\n")
    try:
        for trial in range(trials):
            keras.backend.clear_session()
            if log_stream:
                log_stream.write("Trial: {}\n".format(trial))
            input_layers = list()
            added_layers = dict()
            previous_model_cost = None
            for original_layer in topological_layer_order:
                # Add new layer to cloned network ----------------------------------------------------------------------
                # Flatten out the layer's parents to check what the cloned layer needs to connect to
                original_parents = set()
                for original_node in original_layer._inbound_nodes:
                    for original_parent in original_node.inbound_layers:
                        original_parents.add(original_parent)
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
                ignore = False
                for layer_type in ignored_layer_types:
                    if isinstance(current_layer, layer_type):
                        ignore = True
                        break
                if ignore:
                    if log_stream:
                        log_stream.write("Skip profiling of {}\n".format(current_layer.name))
                    continue
                # Find output layers. Output layers need to be updated with every iteration since we are building
                # the model from a top to bottom fashion.
                output_layers = list()
                for layer in added_layers.values():
                    if layer not in input_layers and len(layer._outbound_nodes) == 0:
                        output_layers.append(layer)
                # Create and compile model -----------------------------------------------------------------------------
                # Inputs & outputs must be tensors not layers
                cloned_model = Model(inputs=[x.input for x in input_layers],
                                     outputs=[x.output for x in output_layers])
                cloned_model.compile(optimizer=optimizer, loss=loss)
                # Start profiling --------------------------------------------------------------------------------------
                current_model_cost = dict()
                for cost in accumulative_costs:
                    current_model_cost[cost["name"]] = float("inf")
                print_format = "[{}:{:4}/{:<4}] Layer: {:16} {{key:30}}: {{value}}\n".format(
                    input_model.name, len(cloned_model.layers), len(input_model.layers), current_layer.name)
                accumulated_cost = 0
                for i, cost in enumerate(accumulative_costs):
                    # These are symbolic tensors no computational load until we run the function
                    # With symbolic tensors the first dimension (Usually the number of samples) constitutes the
                    # batch size. We then specify how many batches we run using steps_per_epoch for the train
                    # function. and using steps for the evaluation & prediction functions
                    input_data, output_data = get_dummy_input_output(cloned_model, cost["batch_size"], random=True)
                    name = cost["name"]
                    func = getattr(cloned_model, cost["func"])
                    args = cost["args"]
                    # Substitute in the input and output data
                    if "x" in args.keys():
                        args["x"] = input_data
                    if "y" in args.keys():
                        args["y"] = output_data
                    # Call the function and record the time
                    for _ in range(num_of_function_calls):
                        t = time.time_ns()
                        func(**args, **global_func_args)
                        tc = time.time_ns() - t
                        if tc < current_model_cost[name]:
                            current_model_cost[name] = tc
                    if accumulative:
                        layer_actual_specific_cost = tc
                    else:
                        # The layer total function cost is the difference between the previous and current iteration cost
                        layer_total_func_cost = current_model_cost[name] -\
                                                (previous_model_cost[name] if previous_model_cost else 0)
                        # The layer actual cost to measure is the total function cost minus all the actual costs before it
                        layer_actual_specific_cost = layer_total_func_cost - accumulated_cost
                        if suppress_negatives and layer_actual_specific_cost < 0:
                            layer_actual_specific_cost = 0
                        accumulated_cost += layer_actual_specific_cost
                    if log_stream:
                        log_stream.write(print_format.format(key=name, value=layer_actual_specific_cost))
                    timings[current_layer.name][name].append(layer_actual_specific_cost)
                previous_model_cost = current_model_cost
                # To confirm that the graph is being reset with each iteration
                # print(len(keras.backend.get_session().graph.get_operations()))
            # The below line closes the session. I encounter OOM errors if i omit this and don't explicitly close the
            # session, and if i include it, i encounter CancelledError: Session has been closed. However the latter
            # error is ignored by tensorflow and execution continues normally.
            # Applying the recent fix below manually seems to get rid of the annoying ignored error
            # https://github.com/dansitu/tensorflow/commit/b124e055a17ca8e4ff897867f96c97e7bfc8eed7
            keras.backend.get_session().close()
    except BaseException as e:
        return e, timings
    return None, timings


def timing_profile2():
    pass


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


if __name__ == "__main__":
    import tensorflow as tf
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser(description="A script that profiles Keras models and produces layer wise timings.")
    parser.add_argument("model",
                        help="The Keras model to profile. See Keras.Applications documentation for all options."
                             "Dummy options include: [dummy_multi, dummy_linear_dense, dummy_linear_cnn, dummy_2_layers]")
    parser.add_argument("-l", "--loss", default="binary_crossentropy",
                        help="The loss function to use. See Keras.Losses documentation for all options.")
    parser.add_argument("-o", "--optimizer", default="sgd",
                        help="The optimizer to use when training. See Keras.Optimizers documentation for all options.")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="The device to use for all operations. If none is specified then it is automated.")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="The batch size used for all functions")
    parser.add_argument("-nb", "--num_of_batches", type=int, default=8,
                        help="The number of batches to run")
    parser.add_argument("-f", "--num_calls", type=int, default=2,
                        help="num_of_function_calls: The number of function calls to make before taking and recording "
                             "the minimum cost. Only the minimum cost of these calls is added to the report.")
    parser.add_argument("-t", "--trials", type=int, default=5,
                        help="The number of layer building & evaluation iterations to do.")
    parser.add_argument("-fp", "--full_profiling", action="store_true",
                        help="If set to true then loss calculation and gradient application will be included in "
                             "profiling")
    parser.add_argument("-sn", "--suppress_negatives", action="store_true",
                        help="If set to true costs that are negative are set to 0")
    parser.add_argument("-ac", "--accumulative", action="store_true",
                        help="If set to true then the accumulative costs are stored. No subtractions are done.")
    parser.add_argument("--out",
                        help="Stream to write the timings to in json format if any. File | stdout | stderr | suppress")
    parser.add_argument("--log",
                        help="Stream to write status messages to if any. File | stdout | stderr | suppress")
    args = parser.parse_args()
    if args.log is not None:
        if args.log == "stdout":
            log = sys.__stdout__
        elif args.log == "stderr":
            log = sys.__stderr__
        elif args.log == "suppress":
            log = False
        else:
            log = open(args.log, "w")
    else:
        log = sys.__stdout__
    try:
        model_func_name = "{}_model".format(args.model)
        if model_func_name in dir(sys.modules[__name__]):
            module = sys.modules[__name__]
            model = getattr(module, model_func_name)()
        else:
            module = __import__("keras.applications", fromlist=[args.model])
            model = getattr(module, args.model)
            model = model(weights=None, include_top=True)
    except AttributeError:
        msg = "'{}' is not a valid dummy or keras model.\n".format(args.model)
        if log:
            log.write(msg)
            sys.exit()
        else:
            print(msg)
    exception, timings = timing_profile(input_model=model, batch_size=args.batch_size, num_of_batches=args.num_of_batches,
                                        loss=args.loss, optimizer=args.optimizer, full_profiling=args.full_profiling,
                                        log_stream=log, num_of_function_calls=args.num_calls, trials=args.trials,
                                        suppress_negatives=args.suppress_negatives, device=args.device,
                                        accumulative=args.accumulative)
    if exception is not None:
        if isinstance(exception, KeyboardInterrupt):
            if log:
                log.write("Profiling stopped by user. Attempting to write gathered data...\n")
            exception = None
        else:
            if log:
                log.write("Unexpected error occurred. Attempting to write gathered data...\n")
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
