"""
A script for getting layer wise timings about any Keras model using model reconstruction.

The model reconstruction idea proposed by Marco is done as follows, we simply take an existing model with N layers, and
then construct a model with the first layer, we run the functions and record the time costs, we then add the second
layer and so on untill we have added all N layers.
he main intuition behind this idea was that if we had a model with 5 layers for example, [1,2,3,4,5] the difference in
cost between the model with layers [1] and [1,2] will give us the cost of layer 2 in the final model. And the difference
between [1,2] and [1,2,3] will give us the cost of layer 3 and so on.

This script relies on the keras_utils file. Including that file in the same directory when running this script will do
the trick.
"""

import argparse
import sys
from datetime import datetime
import socket
import os
import contextlib
import tensorflow as tf
import time
import json
sys.path.append("..")
from model_extraction.tensorflow_utils import *


def profile(input_model, loss, optimizer, batch_size=32, num_of_batches=8, trials=1, verbosity=1, device="gpu",
            use_tracer=True, skip_untrainable_layers=False):
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
    costs = list()
    global_func_args = dict(verbose=verbosity-2)
    costs.append(dict(name="predict", func="predict",
                      args=dict(x=None, steps=num_of_batches)))
    costs.append(dict(name="evaluate", func="evaluate",
                      args=dict(x=None, y=None, steps=num_of_batches)))
    costs.append(dict(name="fit", func="fit",
                      args=dict(x=None, y=None, steps_per_epoch=num_of_batches)))
    # Initialize timings dictionary
    timings = dict()
    for original_layer in topological_layer_order:
        if isinstance(original_layer, InputLayer):
            continue
        timings[original_layer.name] = {"Type": type(original_layer).__name__}
        if tf.executing_eagerly():
            timings[original_layer.name]["data_generation_cost"] = list()
        for cost in costs:
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
                    for i, cost in enumerate(costs):
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
    parser.add_argument("-bs", "--batch_size", type=int, default=8,
                        help="The batch size used for all functions")
    parser.add_argument("-nb", "--num_of_batches", type=int, default=8,
                        help="The number of batches to run")
    parser.add_argument("-t", "--trials", type=int, default=2,
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
    model = get_model(args.model)
    script_time_stamp =  datetime.now().strftime("%m-%d-%H-%M")
    t = time.time_ns()
    exception, timings = profile(input_model=model, batch_size=args.batch_size,
                                 num_of_batches=args.num_of_batches, loss=args.loss,
                                 optimizer=args.optimizer, device=args.device,
                                 verbosity=args.verbosity, trials=args.trials,
                                 use_tracer=not args.use_python_timing,
                                 skip_untrainable_layers=args.skip)
    t = time.time_ns() - t
    if args.verbosity >= 1:
        print("Finished in {} ms".format(t / 1e6))
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
        out = open("{}_{}.profile.json".format(args.model, script_time_stamp), "w")
    report = {"method": "tensorflow_model_reconstruction", "host": socket.gethostname(),
              "report_date": script_time_stamp, "profiling_time": t, "unit": "ns", "args": args.__dict__,
              "timings": timings}
    json.dump(report, out, indent=4)
    if exception is not None:
        raise exception
