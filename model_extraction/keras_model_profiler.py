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
import importlib


def dummy_multi_model():
    """
    https://keras.io/getting-started/functional-api-guide/
    :return: A multi input multi output model for testing
    """
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
    model = Model(name="Dummy", inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
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

    def traverse(root: Layer, visited: set):
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
    return deserialize(serialize(layer))


def generate_dummy_data(model, num_of_samples):
    # Keras gives us a list of shapes in case of multiple inputs / outputs
    model_input_shapes = [model.input_shape] if model.input_shape[0] is None else model.input_shape
    model_output_shapes = [model.output_shape] if model.output_shape[0] is None else model.output_shape
    input_shapes = list()
    for shape in [list(input_shape) for input_shape in model_input_shapes]:
        shape[0] = num_of_samples
        input_shapes.append(shape)
    input_data = [np.random.rand(*shape) for shape in input_shapes]
    output_shapes = list()
    for shape in [list(output_shape) for output_shape in model_output_shapes]:
        shape[0] = num_of_samples
        output_shapes.append(shape)
    output_data = [np.random.rand(*shape) for shape in output_shapes]
    return input_data, output_data


def profile(input_model, num_of_samples, loss, optimizer, full_profiling=False, log_stream=None):
    """
    The function takes a model and profiles the cost that each layer contributes to the total training time.
    The cost can be separated into 4 categories:
    1- Forward pass (Prediction)
    2- Calculate loss
    3- Calculate gradient (Taking the derivative of 1, 2)
    4- Apply the gradient (Taking the calculated gradient and applying it using an optimizer)
    :param input_model: The model to profile
    :param num_of_samples: The number of samples to use in profiling. The higher the more computation & accuracy.
    :param loss: The loss function for the model
    :param optimizer: The optimizer used for the model
    :param full_profiling: Include step 2 and 4 in profiling
    :param log_stream: A stream to direct the output of the profiling. Useful to keep track of the current step.
    If none then no logging happens
    :return: A dict with key=layer.name and value=dict with keys=(forward_pass_cost, gradient_calculation_cost,
    gradient_application_cost, loss_calculation_cost)
    """
    # Warm up run (To set up all backend variables and stuff)
    if log_stream:
        print("Running warm up...")
    input_data, output_data = generate_dummy_data(input_model, 1000)
    input_model.compile(optimizer=optimizer, loss=loss)
    input_model.fit(input_data, output_data, verbose=0)

    timings = dict()
    # Produce layer topological order
    topological_layer_order = list()
    traverse_keras_DFS(input_model, topological_layer_order.append, order="post-order", top_to_bottom=True)
    topological_layer_order.reverse()
    # Build and profile model layer by layer using the topological order
    if log_stream:
        print("Start profiling...")
    input_layers = list()
    added_layers = dict()
    previous_iteration_cost = {"forward_pass_cost": 0, "loss_calculation_cost": 0, "gradient_calculation_cost": 0,
                               "gradient_application_cost": 0}
    for original_layer in topological_layer_order:
        # Add new layer to cloned network ------------------------------------------------------------------------------
        # Flatten out the layer's parents to check what the cloned layer needs to connect to
        original_parents = set()
        for original_node in original_layer._inbound_nodes:
            for original_parent in original_node.inbound_layers:
                original_parents.add(original_parent)
        # Remove all connections of this layer by making a deep clone using properties only
        current_layer = clone_layer(original_layer)
        # Restore appropriate connections in cloned network
        if len(original_parents) == 0:
            # This is an input layer. It is only used for structuring purposes no need to actually profile it. Therefore
            # We add it and continue
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
        # Find output layers. Output layers need to be updated with every iteration since we are building the model from
        # A top to bottom fashion.
        output_layers = list()
        for layer in added_layers.values():
            if layer not in input_layers and len(layer._outbound_nodes) == 0:
                output_layers.append(layer)
        # Create and compile model -------------------------------------------------------------------------------------
        # Inputs & outputs must be tensors not layers
        cloned_model = Model(inputs=[x.input for x in input_layers], outputs=[x.output for x in output_layers])
        cloned_model.compile(optimizer=optimizer, loss=loss)
        # Create dummy data that suits the current input and output layers ---------------------------------------------
        input_data, output_data = generate_dummy_data(cloned_model, num_of_samples)
        # Start profiling ----------------------------------------------------------------------------------------------
        timings[current_layer.name] = dict()
        current_iteration_cost = dict()
        model = cloned_model
        print_format = "[{}:{:4}/{:<4}] Layer: {:16} {{key:30}}: {{value}}\n".format(
            input_model.name, len(model.layers), len(input_model.layers), current_layer.name)
        s0 = time.time_ns()
        # Step 1: Predict and record <time> (Gives us forward_pass_cost)
        model.predict(x=input_data, batch_size=num_of_samples, verbose=0)
        s1 = time.time_ns()
        current_iteration_cost["forward_pass_cost"] = s1 - s0
        timings[current_layer.name]["forward_pass_cost"] =\
            current_iteration_cost["forward_pass_cost"] - previous_iteration_cost["forward_pass_cost"]
        if log_stream:
            log_stream.write(print_format.format(
                key="forward_pass_cost", value=timings[current_layer.name]["forward_pass_cost"]))
        # Step 2: Evaluate and record <time - Step 1> (Gives us loss_calculation_cost)
        if full_profiling:
            model.evaluate(x=input_data, y=output_data, batch_size=num_of_samples, verbose=0)
            s2 = time.time_ns()
            current_iteration_cost["loss_calculation_cost"] = s2 - s1
            timings[current_layer.name]["loss_calculation_cost"] = \
                current_iteration_cost["loss_calculation_cost"] - previous_iteration_cost["loss_calculation_cost"]
            if log_stream:
                log_stream.write(print_format.format(
                    key="loss_calculation_cost", value=timings[current_layer.name]["loss_calculation_cost"]))
        # Step 3: Train using batch_size = samples and record <time - Step 1 - Step 2>
        # (Gives us gradient_calculation_cost)
        model.fit(x=input_data, y=output_data, batch_size=num_of_samples, verbose=0)
        s3 = time.time_ns()
        current_iteration_cost["gradient_calculation_cost"] = s3 - s1 if not full_profiling else s3 - s2
        timings[current_layer.name]["gradient_calculation_cost"] = \
            current_iteration_cost["gradient_calculation_cost"] - previous_iteration_cost["gradient_calculation_cost"]
        if log_stream:
            log_stream.write(print_format.format(
                key="gradient_calculation_cost", value=timings[current_layer.name]["gradient_calculation_cost"]))
        # Step 4: Train using batch_size = 1 and record <time - Step 1 - Step 2 - Step 3>
        # (Gives us gradient_application_cost)
        if full_profiling:
            model.fit(x=input_data, y=output_data, batch_size=1, verbose=0)
            s4 = time.time_ns()
            current_iteration_cost["gradient_application_cost"] = s4 - s3
            timings[current_layer.name]["gradient_application_cost"] = \
                current_iteration_cost["gradient_application_cost"] -\
                previous_iteration_cost["gradient_application_cost"]
            if log_stream:
                log_stream.write(print_format.format(
                    key="gradient_application_cost", value=timings[current_layer.name]["gradient_application_cost"]))
        previous_iteration_cost = current_iteration_cost
    return timings


def visualize_timing_profile(report):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that profiles Keras models and produces layer wise timings.")
    parser.add_argument("model",
                        help="The Keras model to profile. See Keras.Applications documentation for all options."
                             "Entering 'dummy' will use a small dummy model for debugging.")
    parser.add_argument("-n", "--samples", type=int, default=100,
                        help="The number of samples to run for each cost evaluation. The higher the more accurate and "
                             "the more computation time needed.")
    parser.add_argument("-l", "--loss", default="binary_crossentropy",
                        help="The loss function to use. See Keras.Losses documentation for all options.")
    parser.add_argument("-o", "--optimizer", default="sgd",
                        help="The optimizer to use when training. See Keras.Optimizers documentation for all options.")
    parser.add_argument("-fb", "--full_profiling", action="store_true",
                        help="If set to true then loss calculation and gradient application will be included in "
                             "profiling")
    parser.add_argument("--out",
                        help="Stream to write the timings to in json format if any. File | stdout | stderr")
    parser.add_argument("--log",
                        help="Stream to write status messages to if any. File | stdout | stderr")
    args = parser.parse_args()
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
    if not args.model == "dummy":
        module = __import__("keras.applications", fromlist=[args.model])
        model = getattr(module, args.model)
        model = model(weights=None, include_top=True)
    import keras
    import numpy as np
    import time
    import json
    from keras.layers import Layer, serialize, deserialize, InputLayer, Input, Embedding, LSTM, Dense
    from keras.layers.merge import _Merge
    from keras.models import Model
    if args.model == "dummy":
        model = dummy_multi_model()
    timings = profile(input_model=model, num_of_samples=args.samples, loss=args.loss, optimizer=args.optimizer,
                      full_profiling=args.full_profiling, log_stream=log)
    report = {"args": args.__dict__, "timings": timings}
    json.dump(report, out, indent=4)
