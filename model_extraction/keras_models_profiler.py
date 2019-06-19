import keras
import numpy as np
import sys
import time
import json
from keras.applications import VGG19, ResNet50, InceptionV3, DenseNet201
from keras.layers import Layer, serialize, deserialize, InputLayer, Input, Embedding, LSTM, Dense
from keras.layers.merge import _Merge
from keras.models import Model, Sequential


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
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
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


def traverse_DFS(model: Model, processing_function, order="post-order", top_to_bottom=True):
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


def clone_layer(layer: Layer):
    """
    Creates a clone of the layer using only its main properties (Does not copy any connections from the clone)
    Better implementation exists ?
    """
    return deserialize(serialize(layer))


def profile(input_model: Model, num_of_samples, loss, optimizer, output_stream=None, log_stream=None):
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
    :param output_stream: A stream to write the results to with every step. Useful to keep data upon crashes.
    If none then no output is written.
    :param log_stream: A stream to direct the output of the profiling. Useful to keep track of the current step.
    If none then no logging happens
    :return: A dict with key=layer.name and value=(forward_pass_cost, gradient_calculation_cost,
    gradient_application_cost)
    as well as a special entry with key="loss_calculation_cost" value=cost since this is layer independent
    """
    sys.stdout = log_stream
    timings = dict()
    timings["loss_calculation_cost"] = 0
    # Produce layer topological order
    topological_layer_order = list()
    traverse_DFS(input_model, topological_layer_order.append, order="post-order", top_to_bottom=True)
    topological_layer_order.reverse()
    # Build and profile model layer by layer using the topological order
    input_layers = list()
    added_layers = dict()
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
            has_child = False
            for node in layer._outbound_nodes:
                child = node.outbound_layer
                if child is not None:
                    has_child = True
                    break
                else:
                    print("Correct way to identify the existance of a node child. You can remove this now")
            if not has_child:
                output_layers.append(current_layer)
        # Create and compile model -------------------------------------------------------------------------------------
        # Inputs must be tensors not layers
        cloned_model = Model(inputs=[x.input for x in input_layers], outputs=[x.output for x in output_layers])
        cloned_model.compile(optimizer=optimizer, loss=loss)
        # Create dummy data that suits the current input and output layers ---------------------------------------------
        input_shapes = list()
        for shape in [list(input_layer.input_shape) for input_layer in input_layers]:
            shape[0] = num_of_samples
            input_shapes.append(shape)
        input_data = [np.random.rand(*shape) for shape in input_shapes]
        output_shapes = list()
        for shape in [list(output_layer.output_shape) for output_layer in output_layers]:
            shape[0] = num_of_samples
            output_shapes.append(shape)
        output_data = [np.random.rand(*shape) for shape in output_shapes]
        # Start profiling ----------------------------------------------------------------------------------------------
        verbose = 0 if log_stream is None else 1
        timings[current_layer.name] = dict()
        model = cloned_model
        if verbose:
            print("ts: {} ] Profiling {} with {}/{} layers".format(
                time.time_ns(), model.name, len(model.layers), len(input_model.layers)))
            model.summary()
        s0 = time.time_ns()
        # Step 1: Predict and record <time> (Gives us forward_pass_cost)
        if verbose:
            print("ts: {} ] Profiling forward_pass_cost".format(s0))
        model.predict(x=input_data, batch_size=num_of_samples, verbose=verbose)
        s1 = time.time_ns()
        # Step 2: Evaluate and record <time - Step 1> (Gives us loss_calculation_cost)
        if verbose:
            print("ts: {} ] Profiling loss_calculation_cost".format(s1))
        model.evaluate(x=input_data, y=output_data, batch_size=num_of_samples, verbose=verbose)
        s2 = time.time_ns()
        # Step 3: Train using batch_size = samples and record <time - Step 1 - Step 2>
        # (Gives us gradient_calculation_cost)
        if verbose:
            print("ts: {} ] Profiling gradient_calculation_cost".format(s2))
        model.fit(x=input_data, y=output_data, batch_size=num_of_samples, verbose=verbose)
        s3 = time.time_ns()
        # Step 4: Train using batch_size = 1 and record <time - Step 1 - Step 2 - Step 3>
        # (Gives us gradient_application_cost)
        if verbose:
            print("ts: {} ] Profiling gradient_application_cost".format(s3))
        model.fit(x=input_data, y=output_data, batch_size=1, verbose=verbose)
        s4 = time.time_ns()
        if verbose:
            print("ts: {} ] Finished {} with {}/{} layers".format(
                time.time_ns(), model.name, len(model.layers), len(input_model.layers)))
        # Write timings ------------------------------------------------------------------------------------------------
        timings[current_layer.name]["forward_pass_cost"] = s1-s0
        timings["loss_calculation_cost"] += s2 - s1
        timings[current_layer.name]["gradient_calculation_cost"] = s3-s2
        timings[current_layer.name]["gradient_application_cost"] = s4-s3
        if output_stream is not None:
            json.dump(timings, output_stream)
    timings["loss_calculation_cost"] /= len(input_model.layers)
    if output_stream is not None:
        json.dump(timings, output_stream)
    return timings
