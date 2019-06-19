import tensorflow as tf
from tensorflow.python.keras.applications import VGG19, ResNet50, InceptionV3, DenseNet201
from tensorflow.python.keras.layers import Layer, serialize, deserialize
from tensorflow.python.keras.models import Model


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
            for node in root.outbound_nodes:
                child_layer = node.outbound_layer
                if child_layer not in visited:
                    traverse(root=child_layer, visited=visited)
        else:
            for node in root.inbound_nodes:
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
    return deserialize(serialize(layer))


def profile(input_model: Model, samples, loss, optimizer):
    """
    The function takes a model and profiles the cost that each layer contributes to the total training time.
    The cost can be separated into 4 categories:
    1- Forward pass (Prediction)
    2- Calculate loss
    3- Calculate gradient (Taking the derivative of 1, 2)
    4- Apply the gradient (Taking the calculated gradient and applying it using an optimizer)
    Training = 1-4
    Evaluation = 1-2
    Prediction = 1
    Forward pass = 1-2
    Backward pass = 3-4
    :param input_model: The model to profile
    :param samples: The number of samples to use in profiling
    :param loss: The loss function for the model
    :param optimizer: The optimizer used for the model
    :return: A dict with key=layer.name and value=(forward_pass_cost, gradient_calculation_cost,
    gradient_application_cost)
    as well as a special entry with key="loss_calculation_cost" value=cost since this is layer independent
    """
    # Produce layer topological order
    topological_layer_order = list()
    traverse_DFS(input_model, topological_layer_order.append, order="post-order", top_to_bottom=True)
    topological_layer_order.reverse()
    # Build and profile model layer by layer using the topological order
    input_layers = list()
    output_layers = list()
    added_layers = dict()
    for original_layer in topological_layer_order:
        cloned_layer = clone_layer(original_layer)
        # Add layer to appropriate place in network

        # Compile model

        # Start profiling
        # Step 1: Predict and record <time> (Gives us forward_pass_cost)

        # Step 2: Evaluate and record <time - Step 1> (Gives us loss_calculation_cost)

        # Step 3: Train using batch_size = samples and record <time - Step 1 - Step 2>
        # (Gives us gradient_calculation_cost)

        # Step 4: Train using batch_size = 1 and record <time - Step 1 - Step 2 - Step 3>
        # (Gives us gradient_application_cost)
        

model = VGG19(weights=None, include_top=True)