import torch
import torchvision.models as models
from schedule_simulator_core.DAGs import Layer, DAG, LOCAL_EXTRA_PREFIX
from model_extraction.pytorch_utils import *
import torch.jit as jit
import torch.onnx as onnx
import inspect


def pytorch_model_to_DAG(model):
    # Run the Pytorch graph to get a trace and generate a graph from it
    model_inputs, _ = get_dummy_input_output(model, 1)
    trace = torch.jit.trace(model, tuple(model_inputs), check_trace=False)
    torch_graph = trace.graph
    inputs = [i.unique() for i in torch_graph.inputs()]
    outputs = [o.unique() for o in torch_graph.outputs()]
    # The problem with the graph generated is that it is op level. Therefore we follow the following algorithm to reduce
    # it to a layer level graph
    # 1. Collapse ops into a layer dictionary with input and output lists of ids
    layers = dict()
    unmatched_links = dict()
    for op in torch_graph.nodes():
        # Inputs/outputs
        inputs = {i.unique() for i in op.inputs()}
        outputs = {o.unique() for o in op.outputs()}
        layer_name = get_module_name(op)
        module = get_module(model, layer_name)
        if is_parent_module(module):
            for inp in inputs:
                unmatched_links[inp] = outputs
            continue
        if layer_name in layers:
            layers[layer_name]["inputs"] = layers[layer_name]["inputs"].union(inputs)
            layers[layer_name]["outputs"] = layers[layer_name]["outputs"].union(outputs)
        else:
            layers[layer_name] = {"inputs": inputs, "outputs": outputs}
    print(layers)
    print(unmatched_links)
    # 2. Remove mutual identifiers within a layer (If an identifier is in the input and output of the same layer then we
    # remove it)
    for layer_name, layer_dict in layers.items():
        reduced_inputs = layer_dict["inputs"] - layer_dict["outputs"]
        reduced_outputs = layer_dict["outputs"] - layer_dict["inputs"]
        layer_dict["inputs"] = reduced_inputs
        layer_dict["outputs"] = reduced_outputs
    print(layers)
    print(unmatched_links)
    # 3. Resolve external identifiers to point to other layers

    # 4. Infer input and output layers


def extract_costs_from_profile():
    pass


def apply_layer_costs_to_dag():
    pass


def ins(obj):
    print("str       : {}".format(obj))
    print("Type      : {}".format(type(obj)))
    print("Module    : {}".format(inspect.getmodule(obj)))
    print("Attributes: {}".format(dir(obj)))


def dump_trace_graph(model):
    """
    https://github.com/waleedka/hiddenlayer/blob/master/hiddenlayer/pytorch_builder.py
    """
    # Run the Pytorch graph to get a trace and generate a graph from it
    model_inputs, _ = get_dummy_input_output(model, 1)
    trace = torch.jit.trace(model, tuple(model_inputs), check_trace=False)
    torch_graph = trace.graph
    inputs = [i.unique() for i in torch_graph.inputs()]
    outputs = [o.unique() for o in torch_graph.outputs()]
    print(model)
    print("Graph inputs : {}".format(inputs))
    print("Graph outputs: {}".format(outputs))
    for torch_node in torch_graph.nodes():
        scope = torch_node.scopeName()
        name = get_module_name(torch_node)
        # Op
        op = torch_node.kind()
        # Inputs/outputs
        inputs = [i.unique() for i in torch_node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]
        print("name: {:25} scope: {:35} Op: {:20} inputs: {:60} outputs: {:20}".format(name, scope, str(op), str(inputs), str(outputs)))


net = models.inception_v3(aux_logits=False)
# dump_trace_graph(net)
pytorch_model_to_DAG(net)