import torch
import torchvision.models as models
from schedule_simulator_core.DAGs import Layer, DAG, LOCAL_EXTRA_PREFIX
from model_extraction.pytorch_utils import *
import torch.jit as jit
import torch.onnx as onnx
import inspect


def pytorch_model_to_DAG(model, skip_untrainable_layers=False):
    """
    This method uses the jit trace to get the graph info of the model.
    :param model: The model to convert
    :param skip_untrainable_layers: Whether we skip layers with no trainable parameters
    :return: a simulator DAG object that represents this network
    """
    # Run the Pytorch graph to get a trace and generate a graph from it.
    model_inputs, _ = get_dummy_input_output(model, 2)
    trace = torch.jit.trace(model, tuple(model_inputs), check_trace=False)
    torch_graph = trace.graph
    inputs = [i.unique() for i in torch_graph.inputs()]
    outputs = [o.unique() for o in torch_graph.outputs()]
    # The problem with the graph generated is that it is op level. Therefore we follow the following algorithm to reduce
    # it to a layer level graph
    # 1. Collapse ops into a layer dictionary with input and output lists of ids
    layers = dict()
    input_links = dict()
    output_links = dict()
    for op in torch_graph.nodes():
        # Inputs/outputs
        inputs = {i.unique() for i in op.inputs()}
        outputs = {o.unique() for o in op.outputs()}
        layer_name = get_module_name_from_op(op)
        module = get_module(model, layer_name)
        trainable_params = count_trainable_params(module)
        if is_parent_module(module) or (skip_untrainable_layers and trainable_params == 0):
            for inp in inputs:
                input_links[inp] = outputs
            for out in outputs:
                output_links[out] = inputs
            continue
        if layer_name in layers:
            layers[layer_name]["inputs"] = layers[layer_name]["inputs"].union(inputs)
            layers[layer_name]["outputs"] = layers[layer_name]["outputs"].union(outputs)
        else:
            layers[layer_name] = {"inputs": inputs, "outputs": outputs, "trainable_params": trainable_params,
                                  "type": type(module).__name__}
    # 2. Remove mutual identifiers within a layer (If an identifier is in the input and output of the same layer then we
    # remove it)
    for layer_name, layer_dict in layers.items():
        reduced_inputs = layer_dict["inputs"] - layer_dict["outputs"]
        reduced_outputs = layer_dict["outputs"] - layer_dict["inputs"]
        layer_dict["inputs"] = reduced_inputs
        layer_dict["outputs"] = reduced_outputs
    # 3. Add unresolved layer links
    for layer_name, layer_dict in layers.items():
        for inp in layer_dict["inputs"]:
            if inp in input_links:
                input_links[inp].add(layer_name)
            else:
                input_links[inp] = {layer_name}
        for out in layer_dict["outputs"]:
            if out in output_links:
                output_links[out].add(layer_name)
            else:
                output_links[out] = {layer_name}

    # 4. Resolve external identifiers to point to other layers (Recursive algorithm)
    def follow_link(ids: set, use_input_links=True):
        new_ids = set()
        for idd in ids:
            if isinstance(idd, str):
                new_ids.add(idd)
            elif use_input_links:
                if idd in output_links:
                    st = follow_link(output_links[idd], use_input_links)
                    new_ids.update(st)
            else:
                if idd in input_links:
                    st = follow_link(input_links[idd], use_input_links)
                    new_ids.update(st)
        return new_ids
    for layer_name, layer_dict in layers.items():
        layer_dict["inputs"] = follow_link(layer_dict["inputs"], use_input_links=True)
        layer_dict["outputs"] = follow_link(layer_dict["outputs"], use_input_links=False)
    # 5. Infer input layers
    input_layers = set()
    for layer_name, layer_dict in layers.items():
        if len(layer_dict["inputs"]) == 0:
            input_layers.add(layer_name)
    # 6. Convert dicts into the simulator DAG object
    # 6.a Add layers
    sim_layers = dict()
    for layer_name, layer_dict in layers.items():
        sim_layers[layer_name] = Layer(0, 0, layer_dict["trainable_params"] * 4, name=layer_name,
                                       type=layer_dict["type"])
    # 6.b Connect layers
    for layer_name, sim_layer in sim_layers.items():
        sim_layer.input_layers = list()
        for input_layer_name in layers[layer_name]["inputs"]:
            sim_layer.input_layers.append(sim_layers[input_layer_name])
        sim_layer.output_layers = list()
        for output_layer_name in layers[layer_name]["outputs"]:
            sim_layer.output_layers.append(sim_layers[output_layer_name])
    sim_input_layers = list()
    for input_layer_name in input_layers:
        sim_input_layers.append(sim_layers[input_layer_name])
    # 6.c Create DAG object
    extraction_method = dict(library="pytorch", skip_untrainable_layers=skip_untrainable_layers)
    extras = {"comm_unit": "B", "forward_pass_unit": None, "backward_pass_unit": None,
              "extraction_info": extraction_method}
    prefixed_extras = dict()
    for k, v in extras.items():
        prefixed_extras[LOCAL_EXTRA_PREFIX+k] = v
    return DAG(dag_input_layers=sim_input_layers, name=type(model).__name__, **prefixed_extras)


def extract_costs_from_module_hooks_profile(profiling_report, reduce_func=None, skip_first_batch=False):
    if reduce_func is None:
        reduce_func = lambda x: min(x)
    layer_costs = profiling_report["layer_costs"]
    for layer_name, cost_dict in layer_costs.items():
        if skip_first_batch:
            cost_dict["forward_pass_units"].pop(0)
            cost_dict["backward_pass_units"].pop(0)
        layer_costs[layer_name]["forward_pass_units"] = reduce_func(cost_dict["forward_pass_units"])
        layer_costs[layer_name]["backward_pass_units"] = reduce_func(cost_dict["backward_pass_units"])
    profile_info = dict()
    for key, value in profiling_report.items():
        if key != "layer_costs":
            profile_info[key] = value
    return dict(profile_info=profile_info, layer_costs=layer_costs)


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
        name = get_module_name_from_op(torch_node)
        # Op
        op = torch_node.kind()
        # Inputs/outputs
        inputs = [i.unique() for i in torch_node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]
        print("name: {:25} scope: {:35} Op: {:20} inputs: {:60} outputs: {:20}".format(name, scope, str(op), str(inputs), str(outputs)))
