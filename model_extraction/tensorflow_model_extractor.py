from schedule_simulator_core.DAGs import Layer, DAG, LOCAL_EXTRA_PREFIX
from model_extraction.tensorflow_utils import traverse_keras_DFS, get_layer_children, get_layer_parents
import numpy as np


def keras_model_to_DAG(model, skip_untrainable_layers=False):
    input_layers = set()
    all_layers = dict()
    skipped_layers_connections = dict()  # Needed to forward connections

    def skip(keras_layer):
        return skip_untrainable_layers and keras_layer.count_params() == 0

    i = 0
    def add_layer(keras_layer):
        nonlocal i
        if skip(keras_layer):
            connections = {"parents": [x.name for x in get_layer_parents(keras_layer)],
                           "children": [x.name for x in get_layer_children(keras_layer)]}
            skipped_layers_connections[keras_layer.name] = connections
            return
        param_count = keras_layer.count_params()
        comm_units = param_count * 4  # Each parameter is a 4 bytes
        comp_units = 0  # To be applied through profiling reports
        sim_layer = Layer(comp_units, comp_units, comm_units, name=keras_layer.name, type=type(keras_layer).__name__)
        all_layers[keras_layer.name] = sim_layer
        i += 1

    def connect_layer(keras_layer):
        if skip(keras_layer):
            return
        sim_layer = all_layers[keras_layer.name]
        sim_layer.input_layers = set()
        sim_layer.output_layers = set()

        def add_parents_after_skipping(keras_layer, parents_set):
            for parent in get_layer_parents(keras_layer):
                if skip(parent):
                    add_parents_after_skipping(parent, parents_set)
                else:
                    parents_set.add(parent)

        def add_children_after_skipping(keras_layer, children_set):
            for child in get_layer_children(keras_layer):
                if skip(child):
                    add_children_after_skipping(child, children_set)
                else:
                    children_set.add(child)

        keras_parents = set()
        add_parents_after_skipping(keras_layer, keras_parents)
        keras_children = set()
        add_children_after_skipping(keras_layer, keras_children)
        if len(keras_parents) == 0:
            input_layers.add(sim_layer)
        for keras_parent_layer in keras_parents:
            sim_layer.input_layers.add(all_layers[keras_parent_layer.name])
        for keras_child_layer in keras_children:
            sim_layer.output_layers.add(all_layers[keras_child_layer.name])

    for fun in [add_layer, connect_layer]:
        traverse_keras_DFS(model, processing_function=fun, order="pre-order", top_to_bottom=True)
    extraction_method = dict(library="tensorflow", skip_untrainable_layers=skip_untrainable_layers)
    extras = {"comm_unit": "B", "forward_pass_unit": None, "backward_pass_unit": None,
              "extraction_info": extraction_method}
    prefixed_extras = dict()
    for k, v in extras.items():
        prefixed_extras[LOCAL_EXTRA_PREFIX+k] = v
    return DAG(input_layers, name=model.name, **prefixed_extras)


def extract_costs_from_model_reconstruct_profile(profiling_report, suppress_negatives=2, reduce_func=None):
    """
    :param profiling_report: a report generated by profiling a keras model using the
    tensorflow_model_reconstruction_profiler.py tool
    :param suppress_negatives: False | after | before
    0: values are left as is
    1: Negative values are left to affect next layer calculation however it is recorded as a 0 at the end. This option
    can increase the total cost of the dag.
    2: Negative values set to 0 before being sent to the next layer so that it does not affect next layers. This option
    guarantees that the total cost of layers does not exceed the real total cost.
    :return: A dict(key=layer.name, value=dict(key=cost_name, value=COST))
    """
    timings = profiling_report["timings"]
    layer_costs = dict()
    accumulative_cost = {"forward_pass_units": 0, "backward_pass_units": 0}
    fw = 0
    bw = 0
    for layer_name in timings:
        evaluate = timings[layer_name]["evaluate"]
        fit = timings[layer_name]["fit"]
        if isinstance(evaluate, dict):
            evaluate = evaluate["durations"]
            fit = fit["durations"]
        if reduce_func is None:
            reduce_func = lambda x: min(x)
        fit = reduce_func(fit)
        evaluate = reduce_func(evaluate)
        current_cost = dict()
        current_cost["forward_pass_units"] = evaluate - accumulative_cost["forward_pass_units"]
        current_cost["backward_pass_units"] = fit - evaluate - accumulative_cost["backward_pass_units"]
        if suppress_negatives == 1:
            accumulative_cost["forward_pass_units"] += current_cost["forward_pass_units"]
            accumulative_cost["backward_pass_units"] += current_cost["backward_pass_units"]
        if suppress_negatives > 0:
            if current_cost["forward_pass_units"] < 0:
                fw += 1
                print("Suppressing {:18} forward_pass_units  {}".format(layer_name, current_cost["forward_pass_units"]))
                current_cost["forward_pass_units"] = 0
            if current_cost["backward_pass_units"] < 0:
                bw += 1
                print("Suppressing {:18} backward_pass_units {}".format(layer_name, current_cost["backward_pass_units"]))
                current_cost["backward_pass_units"] = 0
        if suppress_negatives != 1:
            accumulative_cost["forward_pass_units"] += current_cost["forward_pass_units"]
            accumulative_cost["backward_pass_units"] += current_cost["backward_pass_units"]
        layer_costs[layer_name] = current_cost
    profile_info = dict()
    for key, value in profiling_report.items():
        if key != "timings":
            profile_info[key] = value
    if suppress_negatives > 0:
        print("Suppressed {}/{} forward pass costs".format(fw, len(timings)))
        print("Suppressed {}/{} backward pass costs".format(bw, len(timings)))
    return dict(profile_info=profile_info, layer_costs=layer_costs)


def extract_costs_from_layer_name_mapping_profile(profiling_report, reduce_func=None,
                                                  skip_first_batch_of_every_trial=False):
    layer_costs = profiling_report["layer_costs"]
    if reduce_func is None:
        reduce_func = lambda x: min(x)
    for layer_name, cost_dict in layer_costs.items():
        if skip_first_batch_of_every_trial:
            for cost_name, cost_list in cost_dict.items():
                for i in range(profiling_report["args"]["trials"]):
                    cost_list.pop(i * profiling_report["args"]["num_of_batches"] - i)
        layer_costs[layer_name]["forward_pass_units"] = reduce_func(cost_dict["forward_pass_sequential_units"])
        layer_costs[layer_name]["backward_pass_units"] = reduce_func(cost_dict["backward_pass_sequential_units"])
    profile_info = dict()
    for key, value in profiling_report.items():
        if key != "layer_costs":
            profile_info[key] = value
    return dict(profile_info=profile_info, layer_costs=layer_costs)
