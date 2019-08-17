import sys
sys.path.append("..")
from schedule_simulator_core.DAGs import LOCAL_EXTRA_PREFIX


def apply_layer_costs_to_dag(dag, extracted_costs):
    """
    :param dag: The simulator dat to apply the profile to
    :param suppress_negatives: passed directly to extract_cost_units_from_profile function
    :param scaling_factor: passed directly to extract_cost_units_from_profile function
    """
    dag.extras["{}profile_info".format(LOCAL_EXTRA_PREFIX)] = extracted_costs["profile_info"]
    extracted_costs = extracted_costs["layer_costs"]
    def apply_timing(sim_layer):
        layer_name = sim_layer.extras["name"]
        if layer_name not in extracted_costs:
            print("Skipping layer {} since its costs were not found in the costs report.".format(layer_name))
            return
        layer_timing = extracted_costs[layer_name]
        if "forward_pass_units" in layer_timing:
            dag.extras["{}forward_pass_unit".format(LOCAL_EXTRA_PREFIX)] = "ns"
            sim_layer.forward_pass_units = layer_timing["forward_pass_units"]
        if "backward_pass_units" in layer_timing:
            dag.extras["{}backward_pass_unit".format(LOCAL_EXTRA_PREFIX)] = "ns"
            sim_layer.backward_pass_units = layer_timing["backward_pass_units"]
        if "communication_units" in layer_timing:
            dag.extras["{}comm_unit".format(LOCAL_EXTRA_PREFIX)] = "ns"
            sim_layer.communication_units = layer_timing["communication_units"]
    dag.traverse_DFS(processing_function=apply_timing)


def remove_untrainable_layers(dag):
    """
    Removes layers that have communication cost as 0. Removed layers will have their forward pass cost added to the next
    layer, and their backward pass cost added to the previous layer according to the topological order of which they are
    processed in the simulator.
    :return: None, modification is done in place.
    """
    # TODO Verify your logic here with professor
    # Delegate layer forward and backward costs of untrainable layers
    # This does not depend on the dag structure but rather on the execution order in the simulator.
    for i, layer in enumerate(dag.topological_order):
        if layer.communication_units == 0:
            if i-1 > 0:
                prev_layer = dag.topological_order[i-1]
                prev_layer.backward_pass_units += layer.backward_pass_units
            if i+1 < len(dag.topological_order):
                next_layer = dag.topological_order[i+1]
                next_layer.forward_pass_units += layer.forward_pass_units
    # Remove untrainable layers from the linked structure
    c = 0
    for layer in dag.topological_order:
        if layer.communication_units == 0:
            c += 1
            dag.remove_layer(layer)
    print("Removed {} untrainable layers.".format(c))
    dag.produce_topological_order()
    dag.extract_dependencies()
    dag.extras["{}extraction_info".format(LOCAL_EXTRA_PREFIX)]["remove_untrainable_layers"] = True


def produce_dag(model_name, library, profiling_report_path=None, output_file_name=None,
                skip_untrainable_layers=True,
                dag_construction_args=None, cost_extraction_args=None):
    import json
    from schedule_simulator_core.DAGs import serialize_dag
    if dag_construction_args is None:
        dag_construction_args = dict()
    if cost_extraction_args is None:
        cost_extraction_args = dict()
    # 1. Create dag
    if library == "tensorflow":
        from model_extraction.tensorflow_utils import get_model
        from model_extraction.tensorflow_model_extractor import keras_model_to_DAG
        model = get_model(model_name)
        dag = keras_model_to_DAG(model, **dag_construction_args)
    elif library == "pytorch":
        from model_extraction.pytorch_utils import get_model
        from model_extraction.pytorch_model_extractor import pytorch_model_to_DAG
        model = get_model(model_name)
        dag = pytorch_model_to_DAG(model, **dag_construction_args)
    else:
        raise Exception("'{}' is an invalid library. Please choose one of the following: 'tensorflow', 'pytorch'.")
    if profiling_report_path is not None:
        with open(profiling_report_path) as report_file:
            report = json.load(report_file)
        # 2. Extract costs
        if "method" not in report:
            raise Exception("No valid method has been supplied in the report")
        method = report["method"]
        device = report["args"]["device"]
        if library not in method:
            # TODO Allow models and profiling reports to differ. Store profiling data using their topological order index
            # instead of using names.
            raise Exception("Profile and model libraries must match. Layer names differ from library to library and we"
                            "would not be able to find a direct mapping.")
        if method == "pytorch_module_hooks":
            from model_extraction.pytorch_model_extractor import extract_costs_from_module_hooks_profile
            layer_costs = extract_costs_from_module_hooks_profile(report, **cost_extraction_args)
        elif method == "tensorflow_layer_name_mapping":
            from model_extraction.tensorflow_model_extractor import extract_costs_from_layer_name_mapping_profile
            layer_costs = extract_costs_from_layer_name_mapping_profile(report, **cost_extraction_args)
        elif method == "tensorflow_model_reconstruction":
            from model_extraction.tensorflow_model_extractor import extract_costs_from_model_reconstruct_profile
            layer_costs = extract_costs_from_model_reconstruct_profile(report, **cost_extraction_args)
        else:
            raise Exception("No valid method has been supplied in the report")
        # 3. Apply costs to dag
        apply_layer_costs_to_dag(dag, layer_costs)
    # 4. Remove untrainable layers
    if skip_untrainable_layers:
        remove_untrainable_layers(dag)
    # 5. Write dag to file
    if output_file_name is None:
        output_file_name = ("{}.dag".format(model_name) if profiling_report_path is None else
                            "{}_{}_{}.dag".format(model_name, device, method))
    with open(output_file_name, "w") as file:
            file.write(serialize_dag(dag))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="A script that produces a serialized dag from profiling reports.")
    parser.add_argument("model",
                        help="The model name. To see available names check the get_model method in "
                             "pytorch_utils or tensorflow_utils")
    parser.add_argument("library", choices=["tensorflow", "pytorch"],
                        help="The library to use to look for the model and construct a dag.")
    parser.add_argument("-p", "--profile",
                        help="The profiling report path.")
    parser.add_argument("-o", "--output",
                        help="The outputted dag file name")
    parser.add_argument("-ka", "--keep_all", default=False, action="store_true",
                        help="Whether we should keep all layers, even untrainable ones")
    args = parser.parse_args()
    produce_dag(model_name=args.model, library=args.library, profiling_report_path=args.profile,
                output_file_name=args.output, skip_untrainable_layers=not args.keep_all)
