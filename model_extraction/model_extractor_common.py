from schedule_simulator_core.DAGs import Layer, LOCAL_EXTRA_PREFIX


def apply_layer_costs_to_dag(dag, extracted_costs):
    """
    :param dag: The simulator dat to apply the profile to
    :param suppress_negatives: passed directly to extract_cost_units_from_profile function
    :param scaling_factor: passed directly to extract_cost_units_from_profile function
    """
    def apply_timing(sim_layer: Layer):
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
    dag.traverse_BFS(processing_function=apply_timing)


def produce_dag(model, profiling_report, library):
    pass