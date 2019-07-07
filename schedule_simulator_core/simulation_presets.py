"""
This module provides preset simulation setups for quick reuse
"""

from schedule_simulator_core.core import ProcessingUnit
from schedule_simulator_core.schedulers import FIFOScheduler
from schedule_simulator_core.DNN_functions import train
from schedule_simulator_core.utils import SimPrinter, generate_ascii_timeline, group_dict
import simpy
import numpy as np
import itertools
import time
import json
import argparse
import datetime

class GpuNetworkSim:
    def __init__(self, gpu_rate, network_rate, gpu_scheduler, network_scheduler, dag, batch_size, n_of_batches):
        self.args = dict(gpu_rate=gpu_rate, network_rate=network_rate, gpu_scheduler=str(gpu_scheduler),
                         network_scheduler=str(network_scheduler), dag=str(dag), batch_size=batch_size,
                         n_of_batches=n_of_batches)
        self.dag = dag
        self.env = simpy.Environment()
        self.gpu = ProcessingUnit(env=self.env, scheduler=gpu_scheduler, rate=gpu_rate, name="GPU",
                                  store_timeline=False)
        self.gpu_process = self.env.process(self.gpu.main_process())
        self.network = ProcessingUnit(env=self.env, scheduler=network_scheduler, rate=network_rate, name="Network",
                                      store_timeline=False)
        self.network_process = self.env.process(self.network.main_process())
        self.training_process = self.env.process(train(dag=self.dag, env=self.env, n_of_batches=n_of_batches,
                                                       batch_size=batch_size, computation_queue=self.gpu,
                                                       communication_queue=self.network))
        def close():
            yield self.training_process
            self.gpu_process.interrupt()
            self.network_process.interrupt()
        self.closing_process = self.env.process(close())

    def run(self, print_timeline=True, verbosity=1, time_line_args=None):
        results = dict()
        printer = SimPrinter(verbosity=verbosity).print
        self.gpu._sim_printer = printer
        self.network._sim_printer = printer
        self.env.run()
        if print_timeline:
            args = dict(time_grouping=1, row_labels=["type"], cell_labels=["index"], group_name_width=30, cell_width=4)
            if time_line_args is not None:
                args.update(time_line_args)
            for unit in [self.gpu, self.network]:
                report = generate_ascii_timeline(unit, **args)
                print(report)
                print("")

    def summarize(self):
        if self.env.now == 0:
            raise Exception("Cannot summarize before running the simulation!")
        summary = dict()
        # Get statistics about the dag layer wise costs
        dag_layer_costs = self.dag.get_layer_costs()
        dag_layer_costs_description = dict()
        for cost_name, costs in dag_layer_costs.items():
            stats = dict()
            # We cast these values to ensure they are JSON serializable
            stats["sum"] = int(np.sum(costs))
            stats["mean"] = float(np.mean(costs))
            stats["std"] = float(np.std(costs))
            dag_layer_costs_description[cost_name] = stats
        for cost_name, description in dag_layer_costs_description.items():
            for subkey in description:
                summary["{}.{}".format(cost_name, subkey)] = description[subkey]
        # Dag extras:
        for ek, ev in self.dag.extras.items():
            try:
                json.dumps(ev)
            except TypeError:
                ev = str(ev)
            summary["{}_{}".format("dag", ek)] = ev
        # General statistics. Add whatever you need here
        summary["gpu_util"] = self.gpu.get_utilization()
        summary["net_util"] = self.network.get_utilization()
        summary["total_time_steps"] = self.env.now
        # Add options used
        summary.update(self.args)
        return summary

    @staticmethod
    def summarize_group(simulations, include_simulation_indices=True):
        group_summary = dict()
        for key in simulations[0].summarize().keys():
            group_summary[key] = list()
        for simulation in simulations:
            for key, value in simulation.summarize().items():
                group_summary[key].append(value)
        return group_summary

    @staticmethod
    def run_group(gpu_rate, network_rate, gpu_scheduler, network_scheduler, dag, batch_size, n_of_batches,
                  clear_output=True, resolution=1e3, number_of_processes=None, save_timeline=False,
                  output_file_name=None, saving_interval=5, resolution_warning_threshold=0.1):
        """
        All of the required arguments can be either a value or a list of values to try.
        The product of all options will be taken and then iterated to generate simulations with all possible
        combinations.
        :param gpu_rate: (Value or list of values)
        :param network_rate: (Value or list of values)
        :param gpu_scheduler: (Value or list of values)
        :param network_scheduler: (Value or list of values)
        :param dag: The dag to run. forward & backward pass units are assumed to be in nanoseconds
        communication units is assumed to be in bytes. (Value or list of values)
        :param batch_size: How many forward, backward passes should we do before communicating? (Value or list of values)
        :param n_of_batches: How many batches or iterations should we run? (Value or list of values)
        :param clear_output: Whether or not to clear output of the cell if run in a notebook
        :param resolution: Units are all converted to seconds and then multiplied by the resolution and then cast to an
        integer. Therefore a resolution of 1e3 means a resolution of 1ms.
        Idea: I should make the function choose the optimal resolution based on the combination given so that it is not
        too large that the simulations take forever and not too small that they do not give accurate results.
        :param number_of_processes: The number of independent processes to spawn. If set to None then the total number
        of effective cores will be used. (Not implemented yet)
        Running all combinations of arguments can mean that hundreds of independent simulations
        will be run. However, a python interpreter spawns a single process which can only use one effective core which
        prevents us of using the full capacity of a multi-core processor. Which is why launching the simulations in a
        multi process setup when running on a multi-core cpu will speed it up tremendously.
        For reference:
        https://askubuntu.com/questions/949437/python-interpreter-only-using-12-cpu-power
        https://docs.python.org/3.6/library/multiprocessing.html
        :param saving_interval: Save report every x elapsed minutes.
        :return if return_simulation_objects is true then the function returns a list of all simulations run.
        Other wise a summary dict is returned
        {"resolution": resolution, "num_of_simulations": len(args_combinations), "args": [str(x) for x in args.items()],
        "results": results}
        """
        if output_file_name is None:
            output_file_name = "GpuNetworkSim_{}.simgroup.json".format(datetime.datetime.now().strftime("%m-%d-%H-%M"))
        args = dict(gpu_rate=gpu_rate, network_rate=network_rate, gpu_scheduler=gpu_scheduler,
                    network_scheduler=network_scheduler, dag=dag, batch_size=batch_size, n_of_batches=n_of_batches)
        constants = set()
        for k, v in args.copy().items():
            try:
                if len(v) == 1:
                    constants.add(k)
            except TypeError:
                constants.add(k)
                args[k] = [args[k]]
        # Define scaling functions based on the assumptions of the dag's units mentioned above
        def scale_comp_units(value, gpu_rate):
            """From nanoseconds to seconds to resolution units"""
            return int(value * 1e-9 / gpu_rate * resolution)

        def scale_comm_units(value, network_rate):
            """From bytes to seconds to resolution units"""
            return int(value * 1e-9 * 8 / network_rate * resolution)

        # Check if a combination is compromised with the resolution
        total_zeros = dict(bp=0, fp=0, comm=0)
        num_of_values = 0
        warnings = list()
        for comb in list(itertools.product(args["dag"], args["gpu_rate"], args["network_rate"])):
            _dag, _gpu_rate, _network_rate = comb
            local_zeros = dict(bp=0, fp=0, comm=0)
            local_num_of_values = 0
            def add_zeros(layer):
                nonlocal local_num_of_values
                local_num_of_values += 1
                if layer.forward_pass_units != 0 and scale_comm_units(layer.forward_pass_units, _gpu_rate) == 0:
                    local_zeros["fp"] += 1
                if layer.backward_pass_units != 0 and scale_comm_units(layer.backward_pass_units, _gpu_rate) == 0:
                    local_zeros["bp"] += 1
                if layer.communication_units != 0 and scale_comm_units(layer.communication_units, _network_rate) == 0:
                    local_zeros["comm"] += 1
            _dag.traverse_BFS(processing_function=add_zeros)
            for key in local_zeros:
                total_zeros[key] += local_zeros[key]
                zeros_perc = local_zeros[key] / local_num_of_values
                if zeros_perc > resolution_warning_threshold:
                    warnings.append((comb, key, zeros_perc))
            num_of_values += local_num_of_values
        if len(warnings) > 0:
            exception_msg = "Resolution Error\n"
            exception_msg += "The following combinations are compromised with the current resolution:\n"
            for warning in warnings:
                comb, key, zeros_perc = warning
                _dag, _gpu_rate, _network_rate = comb
                exception_msg += ("Compromised unit: '{:4}' Resolution inflicted zeros: {:.2f}%\n"
                                  "Combination: (dag:{}, gpu_rate: {}, network_rate: {}\n").format(
                    key, zeros_perc*100, _dag, _gpu_rate, _network_rate)
            exception_msg += "Consider increasing the resolution or changing the arguments.\n"
            exception_msg += "To suppress this warning and continue increase the resolution_warning_threshold."
            raise Exception(exception_msg)

        def print_header():
            print("Resolution: {}".format(resolution))
            print("Resolution inflicted zeros: fp_units: {:.2f}% bp_units: {:.2f}% comm_units: {:.2f}%".format(
                total_zeros["fp"] / num_of_values * 100,
                total_zeros["bp"] / num_of_values * 100,
                total_zeros["comm"] / num_of_values * 100,
            ))
            print("Constant arguments: {}".format(constants))
            print("Variable arguments: {}".format(args.keys() - constants))
            print("-"*100)
        if not clear_output:
            print_header()
        args_combinations = list(itertools.product(*args.values()))
        # Initialize summary dictionary
        summary = dict(total_time_elapsed=0, total_num_of_simulations=len(args_combinations),
                       finished_simulations=0, args=dict(), results=dict())
        summary["args"]["resolution"] = resolution
        for key, value in args.items():
            new_value = list()
            for v in value:
                try:
                    json.dumps(v)
                except TypeError:
                    v = str(v)
                new_value.append(v)
            summary["args"][key] = new_value
        summary["results"]["sim_index"] = list(range(len(args_combinations)))
        group_begin_time = time.time()
        save_interval_begin_time = time.time()
        try:
            # Run a simulation for each argument combination
            for sim_i, args_combination in enumerate(args_combinations):
                # Map combination to argument names
                sub_args = dict()
                for i, key in enumerate(args.keys()):
                    sub_args[key] = args_combination[i]
                # If we are running in a notebook then clear the output if specified
                if clear_output:
                    try:
                        from IPython.display import clear_output
                        clear_output(wait=True)
                    except:
                        pass
                    print_header()
                # Print the current combination of arguments
                print("{:20}: {:<3}/{:<3}".format("Simulation", sim_i+1, len(args_combinations)))
                for arg_key, arg_value in sub_args.items():
                    print("{:20}: {}".format(arg_key, arg_value))
                # Scale & unify units
                # This block mitigates a defect in the simulator and it should be edited once that defect is fixed !!
                # ------------------------------------------------------------------------------------------------------
                # (We apply rates here because if we apply them using the processing unit rate, the simulation will take
                # a huge amount of time since we would be essentially running with a very high resolution)
                sub_args["dag"] = sub_args["dag"].copy()
                def scale_units(layer):
                    layer.communication_units = scale_comm_units(layer.communication_units, sub_args["network_rate"])
                    layer.forward_pass_units = scale_comm_units(layer.forward_pass_units, sub_args["gpu_rate"])
                    layer.backward_pass_units = scale_comm_units(layer.backward_pass_units, sub_args["gpu_rate"])
                sub_args["dag"].traverse_BFS(processing_function=scale_units)
                # Since the rates won't be passed on to the simulation processing units and it won't appear in the
                # summary, let us save it now and add it later to the summary below
                patch = dict()
                patch["comm_units_scaling_rate"] = sub_args["network_rate"]
                patch["comp_units_scaling_rate"] = sub_args["gpu_rate"]
                # Since we effectively applied the rate by scaling the units we can set the processing rate to 1
                sub_args["network_rate"] = 1
                sub_args["gpu_rate"] = 1
                # ------------------------------------------------------------------------------------------------------
                # Create, run, and append simulation
                simulation = GpuNetworkSim(**sub_args)
                sim_time_begin = time.time()
                simulation.run(print_timeline=False, verbosity=0)
                # Update summary --
                sim_summary = simulation.summarize()
                sim_summary["execution_duration"] = time.time() - sim_time_begin
                # We add the network and gpu rates here as mentioned in the block above
                sim_summary.update(patch)
                for key, value in sim_summary.items():
                    if key in summary["results"]:
                        summary["results"][key].append(value)
                    else:
                        summary["results"][key] = [value]
                summary["total_time_elapsed"] = time.time() - group_begin_time
                summary["finished_simulations"] = sim_i + 1
                if time.time() - save_interval_begin_time > saving_interval * 60:
                    save_interval_begin_time = time.time()
                    with open(output_file_name, "w") as output_file:
                        json.dump(summary, output_file, indent=4)
        except KeyboardInterrupt:
            print("Simulations stopped by user. Returning gathered results..")
        with open(output_file_name, "w") as output_file:
            json.dump(summary, output_file, indent=4)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
