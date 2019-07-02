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


class GpuNetworkSim:
    def __init__(self, gpu_rate, network_rate, gpu_scheduler, network_scheduler, dag, batch_size, n_of_batches):
        self.args = dict(gpu_rate=gpu_rate, network_rate=network_rate, gpu_scheduler=str(gpu_scheduler),
                         network_scheduler=str(network_scheduler), dag=str(dag), batch_size=batch_size,
                         n_of_batches=n_of_batches)
        self.dag = dag
        self.env = simpy.Environment()
        self.gpu = ProcessingUnit(env=self.env, scheduler=gpu_scheduler, rate=gpu_rate, name="GPU")
        self.gpu_process = self.env.process(self.gpu.main_process())
        self.network = ProcessingUnit(env=self.env, scheduler=network_scheduler, rate=network_rate, name="Network")
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
        # General statistics. Add whatever you need here
        summary["gpu_util"] = self.gpu.get_utilization()
        summary["net_util"] = self.network.get_utilization()
        summary["time"] = self.env.now
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
                  clear_output=True, resolution=1e3, return_simulation_objects=False):
        """
        All of the required arguments can be either a value or a list of values to try.
        The product of all options will be taken and then iterated to generate simulations with all possible
        combinations
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
        :param return_simulation_objects: Setting this to true will return a list of simulations instead of the summary
        file. Enabling this option will consume a huge amount of RAM since all of the simulations data is kept in
        memory. Therefore only enable for a small number of simulations
        :return if return_simulation_objects is true then the function returns a list of all simulations run.
        Other wise a summary dict is returned { "resolution": resolution, "args": args, "results": results}
        """
        constants = set()
        args = dict(gpu_rate=gpu_rate, network_rate=network_rate, gpu_scheduler=gpu_scheduler,
                    network_scheduler=network_scheduler, dag=dag, batch_size=batch_size, n_of_batches=n_of_batches)
        for k, v in args.copy().items():
            try:
                if len(v) == 0:
                    constants.add(k)
            except TypeError:
                constants.add(k)
                args[k] = [args[k]]
        if not clear_output:
            print("Arguments kept constant: {}".format(constants))
        args_combinations = list(itertools.product(*args.values()))
        results = dict()
        results["sim_index"] = list(range(len(args_combinations)))
        simulations = list()
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
                # Print the current combination of arguments
                print("Arguments kept constant: {}".format(constants))
                print("{:20}: {:<3}/{:<3}".format("Simulation", sim_i+1, len(args_combinations)))
                for arg_key, arg_value in sub_args.items():
                    print("{:20}: {}".format(arg_key, arg_value))
                # Scale & unify units (We apply rates here because if we apply them using the processing unit rate,
                # the simulation will take a huge amount of time since we would be essentially running with a
                # very high resolution)
                sub_args["dag"] = sub_args["dag"].copy()
                communication_units_scale = 1e-9 * 8 / (sub_args["network_rate"])  # From bytes to seconds
                computation_units_scale = 1e-9 / (sub_args["gpu_rate"])  # From nanoseconds to seconds

                def scale_units(layer):
                    layer.communication_units = int(layer.communication_units * resolution * communication_units_scale)
                    layer.forward_pass_units = int(layer.forward_pass_units * resolution * computation_units_scale)
                    layer.backward_pass_units = int(layer.backward_pass_units * resolution * computation_units_scale)
                sub_args["dag"].traverse_BFS(processing_function=scale_units)
                # Create, run, and append simulation
                simulation = GpuNetworkSim(**sub_args)
                simulation.run(print_timeline=False, verbosity=0)
                if return_simulation_objects:
                    simulations.append(simulation)
                else:
                    for key, value in simulation.summarize().items():
                        if key in results:
                            results[key].append(value)
                        else:
                            results[key] = [value]
        except KeyboardInterrupt:
            print("Simulations stopped by user. Returning gathered results..")
        if return_simulation_objects:
            return simulations
        else:
            return {"resolution": resolution,
                    "args": [str(x) for x in args],
                    "results": results}
