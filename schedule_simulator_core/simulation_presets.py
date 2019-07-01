"""
This module provides preset simulation setups for quick reuse
"""

from schedule_simulator_core.core import ProcessingUnit
from schedule_simulator_core.schedulers import FIFOScheduler
from schedule_simulator_core.DNN_functions import train
from schedule_simulator_core.io_utils import SimPrinter, generate_report
import simpy
import numpy as np

class GpuNetworkSchedulingSim:
    def __init__(self, gpu_rate, network_rate, schedulers, dag):
        self.gpu_rate = gpu_rate
        self.network_rate = network_rate
        self.schedulers = schedulers
        self.dag = dag
        self.last_run_result_objects = None
        self.last_run_args = None

    def run_simulation(self, batch_size, n_of_batches, print_report=True, verbosity=1, report_args=None):
        args = dict(batch_size=batch_size, n_of_batches=n_of_batches, print_report=print_report, verbosity=1,
                    report_args=report_args)
        results = dict()
        printer = SimPrinter(verbosity=verbosity).print
        for scheduler in self.schedulers:
            env = simpy.Environment()

            gpu = ProcessingUnit(env=env, scheduler=FIFOScheduler(), rate=self.gpu_rate, name="GPU", sim_printer=printer)
            gpu_process = env.process(gpu.main_process())

            network = ProcessingUnit(env=env, scheduler=scheduler, rate=self.network_rate, name="Network", sim_printer=printer)
            network_process = env.process(network.main_process())

            training_process = env.process(train(dag=self.dag, env=env, n_of_batches=n_of_batches,
                                                 batch_size=batch_size, computation_queue=gpu,
                                                 communication_queue=network))

            def close():
                yield training_process
                gpu_process.interrupt()
                network_process.interrupt()
            closing_process = env.process(close())
            results[scheduler] = (env, gpu, network)
            env.run()
        if print_report:
            args = dict(time_grouping=1, row_labels=["type"], cell_labels=["index"], group_name_width=30, cell_width=4)
            if report_args is not None:
                args.update(report_args)
            for scheduler in results:
                for unit in results[scheduler][1:]:
                    report = generate_report(unit, **args)
                    print(report)
                    print("")
        self.last_run_result_objects = results
        self.last_run_args = args

    @staticmethod
    def parse_simulations_results(simulations):
        """
        Calculates statistics regarding all simulations
        Intended to make plotting results easy.
        :param simulations: A list of simulations to parse
        :return: dict(key=scheduler_name, value=dict(key=stat_name, value=a list that has a value for each simulation))
        for description values they are generated using the below method.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html
        """
        from scipy.stats import describe
        import numpy as np
        results = dict()
        for simulation in simulations:
            dag_layer_costs = simulation.dag.get_layer_costs()
            dag_layer_costs_description = dict()
            for cost_name, costs in dag_layer_costs.items():
                stats = dict()
                stats["sum"] = np.sum(costs)
                stats["mean"] = np.mean(costs)
                stats["std"] = np.std(costs)
                dag_layer_costs_description[cost_name] = stats
            for scheduler in simulation.last_run_result_objects:
                scheduler_name = str(scheduler)
                if scheduler_name not in results:
                    stats = ["gpu_util", "net_util", "time", "batch_size", "n_of_batches"]
                    for cost_name, description in dag_layer_costs_description.items():
                        for subkey in description:
                            stats.append("{}.{}".format(cost_name, subkey))
                    results[scheduler_name] = dict()
                    for stat in stats:
                        results[scheduler_name][stat] = list()
                scheduler_results = results[scheduler_name]
                env, gpu, network = simulation.last_run_result_objects[scheduler]
                scheduler_results["gpu_util"].append(gpu.get_utilization())
                scheduler_results["net_util"].append(network.get_utilization())
                scheduler_results["time"].append(env.now)
                scheduler_results["batch_size"].append(simulation.last_run_args["batch_size"])
                scheduler_results["n_of_batches"].append(simulation.last_run_args["n_of_batches"])
                for cost_name, description in dag_layer_costs_description.items():
                    for subkey in description:
                        scheduler_results["{}.{}".format(cost_name, subkey)].append(description[subkey])
        return results

    @staticmethod
    def run_simulations_group(base_dag, schedulers, network_rates, batch_sizes, nums_of_batches,
                              simulation_resolution=1e3):
        """
        :param base_dag: The dag that we will keep copying from with each simulation.
        Its computational units are assumed to be in nanoseconds.
        Its communication units are assumed to be in bytes.
        :param schedulers: The schedulers to compare
        :param batch_sizes: A list of batch sizes to try
        :param network_rates: A list of network_rates to try in Gbps
        :param nums_of_batches: A list of numbers of batches to try
        :param simulation_resolution: If set to 1 then costs are set in seconds,
        If set to 1e3 then costs are in milliseconds, the lower the resolution, the faster the simulation but the lower
        the accuracy.
        """
        simulations = list()
        for num_of_batches in nums_of_batches:
            for batch_size in batch_sizes:
                for network_rate in network_rates:
                    try:
                        from IPython.display import clear_output
                        clear_output(wait=True)
                    except:
                        pass
                    print("num_of_batches  : {}".format(num_of_batches))
                    print("batch_size      : {}".format(batch_size))
                    print("network_rate    : {:.2} Gbps".format(network_rate))
                    dag = base_dag.clone()
                    # Unify units (We apply rates here because if we apply them using the processing unit rate, the simulation will
                    # take a huge amount of time since we would be essentially running with a very high resolution)
                    communication_units_scale = 8 / (network_rate * 1e9)  # From bytes to seconds
                    computation_units_scale = 1e-9  # From nanoseconds to seconds

                    def scale_units(layer):
                        layer.communication_units *= simulation_resolution * communication_units_scale
                        layer.forward_pass_units *= simulation_resolution * computation_units_scale
                        layer.backward_pass_units *= simulation_resolution * computation_units_scale

                    dag.traverse_BFS(processing_function=scale_units)
                    simulation = GpuNetworkSchedulingSim(gpu_rate=1, network_rate=1, schedulers=schedulers, dag=dag)
                    simulation.run_simulation(batch_size=batch_size, n_of_batches=num_of_batches, print_report=False,
                                              verbosity=0)
                    simulations.append(simulation)
        results = GpuNetworkSchedulingSim.parse_simulations_results(simulations)
        return results

    @staticmethod
    def comp_to_comm_ratio(results, scheduler):
        return (np.array(results[scheduler]["comp_units.sum"]) * np.array(results[scheduler]["batch_size"])) \
               / np.array(results[scheduler]["comm_units.sum"])

    @staticmethod
    def speedup_over_fifo(results, scheduler):
        return np.array(results["FIFOScheduler"]["time"]) / np.array(results[scheduler]["time"])
