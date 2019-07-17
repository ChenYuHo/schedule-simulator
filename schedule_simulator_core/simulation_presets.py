"""
This module provides preset simulation setups for quick reuse
Todo generalize the functions used in GpuNetworkSim so that they can be reused with other simulations
"""
import simpy
import statistics
import itertools
import time
import json
import datetime
import multiprocessing
import threading
from queue import Empty
import os
import traceback
import sys
sys.path.append("..")
from schedule_simulator_core.schedulers import FIFOScheduler, TopologicalPriorityScheduler
from schedule_simulator_core.core import ProcessingUnit
from schedule_simulator_core.DNN_functions import train
from schedule_simulator_core.utils import SimPrinter, trim, sort_table, get_gaps, Mbps_to_Bpns, get_normalized_gap_durations
from schedule_simulator_core.DAGs import deserialize_dag, LOCAL_EXTRA_PREFIX
SIM_GROUP_REPORT_POSTFIX = ".simgroup.json"


class GpuNetworkSim:
    def __init__(self, network_bandwidth, gpu_scheduler, network_scheduler, dag, batch_size, n_of_batches,
                 keep_timeline=True):
        self.args = dict(network_bandwidth=network_bandwidth, gpu_rate=1, network_rate=Mbps_to_Bpns(network_bandwidth),
                         gpu_scheduler=str(gpu_scheduler), network_scheduler=str(network_scheduler), dag=str(dag),
                         batch_size=batch_size, n_of_batches=n_of_batches, keep_timeline=keep_timeline)
        self.dag = dag
        self.env = simpy.Environment()
        self.keep_timeline = keep_timeline
        self.gpu = ProcessingUnit(env=self.env, scheduler=gpu_scheduler, rate=self.args["gpu_rate"], name="GPU",
                                  keep_timeline=self.keep_timeline)
        self.gpu_process = self.env.process(self.gpu.main_process())
        self.network = ProcessingUnit(env=self.env, scheduler=network_scheduler, rate=self.args["network_rate"],
                                      name="Network", keep_timeline=self.keep_timeline)
        self.network_process = self.env.process(self.network.main_process())
        self.training_process = self.env.process(train(dag=self.dag, env=self.env, n_of_batches=n_of_batches,
                                                       batch_size=batch_size, computation_queue=self.gpu,
                                                       communication_queue=self.network))
        def close():
            yield self.training_process
            self.gpu_process.interrupt()
            self.network_process.interrupt()
        self.closing_process = self.env.process(close())

    def run(self, verbosity=1):
        printer = SimPrinter(verbosity=verbosity).print
        self.gpu._sim_printer = printer
        self.network._sim_printer = printer
        self.env.run()

    def summarize(self, include_gaps=True):
        if self.env.now == 0:
            raise Exception("Cannot summarize before running the simulation!")
        summary = dict()
        # Get statistics about the dag layer wise costs
        dag_layer_costs = self.dag.get_layer_costs()
        dag_layer_costs_description = dict()
        for cost_name, costs in dag_layer_costs.items():
            stats = dict()
            # numpy was used previously to calculate the stats in this block however it was replaced due to two
            # problems:
            # numpy results are not json serializable. Can be fixed by simply casting using float() or int().
            # numpy uses int32 by default and was causing overflow errors silently. Can be fixed by specifying the type
            # used with each function call (np.sum(array, dtype="float64") however even a float 64 value can overflow
            # so we simply use built in python functions instead.
            stats["sum"] = sum(costs)
            stats["mean"] = statistics.mean(costs)
            stats["std"] = statistics.stdev(costs)
            dag_layer_costs_description[cost_name] = stats
        for cost_name, description in dag_layer_costs_description.items():
            for subkey in description:
                summary["{}.{}".format(cost_name, subkey)] = description[subkey]
        # Dag extras:
        for ek, ev in self.dag.extras.items():
            if ek.startswith(LOCAL_EXTRA_PREFIX):
                continue
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
        if include_gaps:
            gpu_gaps = get_gaps(self.gpu)[tuple()]
            summary["$list$gpu_gaps_durations"] = [e-b for b, e in gpu_gaps]
            summary["$list$network_gaps_durations"] = [e-b for b, e in get_gaps(self.network)[tuple()]]
            grouped_gaps = get_gaps(self.gpu, group_labels=["type", "batch"])
            forward_pass_gaps = list()
            for group in grouped_gaps.keys():
                if group[0] == "forward_pass":
                    forward_pass_gaps.extend(grouped_gaps[group])
            summary["$list$forward_pass_gaps_durations"] = [e-b for b, e in forward_pass_gaps]
            def cost(index):
                return self.dag.topological_order[index].communication_units / self.network.rate
            summary["$list$gpu_normalized_gaps_durations"] = get_normalized_gap_durations(self.gpu, gpu_gaps, cost)
            summary["$list$forward_pass_normalized_gaps_durations"] = get_normalized_gap_durations(
                self.gpu, forward_pass_gaps, cost)
        return summary

    @staticmethod
    def summarize_group(simulations):
        group_summary = dict()
        for key in simulations[0].summarize().keys():
            group_summary[key] = list()
        for simulation in simulations:
            for key, value in simulation.summarize().items():
                group_summary[key].append(value)
        return group_summary

    @staticmethod
    def _run_sub_group_process(input_queue, running_queue, output_queue, args, output_trace_file_name, include_gaps,
                               include_util_in_trace=True):
        print("pid[{}] starting process..".format(os.getpid()))
        while not input_queue.empty():
            sim_i, args_combination = input_queue.get()
            try:
                running_queue.put((os.getpid(), sim_i))
                # Map combination to argument names
                sub_args = dict()
                for i, key in enumerate(args.keys()):
                    sub_args[key] = args_combination[i]
                # Restore locks for schedulers
                sub_args["gpu_scheduler"]._lock = threading.Lock()
                sub_args["network_scheduler"]._lock = threading.Lock()
                if output_trace_file_name is None and not include_gaps:
                    sub_args["timeline_format"] = None
                # Compute rates to be used with simulation
                # Create, run, and append simulation
                simulation = GpuNetworkSim(**sub_args)
                sim_time_begin = time.time()
                simulation.run(verbosity=0)
                # Update summary --
                sim_summary = simulation.summarize(include_gaps=include_gaps)
                final_summary = dict()
                final_summary["sim_index"] = sim_i
                final_summary["execution_duration"] = time.time() - sim_time_begin
                final_summary.update(sim_summary)
                if output_trace_file_name is not None:
                    from schedule_simulator_core.utils import generate_chrome_trace_timeline, join_chrome_traces
                    traces = list()
                    for unit in [simulation.gpu, simulation.network]:
                        traces.append(
                            generate_chrome_trace_timeline(unit, group_labels=["unit_name"], row_labels=["type"],
                                                           cell_labels=["name"],
                                                           utilization_bins=500 if include_util_in_trace else None,
                                                           return_dict=True, multiplier=1e-3))
                    final_trace = join_chrome_traces(traces, use_trace_dict=True)
                    trace_metadata = dict()
                    for key in final_summary:
                        if "$" not in key:
                            trace_metadata[key] = final_summary[key]
                    final_trace.update(trace_metadata)
                    with open("{}_sim{}_.chrometrace.json".format(output_trace_file_name, sim_i), "w") as f:
                        json.dump(final_trace, f, indent=4)
                output_queue.put((os.getpid(), sim_i, final_summary))
            except Exception as e:
                traceback.print_exc()
                print("pid[{}] Simulation {} with args {} failed to run. Skipping it..".format(
                    os.getpid(), sim_i, args_combination))
                output_queue.put((os.getpid(), sim_i, None))  # To signify the failure of this simulation
            except KeyboardInterrupt:
                print("pid[{}] process closed by user".format(os.getpid()))
                break
        print("pid[{}] finished process..".format(os.getpid()))

    @staticmethod
    def run_group(network_bandwidth, gpu_scheduler, network_scheduler, dag, batch_size, n_of_batches,
                  number_of_processes=None, saving_interval=5 * 60, print_interval=2, clear_output=True,
                  save_chrome_traces=False, join_traces_in_one_file = True, include_util_in_trace=True,
                  include_gaps=False, output_file_name="default"):
        """
        All of the required arguments can be either a value or a list of values to try.
        The product of all options will be taken and then iterated to generate simulations with all possible
        combinations.
        :param network_bandwidth: (Value or list of values) unit is Mbps
        :param gpu_scheduler: (Value or list of values)
        :param network_scheduler: (Value or list of values)
        :param dag: The dag to run. forward & backward pass units are assumed to be in nanoseconds.
        communication units is assumed to be in bytes. (Value or list of values)
        :param batch_size: How many forward, backward passes should we do before communicating? (Value or list of values)
        :param n_of_batches: How many batches or iterations should we run? (Value or list of values)
        :param number_of_processes: The number of independent processes to spawn. If set to None then the total number
        of effective cores will be used.
        Running all combinations of arguments can mean that hundreds of independent simulations
        will be run. However, a python interpreter spawns a single process which can only use one effective core which
        prevents us of using the full capacity of a multi-core processor. Which is why launching the simulations in a
        multi process setup when running on a multi-core cpu will speed it up tremendously.
        For reference:
        https://askubuntu.com/questions/949437/python-interpreter-only-using-12-cpu-power
        https://docs.python.org/3.6/library/multiprocessing.html
        :param saving_interval: Save report every x elapsed seconds. Set to None to disable interval saving
        :param print_interval: Print status at a maximum of every x elapsed seconds. If the simulator does not have any
        updates over the last print statement it will not print even after the interval.
        :param clear_output: Whether or not to clear output before each status print.
        :param save_chrome_traces: Whether or not we should save chrome traces of all simulations. This will choose to
        save timelines of the simulations in the 'jobwise' format.
        :param include_gaps: Whether or not we should include the gap durations of each simulation in the data. This
        also requires that timelines of the simulations are saved in 'jobwise' format.
        If both include_gaps and save_chrome_traces are set to false, then no timeline is kept for the simulations which
        can make them run faster.
        :param output_file_name: The name to save to, if set to 'default', then a default timestamped name is used.
        If set to None, then no saving will occur neither on intervals nor on the end of running.
        :return if return_simulation_objects is true then the function returns a list of all simulations run.
        Other wise a summary dict is returned
        {"resolution": resolution, "num_of_simulations": len(args_combinations), "args": [str(x) for x in args.items()],
        "results": results}
        """
        # Parse and generate argument combinations ---------------------------------------------------------------------
        if output_file_name == "default":
            output_file_name = "GpuNetworkSim_{}".format(datetime.datetime.now().strftime("%m-%d-%H-%M"))
        args = dict(network_bandwidth=network_bandwidth, gpu_scheduler=gpu_scheduler,
                    network_scheduler=network_scheduler, dag=dag, batch_size=batch_size, n_of_batches=n_of_batches)
        constants = set()
        for k, v in args.copy().items():
            try:
                if len(v) == 1:
                    constants.add(k)
            except TypeError:
                constants.add(k)
                args[k] = [args[k]]
        args_combinations = list(itertools.product(*args.values()))

        # Define printing functions ------------------------------------------------------------------------------------
        def print_header():
            print("Constant arguments: {}".format(constants))
            print("Variable arguments: {}".format(args.keys() - constants))
            print("-"*100)

        def print_state():
            nonlocal clear_output
            if clear_output:
                try:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    from IPython.display import clear_output
                    clear_output(wait=True)
                except:
                    pass
                print_header()
            # Update the running combinations
            while not running_queue.empty():
                pid, sim_i = running_queue.get_nowait()
                running_sim_indexes.update({pid: sim_i})
            # Print the current combinations of arguments
            print("{:20}: {:<.2f}%".format("progress", output_counter/len(args_combinations)*100))
            print("{:20}: {:<3}/{:<3}".format("successful_sims", output_counter - failed_simulations,
                                              len(args_combinations)))
            print("{:20}: {:<3}/{:<3}".format("failed_sims", failed_simulations, len(args_combinations)))
            print("{}{}{}".format("-" * 44, " Processes ", "-"*45))
            sub_args = dict(pid=list(), sim_index=list(), sims_processed=list())
            for pid, sim_i in running_sim_indexes.items():
                args_combination = args_combinations[sim_i]
                sub_args["pid"].append(pid)
                sub_args["sims_processed"].append(
                    finished_sims_per_process[pid] if pid in finished_sims_per_process else 0)
                sub_args["sim_index"].append(sim_i)
                for i, key in enumerate(args.keys()):
                    if key in sub_args:
                        sub_args[key].append(args_combination[i])
                    else:
                        sub_args[key] = [args_combination[i]]
            for arg_key, arg_value in sub_args.items():
                values = ""
                for v in arg_value:
                    values += "{:10} ".format(trim(str(v), 10))
                print("{:20}: {}".format(arg_key, values))
        if not clear_output:
            print_header()

        # Initialize summary dictionary --------------------------------------------------------------------------------
        summary = dict(total_time_elapsed=0, total_num_of_simulations=len(args_combinations),
                       finished_simulations=0, args=dict(), results=dict())
        for key, value in args.items():
            new_value = list()
            for v in value:
                try:
                    json.dumps(v)
                except TypeError:
                    v = str(v)
                new_value.append(v)
            summary["args"][key] = new_value
        group_begin_time = time.time()
        save_interval_begin_time = time.time()
        # Create worker processes --------------------------------------------------------------------------------------
        # Pickling is the main method used to communicate objects between processes. An object is teared down and
        # recreated each time it is passed between two processes. However, Locks cannot be pickled therefore we must
        # remove them from all schedulers. In addition, local functions cannot be pickled, therefore we defined the
        # worker process as a static function above this one.
        for sch in args["gpu_scheduler"]:
            sch._lock = None
        for sch in args["network_scheduler"]:
            sch._lock = None
        input_queue = multiprocessing.Queue()
        for sim_i, args_combination in enumerate(args_combinations):
            input_queue.put((sim_i, args_combination))
        output_queue = multiprocessing.Queue()
        running_queue = multiprocessing.Queue()
        running_sim_indexes = dict()
        finished_sims_per_process = dict()
        worker_pool = multiprocessing.Pool(number_of_processes, initializer=GpuNetworkSim._run_sub_group_process,
                                           initargs=[input_queue, running_queue, output_queue, args,
                                                     output_file_name if save_chrome_traces else None, include_gaps,
                                                     include_util_in_trace])
        worker_pool.close()  # Do not receive anymore requests
        # Run main process loop ----------------------------------------------------------------------------------------
        print("pid[{}] Main process loop starting".format(os.getpid()))
        output_counter = 0
        failed_simulations = 0
        print_timer = time.time()
        try:
            while output_counter != len(args_combinations):
                try:
                    output = output_queue.get(timeout=print_interval)
                except Empty:
                    continue
                pid, sim_i, sim_summary = output
                output_counter += 1
                if pid in finished_sims_per_process:
                    finished_sims_per_process[pid] += 1
                else:
                    finished_sims_per_process[pid] = 1
                if sim_summary is None:
                    failed_simulations += 1
                    continue
                for key, value in sim_summary.items():
                    if key in summary["results"]:
                        summary["results"][key].append(value)
                    else:
                        summary["results"][key] = [value]
                summary["total_time_elapsed"] = time.time() - group_begin_time
                summary["finished_simulations"] = output_counter
                if saving_interval is not None and output_file_name is not None and \
                        time.time() - save_interval_begin_time > saving_interval:
                    save_interval_begin_time = time.time()
                    with open("{}{}".format(output_file_name, SIM_GROUP_REPORT_POSTFIX), "w") as output_file:
                        json.dump(summary, output_file, indent=4)
                if time.time() - print_timer > print_interval:
                    print_timer = time.time()
                    print_state()
            print_state()
        except KeyboardInterrupt:
            print("pid[{}] Main process loop closed by user".format(os.getpid()))
        worker_pool.join()
        print("pid[{}] Main process is saving and returning results".format(os.getpid()))
        if len(summary["results"]) == 0:
            raise Exception("All simulations have failed.")
        sort_table(summary["results"], key="sim_index")
        if output_file_name is not None:
            with open("{}{}".format(output_file_name, SIM_GROUP_REPORT_POSTFIX), "w") as output_file:
                json.dump(summary, output_file, indent=4)
            if save_chrome_traces and join_traces_in_one_file:
                from schedule_simulator_core.utils import join_chrome_traces
                prefixes = list()
                traces = list()
                for i in range(len(args_combinations)):
                    try:
                        file_name = "{}_sim{}_.chrometrace.json".format(output_file_name, i)
                        with open(file_name) as trace_file:
                            traces.append(trace_file.read())
                            prefixes.append("sim{}".format(i))
                        os.remove(file_name)
                    except FileNotFoundError:
                        print("Trace file for simulation {} was not found. ignoring it".format(i))
                with open("{}.chrometrace.json".format(output_file_name), "w") as output:
                    output.write(join_chrome_traces(traces, prefixes=prefixes))
        return summary


if __name__ == "__main__":
    """
    Example usage
    """
    with open("../model_extraction/dags/VGG16_CPU.dag") as dag_file:
        base_dag = deserialize_dag(dag_file.read())
    schedulers = [FIFOScheduler(), TopologicalPriorityScheduler(preemptive=False),
                  TopologicalPriorityScheduler(preemptive=True)]
    bandwidths = [200]
    summary = GpuNetworkSim.run_group(network_bandwidth=bandwidths,
                                      gpu_scheduler=FIFOScheduler(),
                                      dag=base_dag,
                                      network_scheduler=schedulers,
                                      batch_size=4,
                                      n_of_batches=8,
                                      clear_output=False,
                                      number_of_processes=None,
                                      save_chrome_traces=True,
                                      include_gaps=True,
                                      include_util_in_trace=False
                                     )
