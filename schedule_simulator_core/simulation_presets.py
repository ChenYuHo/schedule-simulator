"""
This module provides preset simulation setups for quick reuse
"""

from schedule_simulator_core.core import ProcessingUnit
from schedule_simulator_core.schedulers import FIFOScheduler
from schedule_simulator_core.DNN_functions import train
from schedule_simulator_core.io_utils import SimPrinter, generate_report
import simpy


class GpuNetworkSchedulingSim:
    def __init__(self, gpu_rate, network_rate, schedulers, dag):
        self.gpu_rate = gpu_rate
        self.network_rate = network_rate
        self.schedulers = schedulers
        self.dag = dag

    def run_simulation(self, batch_size, n_of_batches, print_report=True, verbosity=1, report_args=None):
        results = dict()
        printer = SimPrinter(verbosity=verbosity).print
        for scheduler in self.schedulers:
            env = simpy.Environment()

            gpu = ProcessingUnit(env=env, scheduler=FIFOScheduler(), rate=2, name="GPU", sim_printer=printer)
            gpu_process = env.process(gpu.main_process())

            network = ProcessingUnit(env=env, scheduler=scheduler, rate=1, name="Network", sim_printer=printer)
            network_process = env.process(network.main_process())

            training_process = env.process(train(dag=self.dag, env=env, n_of_batches=n_of_batches,
                                                 batch_size=batch_size, computation_queue=gpu,
                                                 communication_queue=network))

            def close():
                yield training_process
                gpu_process.interrupt()
                network_process.interrupt()
            closing_process = env.process(close())
            results[scheduler] = (gpu, network)
            env.run()
        if print_report:
            args = dict(time_grouping=1, row_labels=["type"], cell_labels=["index"], group_name_width=30, cell_width=4)
            if report_args is not None:
                args.update(report_args)
            for scheduler in results:
                for unit in results[scheduler]:
                    report = generate_report(unit, **args)
                    print(report)
                    print("")
        return results
