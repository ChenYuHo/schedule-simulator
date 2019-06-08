import simpy
from collections import deque
import copy
import schedulers
import math


class Job(simpy.Event):
    def __init__(self, env, units, **extras):
        super().__init__(env)
        self.env = env
        self.units = units
        self.remaining_units = units
        self.extras = extras

    def __str__(self):
        s = "Units: {:2}/{:<2} ".format(self.units - self.remaining_units, self.units)
        for item in self.extras.items():
            s += "{}: {} ".format(item[0],item[1])
        return s


class ProcessingUnit:
    """
    A processing unit base class that can simulate basically anything that processes units with a constant rate
    (ie. CPU, GPU, Aggregator, Network link ...etc)
    Needs a scheduler to operate.
    """
    def __init__(self, env: simpy.Environment, rate=1, name=None, out_pipe=None, sim_printer=None):
        self.name = name
        self.rate = rate
        self.env = env
        self.scheduler = None
        self.out_pipe = out_pipe
        self.processing = False
        self._sim_printer = sim_printer
        self.utilization = dict()

    def mount_scheduler(self, scheduler):
        self.scheduler = scheduler

    def queue(self, job):
        try:
            self.scheduler.queue(job)
        except AttributeError as e:
            print("[Error] Please make sure that you have mounted a valid scheduler on {}".format(self))
            raise e
        self._print("Queued job {}".format(job), 2)

    def main_process(self):
        self._print("Starting main process with processing rate: {}".format(self.rate), 1)
        try:
            while True:
                # Finish as many jobs as you can given the rate per time step
                iteration_rate = self.rate
                will_finish_jobs = []
                current_job = None
                while self.scheduler.count() > 0 and iteration_rate > 0:
                    job = self.scheduler.request()
                    to_process = min(iteration_rate, job.remaining_units)
                    iteration_rate -= to_process
                    job.remaining_units -= to_process
                    if job.remaining_units == 0:
                        will_finish_jobs.append((job, to_process))
                        self.scheduler.remove(job)
                    else:
                        current_job = (job,to_process)
                # Add utilization info
                utilization_list = list()
                for job in will_finish_jobs:
                    utilization_list.append(job)
                if current_job is not None:
                    utilization_list.append(current_job)
                if len(utilization_list) > 0:
                    self.utilization[self.env.now] = utilization_list
                # Step
                yield self.env.timeout(1)
                # Finalize finished jobs
                for job,_ in will_finish_jobs:
                    job.succeed()  # Fire job finished event
                    if self.out_pipe is not None:
                        self.out_pipe.queue(job)
                    self._print("Finished job {}".format(job), 2)
        except InterruptedError:
            self._print("Closed main process.", 1)
        except AttributeError as e:
            print("[Error] Please make sure that you have mounted a valid scheduler on {}".format(self))
            raise e

    def _print(self, msg, verbosity):
        if self._sim_printer:
            self._sim_printer(env=self.env, source=self, msg=msg, verbosity=verbosity)

    def __str__(self):
        return "{}".format(self.name if self.name is not None else type(self).__name__)


if __name__ == '__main__':
    """
    An example usage of the processing unit
    """
    from io_utils import SimPrinter, generate_report
    import random
    env = simpy.Environment()
    gpu = ProcessingUnit(env, rate=1, name="GPU:1", sim_printer=SimPrinter(verbosity=0).print)
    gpu.mount_scheduler(schedulers.FIFOScheduler())
    gpu_process = env.process(gpu.main_process())
    gpu.queue(Job(env,units=3,L=1))
    gpu.queue(Job(env,units=5,L=2))
    for i in range(10):
        job = Job(env,units=random.randint(0,10), name=i, tag="Coco")
        gpu.queue(job)
        job = Job(env, units=random.randint(0, 10), name=i)
        gpu.queue(job)
    env.run(until=300)
    print(generate_report(gpu, start=0,time_grouping=50,row_labels=["L"], cell_labels="count"))