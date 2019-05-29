import simpy
from collections import deque
import copy


class ProcessingUnit:
    """
    A processing unit base class that can simulate basically anything that processes units with a constant rate
    (ie. CPU, GPU, Aggregator, Network link ...etc)
    Uses a simple FIFO queue without scheduling & without sharing resources.
    """
    def __init__(self, env: simpy.Environment, rate=1, name=None, out_buffer=None, printer=None):
        self.name = name
        self.rate = rate
        self.env = env
        self.in_buffer = deque()
        self.out_buffer = out_buffer
        self.processing = False
        self._printer = printer
        self.utilization = dict()

    class Job(simpy.Event):
        def __init__(self, env, units, source=None, name=None, tag=None):
            super().__init__(env)
            self.env = env
            self.units = units
            self.remaining_units = units
            self.source = source
            self.name = name
            self.tag = tag

        def __str__(self):
            return "{}{}{}{}".format(
                "Name: {:<10} ".format(self.name) if self.name is not None else "",
                "Tag: {:<3} ".format(self.tag) if self.tag is not None else "",
                "Src: {:<10} ".format(self.source) if self.source is not None else "",
                "Units: {:2}/{:<2} ".format(self.units-self.remaining_units, self.units)
            )

    def queue_job(self, units, source=None, name=None, tag=None):
        """
        A job object can have the following three properties:
        - computation_units(Mandatory): Simulates the computational cost of this job.
          If computation_units=10 and rate=2 then the job will complete in 5 time steps.
        - name(Optional): Name of issuing process
        - index(Optional): Job index in case the issuing process issues multiple jobs
        """
        job = ProcessingUnit.Job(self.env, units, source, name, tag)
        self.in_buffer.append(job)
        self._print("Queued job {}".format(job), 2)
        return job

    def main_process(self):
        self._print("Starting main process with processing rate: {}".format(self.rate), 1)
        try:
            while True:
                # Finish as many jobs as you can given the rate per time step
                iteration_rate = self.rate
                will_finish_jobs = []
                current_job = None
                job_index = 0
                while len(self.in_buffer) > job_index+1 and iteration_rate > 0:
                    job = self.in_buffer[job_index]
                    to_process = min(iteration_rate, job.remaining_units)
                    iteration_rate -= to_process
                    job.remaining_units -= to_process
                    if job.remaining_units == 0:
                        will_finish_jobs.append(job)
                    else:
                        current_job = job
                    job_index += 1
                # Add utilization info
                # We copy the jobs instead of passing a reference so that the state of the job at that time is preserved
                # Not RAM efficient but its a simple simulator after all
                utilization_list = list()
                for job in will_finish_jobs:
                    utilization_list.append(copy.copy(job))
                if current_job is not None:
                    utilization_list.append(copy.copy(current_job))
                if len(utilization_list) > 0:
                    self.utilization[self.env.now] = {'%': (self.rate - iteration_rate)/self.rate,
                                                      'jobs': utilization_list}
                # Step
                yield self.env.timeout(1)
                # Finalize finished jobs
                for job in will_finish_jobs:
                    job.succeed()  # Fire job finished event
                    self.in_buffer.popleft()
                    self._print("Finished job {}".format(job), 2)
        except InterruptedError:
            self._print("Closed main process.", 1)

    def print_utilization_report(self, time_step=None):
        pass

    def _print(self, msg, verbosity):
        if self._printer:
            self._printer(env=self.env, source=self, msg=msg, verbosity=verbosity)

    def __str__(self):
        return "{}".format(self.name if self.name is not None else type(self).__name__)


# if __name__ == '__main__':
#     """
#     An example usage of the processing unit
#     """
#     from basic_simulator.io_utils import SimPrinter
#     import random
#     env = simpy.Environment()
#     gpu = ProcessingUnit(env, rate=50, name="GPU:1", printer=SimPrinter().print)
#     gpu_process = env.process(gpu.main_process())
#     for i in range(10):
#         gpu.queue_job(random.randint(0,10), tag=i)
#     env.run(until=50)
#     gpu.print_utilization_report()
#     for item in gpu.utilization.items():
#         s = [str(x) for x in item[1]['jobs']]
#         print("t:{:<4}] Utilization: {:6}% Jobs: {}".format(item[0],item[1]['%']*100, s))