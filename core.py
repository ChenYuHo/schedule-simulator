import simpy
from collections import deque
import copy
import schedulers
import math

class Job(simpy.Event):
    def __init__(self, env, units, source=None, name=None, tag=None, priority=None):
        super().__init__(env)
        self.env = env
        self.units = units
        self.remaining_units = units
        self.source = source
        self.name = name
        self.tag = tag
        self.priority = priority
        self._processor = None

    def __str__(self):
        return "{}{}{}{}{}".format(
            "Name: {:<10} ".format(self.name) if self.name is not None else "",
            "Tag: {:<3} ".format(self.tag) if self.tag is not None else "",
            "Src: {:<10} ".format(self.source) if self.source is not None else "",
            "Units: {:2}/{:<2} ".format(self.units - self.remaining_units, self.units),
            "Priority: {:<2} ".format(self.priority) if self.priority is not None else ""
        )


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
                        will_finish_jobs.append(job)
                        self.scheduler.remove(job)
                    else:
                        current_job = job
                # Add utilization info
                # We copy the jobs instead of passing a reference so that the state of the job at that time is preserved
                # Not RAM efficient but its a simple simulator after all so we can get away with it
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
                    if self.out_pipe is not None:
                        self.out_pipe.queue(job)
                    self._print("Finished job {}".format(job), 2)
        except InterruptedError:
            self._print("Closed main process.", 1)
        except AttributeError as e:
            print("[Error] Please make sure that you have mounted a valid scheduler on {}".format(self))
            raise e

    def get_average_utilization(self, start=0, end=None, source=None, name=None, tag=None):
        total = 0
        for item in self.utilization.items():
            t = item[0]
            if t >= start and (end is None or t <= end):
                total += item[1]['%']
        return total/self.env.now

    def generate_report(self, start=0, end=None,
                        time_grouping=1, show_scaled_time=False,
                        row_labels=None, cell_labels=None,
                        show_column_labels=True, show_row_utilization=True, show_header=True,
                        long_value_handling="trim"):
        """
        :param start: From which time step should we generate the report (Inclusive)
        :param end: To which time step (Inclusive)
        :param time_grouping: How many time steps per column. If set to 0 or
        None then only one column is used regardless of time steps
        :param show_scaled_time: If time_group=5 then a value of 3 means t=15 if this is set to true
        :param row_labels: Group rows using none or all of ["source","name","tag"]. If an empty list is passed or None,
        then one group is used for the whole processing unit
        :param cell_labels: Label cells using none or all of ["source","name","tag"]. If an empty list is passed or None,
        then the cell is simply marked as has job "X" or empty "-"
        :param show_column_labels: Whether to show column labels (time steps) or not
        :param show_row_utilization: Whether to print utilization percentage for every row (group)
        :param show_header: Whether to print a header with the unit name and the average utilization of the resource
        :param long_value_handling: "trim", "wrap", "push"
        """

        cell_width = 5
        row_header_width = 10
        report = []
        duration = (env.now if end is None else end) - start
        if not time_grouping:
            time_scale = duration
        scaled_start = math.floor(start/time_grouping)
        scaled_end = math.ceil((env.now if end is None else end) / time_grouping)
        grouped_time_steps = range(scaled_start, scaled_end)
        avg_util = self.get_average_utilization(start, end)
        if show_header:
            report.append("{} Util: {:<3.2f}% -------------------------------------------".format(self, avg_util * 100))

        if show_column_labels:
            if show_scaled_time:
                column_labels = "t({{:<{}}}): ".format(row_header_width-5).format(time_grouping)
            else:
                column_labels = "t{{:<{}}}: ".format(row_header_width-3).format(" ")
            for i in grouped_time_steps:
                column_labels += "{{:{}}}".format(cell_width).format(i if show_scaled_time else i*time_grouping)
            report.append(column_labels)
        # Generate groupings

        return '\n'.join(report)


    def _print(self, msg, verbosity):
        if self._sim_printer:
            self._sim_printer(env=self.env, source=self, msg=msg, verbosity=verbosity)

    def __str__(self):
        return "{}".format(self.name if self.name is not None else type(self).__name__)


if __name__ == '__main__':
    """
    An example usage of the processing unit
    """
    from io_utils import SimPrinter
    import random
    env = simpy.Environment()
    gpu = ProcessingUnit(env, rate=1, name="GPU:1", sim_printer=SimPrinter(verbosity=1).print)
    gpu.mount_scheduler(schedulers.FIFOScheduler())
    gpu_process = env.process(gpu.main_process())
    for i in range(10):
        job = Job(env,units=random.randint(0,10),name=i)
        gpu.queue(job)
    env.run(until=500)
    print(gpu.generate_report(start=0,time_grouping=5))
    # for item in gpu.utilization.items():
    #     s = [str(x) for x in item[1]['jobs']]
    #     print("t:{:<4}] Utilization: {:6}% Jobs: {}".format(item[0],item[1]['%']*100, s))