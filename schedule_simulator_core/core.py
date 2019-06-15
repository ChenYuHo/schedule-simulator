import simpy
from simpy import Interrupt
from schedule_simulator_core import schedulers


class Distribution:
    """
    It can define a constant, or any distribution
    Can be used for any numeric property to express some randomness.
    """
    def __init__(self, distribution, integer=True, *args):
        """
        :param distribution: The distribution function.
        See https://docs.scipy.org/doc/numpy/reference/routines.random.html#distributions for all distributions
        If set to none then the args field is returned (Constant).
        :param args: The arguments needed for the distribution or the value of the constant.
        """
        self.distribution = distribution
        self.args = args
        self.integer = integer

    def __next__(self):
        if self.distribution is None:
            return self.args
        else:
            return self.distribution(*self.args)

    def generate_value(self):
        return self.__next__()


class Job(simpy.Event):
    """
    A simpy event that is used to simulate any kind of job that is fed into a processing unit
    """
    def __init__(self, env: simpy.Environment, units, source=None, result=None, **extras):
        """
        :param env: The simpy environment used in this simulation. Used to be able to block and wait for this event
        :param units: The amount of work needed. Depends on the nature of the processing unit.
        Can be used to describe sizes (KB, MB...), time (seconds, hours...) and any other unit..
        :param result: The result object that can be used to chain jobs.
        :param extras: Custom attributes that help identify this job or its behavior.
        """
        super().__init__(env)
        self.env = env
        self.units = units
        self.source = source
        self.remaining_units = units
        self.extras = extras
        self.result = result

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
    def __init__(self, env: simpy.Environment, scheduler, rate=1, name=None, out_pipe=None, sim_printer=None):
        """
        :param env: The simpy environment used in this simulation.
        :param rate: The rate at which the unit consumes job units
        :param name: An arbitrary name for this unit. (Maybe removed later and replaced by an extras field)
        :param out_pipe: An optional output queue. Used for pipelining units.
        :param sim_printer: The print function that will be used to print the output of this unit
        """
        self.name = name
        self.rate = rate
        self.env = env
        self.scheduler = scheduler
        self.out_pipe = out_pipe
        self.processing = False
        self._sim_printer = sim_printer
        # The utilization structure:
        # Key: time step
        # Value: list of (job, processed units) tuples that were done in that time step
        self.utilization = dict()

    def queue(self, job):
        """
        Simply calls the scheduler queue function.
        Must have attached a valid scheduler before calling this function.
        :param job: the job to queue.
        """
        try:
            self.scheduler.queue(job)
        except AttributeError as e:
            print("[Error] Please make sure that you have provided a valid scheduler on {}".format(self))
            raise e
        self._print("Queued job {}".format(job), 2)

    def main_process(self):
        """
        The main process of this processing unit. Constantly requests jobs from the scheduler and proceeds to consume
        them according to its processing rate. The process continues indefinitely until it is interrupted.
        """
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
                # Finalize finished jobs
                # We do this before waiting to allow processes that are waiting for the job to be notified instantly
                # in the next time step. Otherwise, the processes will always be delayed 1 time step
                for job,_ in will_finish_jobs:
                    job.succeed()  # Fire job finished event
                    if self.out_pipe is not None and job.result is not None:
                        self.out_pipe.queue(job.result)
                    self._print("Finished job {}".format(job), 2)
                # Step
                yield self.env.timeout(1)
        except Interrupt:
            self._print("Closed main process.", 1)
        except AttributeError as e:
            print("[Error] Please make sure that you have mounted a valid scheduler on {}".format(self))
            raise e

    def get_utilization(self, start, end, group=None):
        pass

    def _print(self, msg, verbosity):
        if self._sim_printer:
            self._sim_printer(env=self.env, source=self, msg=msg, verbosity=verbosity)

    def __str__(self):
        return "{}".format(self.name if self.name is not None else type(self).__name__)


if __name__ == '__main__':
    """
    An example usage of the processing unit
    """
    from schedule_simulator_core.io_utils import SimPrinter, generate_report
    import random
    env = simpy.Environment()
    gpu = ProcessingUnit(env, schedulers.FIFOScheduler(),
                         rate=1, name="GPU:1", sim_printer=SimPrinter(verbosity=0).print)
    gpu_process = env.process(gpu.main_process())
    for i in range(10):
        job = Job(env, units=random.randint(1, 10), custom_attr_1=i, custom_attr_2=i % 3)
        gpu.queue(job)
    env.run(until=100)
    print(generate_report(gpu, start=0, time_grouping=1, row_labels=["custom_attr_1"]))
