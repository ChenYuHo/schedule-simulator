import simpy
from simpy import Interrupt
from schedule_simulator_core import schedulers


class Distribution:
    """
    It can define a constant, or any distribution
    Can be used for any numeric property to express some randomness.
    """
    def __init__(self, distribution, integer=True, positive=True, *args):
        """
        :param distribution: The distribution function.
        See https://docs.scipy.org/doc/numpy/reference/routines.random.html#distributions for all distributions
        If set to none then the args field is returned (Constant).
        :param args: The arguments needed for the distribution or the value of the constant.
        """
        self.distribution = distribution
        self.args = args
        self.integer = integer
        self.positive = positive

    def __next__(self):
        if self.distribution is None:
            v = self.args
        else:
            v = self.distribution(*self.args)
        if self.integer:
            v = int(v)
        if self.positive:
            v = max(v, 0)
        return v

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
            s += "{}: {} ".format(item[0], item[1])
        return s

    def match_extras(self, extras):
        if extras is not None:
            for key in extras.keys():
                if key not in self.extras or self.extras[key] != extras[key]:
                    return False
        return True


class ProcessingUnit:
    """
    A processing unit base class that can simulate basically anything that processes units with rate
    (ie. CPU, GPU, Aggregator, Network link ...etc)
    Needs a scheduler to operate.
    """
    def __init__(self, env: simpy.Environment, scheduler, rate=1.0, name=None, out_pipe=None, sim_printer=None,
                 keep_timeline=True):
        """
        :param env: The simpy environment used in this simulation.
        :param rate: The rate at which the unit consumes job units
        :param name: An arbitrary name for this unit. (Maybe removed later and replaced by an extras field)
        :param out_pipe: An optional output queue. Used for pipelining units.
        :param sim_printer: The print function that will be used to print the output of this unit
        :param keep_timeline: Whether to keep a timeline or not. The timeline format is in the following format
        dict(key: job, value: list of {"ts": starting time step, "dur": duration, "pu": processed_units}
        This format is much more memory efficient but slightly less cpu efficient. It can be used to generate a chrome
        trace using the generate_chrome_trace_timeline method in utils.
        """

        self.name = name
        self.rate = rate
        self.env = env
        self.scheduler = scheduler
        self.out_pipe = out_pipe
        self.processing = False
        self._sim_printer = sim_printer
        self.close_event = simpy.Event(self.env)
        self.add_event = simpy.Event(self.env)
        if keep_timeline is not None:
            self.timeline = dict()
        else:
            self.timeline = None
        self.total_processed_units = 0

    def queue(self, job):
        """
        Simply calls the scheduler queue function.
        Must have attached a valid scheduler before calling this function.
        :param job: the job to queue.
        """
        try:
            self.scheduler.queue(job)
            if not self.add_event.triggered:
                self.add_event.succeed()
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
        current_job = None
        current_job_starting_time = None
        finished = False
        try:
            while True:
                to_work_on = self.scheduler.request()
                if to_work_on != current_job:
                    # We will start working on something new or we finished the last job.
                    # Let us update states to include the work that has been done on the last job.
                    if current_job is not None:
                        dur = self.env.now - current_job_starting_time
                        pu = dur * self.rate
                        if not finished:
                            current_job.remaining_units -= pu
                        else:
                            # We set it to 0 instead of relying on subtraction because of floating point precision
                            current_job.remaining_units = 0
                            current_job.succeed()
                            if current_job.result is not None and self.out_pipe is not None:
                                self.out_pipe.queue(current_job.result)
                            self._print("Finished job {}".format(current_job), 2)
                        if self.timeline is not None:
                            timeline_event = {"ts": current_job_starting_time, "dur": dur, "pu": pu}
                            if current_job in self.timeline.keys():
                                self.timeline[current_job].append(timeline_event)
                            else:
                                self.timeline[current_job] = [timeline_event]
                        self.total_processed_units += pu
                        finished = False
                    current_job = to_work_on
                    current_job_starting_time = self.env.now
                if current_job is None:
                    # If we are not working then, wait until we get a new add event
                    yield self.add_event
                else:
                    # If we are working then, wait until we finish or we get a new add event
                    delay_needed = current_job.remaining_units / self.rate
                    elapsed = self.env.now - current_job_starting_time
                    yield simpy.AnyOf(self.env, [self.env.timeout(delay_needed-elapsed), self.add_event])
                if self.add_event.triggered:
                    # We have received a new job
                    self.add_event = simpy.Event(self.env)
                else:
                    # We have finished the current job. Recording its info will happen in the next iteration.
                    self.scheduler.remove(current_job)
                    finished = True
        except Interrupt:
            self._print("Closed main process.", 1)
        except AttributeError as e:
            print("[Error] Please make sure that you have mounted a valid scheduler on {}".format(self))
            raise e

    def close(self):
        self.close_event.succeed()

    def get_utilization(self, start=None, end=None, extras=None):
        """
        A function for returning the utilization percentage of a single group using the timeline
        :param start: The utilization period start. If None then 0 is used.
        :param end: The utilization period end. If None then last event time in the environment is used.
        :param extras: An extras dictionary that will be used to match each job's extras dictionary.
        Only if all the key,value pairs in this dict are present in the job.extras dict will the function include it
        :return: A 0-1 floating point value representing the utilization percentage
        """
        if start is None:
            start = 0
        if end is None:
            end = self.env.now
        total_processed_units = 0
        duration = end - start
        if duration == 0:
            return 0
        total_rate_units = self.rate * duration
        if self.timeline is None:
            if start is None and end is None and extras is None:
                return self.total_processed_units / total_rate_units
            else:
                raise Exception("Cannot pass start, end, or extras options with a unit that stores no timeline.")
        else:
            for job, events in self.timeline.items():
                if not job.match_extras(extras):
                    continue
                for event in events:
                    # Find overlapping interval
                    os = max(start, event["ts"])
                    oe = min(end, event["ts"] + event["dur"])
                    if os >= oe:  # No overlap
                        continue
                    total_processed_units += self.rate * (oe-os)
        return total_processed_units / total_rate_units

    def _print(self, msg, verbosity):
        if self._sim_printer:
            self._sim_printer(env=self.env, source=self, msg=msg, verbosity=verbosity)

    def __str__(self):
        return "{}".format(self.name if self.name is not None else type(self).__name__)


if __name__ == '__main__':
    """
    An example usage of the processing unit
    """
    from schedule_simulator_core.utils import SimPrinter, generate_ascii_timeline, generate_chrome_trace_timeline
    import random
    env = simpy.Environment()
    gpu = ProcessingUnit(env, schedulers.FIFOScheduler(), keep_timeline=True,
                         rate=1/3, name="GPU:1", sim_printer=SimPrinter(verbosity=3).print)
    gpu_process = env.process(gpu.main_process())
    for i in range(100):
        job = Job(env, units=random.randint(1, 10)/1e50, custom_attr_1=i, custom_attr_2=i % 3)
        gpu.queue(job)
    env.run()
