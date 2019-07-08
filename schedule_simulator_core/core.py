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
    A processing unit base class that can simulate basically anything that processes units with a constant rate
    (ie. CPU, GPU, Aggregator, Network link ...etc)
    Needs a scheduler to operate.
    """
    def __init__(self, env: simpy.Environment, scheduler, rate=1, name=None, out_pipe=None, sim_printer=None,
                 timeline_format="jobwise"):
        """
        :param env: The simpy environment used in this simulation.
        :param rate: The rate at which the unit consumes job units
        :param name: An arbitrary name for this unit. (Maybe removed later and replaced by an extras field)
        :param out_pipe: An optional output queue. Used for pipelining units.
        :param sim_printer: The print function that will be used to print the output of this unit
        :param timeline_format:
        "stepwise": dict(key: time step ,value: list of (job, processed units) tuples that were done in that time step)
        It is a very inefficient format for simulations with a very large number of steps. This format can be used to
        generate an ASCII timeline table using the generate_ascii_timeline method in utils
        "jobwise": dict(key: job, value: list of {"ts": starting time step, "dur": duration, "pu": processed_units}
        This format is much more memory efficient but slightly less cpu efficient. It can be used to generate a chrome
        trace using the generate_chrome_trace_timeline method in utils.
        None: If the format is set to None then no timeline is kept. Can be useful when only the final results of the
        simulation are relevant.
        """
        self.name = name
        self.rate = rate
        self.env = env
        self.scheduler = scheduler
        self.out_pipe = out_pipe
        self.processing = False
        self._sim_printer = sim_printer
        self.timeline_format = timeline_format
        if self.timeline_format is not None:
            self.timeline = dict()
        self.total_processed_units = 0

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
                    self.total_processed_units += to_process
                    iteration_rate -= to_process
                    job.remaining_units -= to_process
                    if job.remaining_units == 0:
                        will_finish_jobs.append((job, to_process))
                        self.scheduler.remove(job)
                    else:
                        current_job = (job, to_process)
                # Add timeline info
                if self.timeline_format is not None:
                    timeline_list = will_finish_jobs.copy()
                    if current_job is not None:
                        timeline_list.append(current_job)
                    if self.timeline_format == "stepwise":
                        if len(timeline_list) > 0:
                            self.timeline[self.env.now] = timeline_list
                    elif self.timeline_format == "jobwise":
                        for job, processed_units in timeline_list:
                            if job in self.timeline:
                                found_event = None
                                for event in self.timeline[job]:
                                    # Should we extend this event?
                                    if event["ts"] + event["dur"] == self.env.now:
                                        found_event = event
                                        break
                                if found_event is None:
                                    self.timeline[job].append(dict(ts=self.env.now, dur=1, pu=processed_units))
                                else:
                                    found_event["dur"] += 1
                                    found_event["pu"] += processed_units
                            else:
                                self.timeline[job] = [dict(ts=self.env.now, dur=1, pu=processed_units)]
                # Finalize finished jobs
                # We do this before waiting to allow processes that are waiting for the job to be notified instantly
                # in the next time step. Otherwise, the processes will always be delayed 1 time step
                for job, processed_units in will_finish_jobs:
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

    def get_utilization(self, start=None, end=None, extras=None):
        """
        A function for returning the utilization percentage of a single group using the timeline
        :param start: The utilization period start. If None then 0 is used.
        :param end: The utilization period end. If None then the last time step in the environment is used.
        :param extras: An extras dictionary that will be used to match each job's extras dictionary.
        Only if all the key,value pairs in this dict are present in the job.extras dict will the function include it
        :return: A 0-1 floating point value representing the utilization percentage
        """
        if start is None:
            start = 0
        if end is None:
            end = self.env.now
        total_processed_units = 0
        duration = end - start + 1
        total_rate_units = self.rate * duration
        if self.timeline_format is None:
            if start is None and end is None and extras is None:
                return self.total_processed_units / total_rate_units
            else:
                raise Exception("Cannot pass start, end, or extras options with a unit that stores no timeline.")
        if self.timeline_format == "stepwise":
            for t in self.timeline.keys():
                if t < start or t > end:
                    continue
                for job, processed_units in self.timeline[t]:
                    if job.match_extras(extras):
                        total_processed_units += processed_units
        elif self.timeline_format == "jobwise":
            """
            Since jobwise format does not keep per step information like the stepwise format. We can only average the
            processed units over the duration of the job event to get the per time step units. This is a good
            approximation however it is not fully accurate.
            """
            for job, events in self.timeline.items():
                if not job.match_extras(extras):
                    continue
                for event in events:
                    # Find overlapping interval
                    os = max(start, event["ts"])
                    oe = min(end, event["ts"] + event["dur"])
                    if os >= oe:  # No overlap
                        continue
                    total_processed_units += event["pu"]/event["dur"] * (oe-os)

        else:
            raise Exception("Timeline format not supported")
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
    gpu = ProcessingUnit(env, schedulers.FIFOScheduler(), timeline_format="stepwise",
                         rate=2, name="GPU:1", sim_printer=SimPrinter(verbosity=0).print)
    gpu_process = env.process(gpu.main_process())
    for i in range(10):
        job = Job(env, units=random.randint(1, 10), custom_attr_1=i, custom_attr_2=i % 3)
        gpu.queue(job)
    env.run(until=100)
    print(generate_ascii_timeline(gpu, start=0, time_grouping=1, row_labels=["custom_attr_1"]))
