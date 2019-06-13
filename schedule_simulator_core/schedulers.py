from collections import deque
from threading import Lock
from abc import ABC, abstractmethod

"""
This module provides multiple low level schedulers that can be used to prioritize any tasks or jobs in a certain way.
Each processing unit must have a unique scheduler in which it uses to choose which task to work on.
You should not deal with any of the schedulers methods directly. Instead you simply pass an instance of it the processing
unit and it will handle the rest.

I did not use the synchronized queue module in any of the schedulers and synchronized manually.
This is because the queue module does not support peeking for some reason.
"""


class Scheduler(ABC):
    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def queue(self, job, **kwargs):
        pass

    @abstractmethod
    def request(self, **kwargs):
        pass

    @abstractmethod
    def remove(self, job, **kwargs):
        pass

    def __str__(self):
        return type(self).__name__


class FIFOScheduler(Scheduler):
    def __init__(self, **kwargs):
        self._queue = deque()
        self._lock = Lock()

    def count(self):
        self._lock.acquire()
        a = len(self._queue)
        self._lock.release()
        return a

    def queue(self, job, **kwargs):
        self._lock.acquire()
        self._queue.append(job)
        self._lock.release()

    def request(self, **kwargs):
        self._lock.acquire()
        job = None
        if len(self._queue) > 0:
            job = self._queue[0]
        self._lock.release()
        return job

    def remove(self, job, **kwargs):
        self._lock.acquire()
        self._queue.remove(job)
        self._lock.release()


class TopologicalPriorityScheduler(Scheduler):
    """
    Prioritizes based on the DAG dependencies or the topological order of the layers
    Uses a simple inefficient O(n) implementation of a priority queue implemented using an array list.
    I did not use the heapq module because we will not be necessarily always removing the top of the heap
    """
    def __init__(self, **kwargs):
        self._lock = Lock()
        self._queue = list()
        if "preemptive" in kwargs.keys():
            self.preemptive = kwargs['preemptive']
        else:
            self.preemptive = False
        self._current_job = None

    def count(self):
        self._lock.acquire()
        a = len(self._queue)
        self._lock.release()
        return a

    def queue(self, job, **kwargs):
        self._lock.acquire()
        priority = len(job.source.forward_dependencies)
        self._queue.append((priority, job))
        self._lock.release()

    def request(self, **kwargs):
        self._lock.acquire()
        if self.preemptive or self._current_job is None:
            min_priority = None
            min_job = None
            for priority, job in self._queue:
                if min_priority is None or priority < min_priority:
                    min_priority = priority
                    min_job = job
            self._current_job = min_job
        self._lock.release()
        return self._current_job

    def remove(self, job, **kwargs):
        self._lock.acquire()
        if job == self._current_job:
            self._current_job = None
        for i, item in enumerate(self._queue):
            if item[1] == job:
                self._queue.pop(i)
                break
        self._lock.release()

    def __str__(self):
        return "{}:{}".format(super().__str__(), "Preemptive" if self.preemptive else "Non-preemptive")


class TICScheduler:
    pass


class TACScheduler:
    pass
