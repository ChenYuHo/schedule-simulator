import simpy
import core
from collections import deque
from threading import Lock

"""
This module provides multiple low level schedulers that can be used to prioritize any tasks or jobs in a certain way.
Each processing unit must have a unique scheduler in which it uses to choose which task to work on.
You should not deal with any of the schedulers methods directly. Instead you simply pass an instance of it the processing
unit and it will handle the rest.
"""


class FIFOScheduler:
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


class PreemptiveScheduler:
    pass


class TICScheduler:
    pass


class TACScheduler:
    pass
