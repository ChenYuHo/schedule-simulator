import simpy
from collections import deque


class FIFOScheduler:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.queue = deque()
        self.state = "IDLE"
        # Usage structure is a dictionary of dictionaries. Each layer's index is the first key.
        # The second key is the time. and the value is the tensor index
        self.usage = dict()

    def schedule(self, tensor):
        self.queue.append(tensor)

    def main_process(self):
        print("Starting {} main process".format(type(self).__name__))
        self.usage.clear()
        self.state = "RUNNING"
        while self.state != "CLOSED":
            if len(self.queue) > 0:
                tensor = self.queue[0]
                yield self.env.timeout(1)
            else:
                yield self.env.timeout(1)
                continue
            tensor.size -= 1  # Should be replaced later by bandwidth term
            if tensor.size <= 0:
                self.queue.popleft()
            # Write stats
            if tensor.layer.index not in self.usage:
                self.usage[tensor.layer.index] = dict()
            self.usage[tensor.layer.index][self.env.now-1] = tensor.index

    def close(self):
        self.state = "CLOSED"


class PreemptiveScheduler:
    pass


class TICScheduler:
    pass


class TACScheduler:
    pass
