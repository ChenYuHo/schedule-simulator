from queue import Queue, PriorityQueue, LifoQueue

q = Queue()

q.put_nowait(0)
q.put_nowait(1)
q.put_nowait(2)

print(q[0])