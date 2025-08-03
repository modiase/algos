from typing import Callable
from datetime import datetime, timedelta
import time
import heapq
import random

MILLISECONDS_IN_SECOND = 1000


class Task:
    def __init__(self, f: Callable, n: int):
        self._callback = f
        self._time_due = datetime.now() + timedelta(milliseconds=n)

    def get_time_due(self):
        return self._time_due

    def __call__(self, *args, **kwargs):
        return self._callback(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)


class JobQueue:
    def __init__(self):
        self._queue = []
        self._internal_clock = datetime.now()
        self._DELTA = 10

    def advance(self):
        try:
            _next = heapq.heappop(self._queue)
        except IndexError:
            time.sleep(self._DELTA / MILLISECONDS_IN_SECOND)
            self._update_internal_clock(datetime.now())
        else:
            sleep_time = _next[0] - self._internal_clock.timestamp()
            if sleep_time > 0:
                time.sleep(sleep_time)
            _next[1]()
            self._update_internal_clock(datetime.now())

    def run(self):
        while 1:
            self.advance()

    def _update_internal_clock(self, t: datetime):
        self._internal_clock = t

    def queue_task(self, f: Callable, n: int):
        t = Task(f, n)
        T = (datetime.now() + timedelta(milliseconds=n)).timestamp()
        heapq.heappush(self._queue, (T, t))

    def queue_task(self, t: Task):
        T = t.get_time_due().timestamp()
        heapq.heappush(self._queue, (T, t))


def say_hello():
    print("Hello, World!")


if __name__ == "__main__":
    jq = JobQueue()
    tasks = [
        Task(say_hello, int(random.random() * 10000))
        if random.random() < 0.05
        else Task(lambda: jq.queue_task(Task(say_hello, 100)), 100)
        for _ in range(100)
    ]
    for task in tasks:
        jq.queue_task(task)
    jq.run()
