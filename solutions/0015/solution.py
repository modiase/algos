import datetime as dt
import random as rn
import time


def initialise_infinite_stream():
    x = 0
    val = None
    while 1:
        if val:
            x = val
            val = yield x
        else:
            val = yield x


def main(window_size_seconds=10):
    stream = initialise_infinite_stream()
    start = dt.datetime.now()
    now = dt.datetime.now()
    choice = None
    counter = 1
    time.sleep(rn.random() / 100)
    while now - dt.timedelta(seconds=window_size_seconds) < start:
        now = dt.datetime.now()
        val = next(stream)
        stream.send(int(now.timestamp()) - int(start.timestamp()))
        if rn.random() < 1 / float(counter):
            choice = val
        counter += 1
    return choice


if __name__ == '__main__':
    res = main()
    print(res)
