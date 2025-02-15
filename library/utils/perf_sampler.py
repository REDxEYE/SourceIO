import os
import time
from collections import defaultdict
from functools import wraps
from typing import Callable
from array import array

# Read the environment variable and convert it to a boolean.
# For example, treating "1", "true", "yes", or "on" (case-insensitive) as True.
CAPTURING_ENABLED = os.environ.get("SOURCEIO_PERF_TRACE", "0").lower() in ("1", "true", "yes", "on")
# CAPTURING_ENABLED = False # manual overrides
# CAPTURING_ENABLED = True  # manual overrides
FUNCTION_TIMES: dict[Callable, array] = defaultdict(lambda: array("f"))


def timed(function: Callable):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = function(*args, **kwargs)
        duration = time.perf_counter() - start
        FUNCTION_TIMES[function.__qualname__].append(duration)
        return res

    return wrapper if CAPTURING_ENABLED else function


def print_stats():
    if CAPTURING_ENABLED:
        info = sorted(FUNCTION_TIMES.items(), key=lambda v: sum(v[1]), reverse=True)
        longest_name = len(sorted(info, key=lambda v: len(v[0]))[-1][0])

        print("Execution time info:")
        if not FUNCTION_TIMES:
            print("No timing data collected.")
            return
        print("{0:<{max_w}}\t{1:<16}\t{2:<16}   \t{3:<16}".format("Name",
                                                                  "count",
                                                                  "total, sec",
                                                                  "average, sec",
                                                                  max_w=longest_name + 3))
        for name, times in info:
            print("{0:<{max_w}}\t{1:<16}\t{2:<16.4f}\t{3:<16.5f}".format(name,
                                                                         len(times),
                                                                         sum(times),
                                                                         sum(times) / len(times),
                                                                         max_w=longest_name + 3))


if __name__ == '__main__':
    @timed
    def test():
        time.sleep(2)


    test()
    print_stats()
