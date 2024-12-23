import time

from .extended_enum import ExtendedEnum
from .file_utils import (Buffer, FileBuffer, MemoryBuffer, Readable,
                         WritableMemoryBuffer)
from .tiny_path import TinyPath
from .path_utilities import path_stem, backwalk_file_resolver, corrected_path
from .math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS, SOURCE2_HAMMER_UNIT_TO_METERS


class Timer:
    def __init__(self):
        """Starts the timer when the instance is created."""
        self.start_time = time.perf_counter()  # Store the start time in seconds
        self.end_time = None

    def stop(self):
        """Stop the timer and return the elapsed time in milliseconds."""
        self.end_time = time.perf_counter()  # Record the end time in seconds
        return (self.end_time - self.start_time) * 1000  # Convert seconds to milliseconds

    def elapsed(self):
        """Return the elapsed time in milliseconds since the timer was started, without stopping the timer."""
        return (time.perf_counter() - self.start_time) * 1000  # Convert seconds to milliseconds

    @staticmethod
    def time_function(func):
        """Decorator to measure the execution time of a function in milliseconds."""

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()  # Start time in seconds
            result = func(*args, **kwargs)  # Execute the function
            elapsed_time = (time.perf_counter() - start_time) * 1000  # Calculate elapsed time in milliseconds
            print(f"Function {func.__name__} executed in {elapsed_time:.2f} ms")
            return result

        return wrapper
