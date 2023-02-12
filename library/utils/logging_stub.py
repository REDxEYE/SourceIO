import sys
from logging import (DEBUG, Filter, Formatter, LogRecord, StreamHandler,
                     getLogger)
from typing import Dict

from ..utils.singleton import SingletonMeta


class BPYLoggingManager(metaclass=SingletonMeta):
    def __init__(self):
        self.loggers: Dict[str, BPYLogger] = {}
        self.logger = self.get_logger("LOGGING")
        self.logger.debug('Using Stub logger')

    def set_logging_level(self, level):
        [logger.set_logging_level(level) for logger in self.loggers.values()]

    def get_logger(self, name):
        if name in self.loggers:
            return self.loggers[name]
        logger = self.loggers[name] = BPYLogger(name)
        return logger


def _get_caller_function():
    import inspect
    previous_frame = inspect.currentframe().f_back.f_back
    (filename, line_number, function_name, lines, index) = inspect.getframeinfo(previous_frame)
    return function_name


class BPYLogger:
    class Filter(Filter):
        def __init__(self):
            super().__init__()
            self.function = None

        def filter(self, record: LogRecord):
            record.function = self.function or ""
            return True

    def set_logging_level(self, level):
        self._logger.setLevel(level)

    def __init__(self, name):
        self._filter = self.Filter()

        self._logger = getLogger(name)
        self._logger.addFilter(self._filter)
        self._logger.handlers.clear()
        formatter = Formatter('[%(levelname)s]--[%(name)s:%(function)s] %(message)s')
        sh = StreamHandler(sys.stdout)
        self._logger.addHandler(sh)
        for handler in self._logger.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(DEBUG)
        self._logger.setLevel(DEBUG)
        self.name = name

    def print(self, *args, sep=' ', end='\n', ):
        self._filter.function = _get_caller_function()
        self._logger.info(sep.join(map(str, args)))
        sys.stdout.flush()

    def debug(self, message):
        self._filter.function = _get_caller_function()
        self._logger.debug(message)
        sys.stdout.flush()

    def info(self, message):
        self._filter.function = _get_caller_function()
        self._logger.info(message)
        sys.stdout.flush()

    def warn(self, message):
        self._filter.function = _get_caller_function()
        self._logger.warning(message)
        sys.stdout.flush()

    def error(self, message):
        self._filter.function or _get_caller_function()
        self._logger.error(message)
        sys.stdout.flush()

    def exception(self, message, exception=None):
        self._filter.function or _get_caller_function()
        self._logger.exception(message, exc_info=exception)
