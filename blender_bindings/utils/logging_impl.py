import sys
from logging import DEBUG, Filter, Formatter, Logger, LogRecord, StreamHandler
from typing import Dict

import bpy

from ...library.utils.singleton import SingletonMeta


def get_log_file(filename):
    file = bpy.data.texts.get(filename, None)
    if file is None:
        return bpy.data.texts.new(filename), True
    return file, False


def _get_caller_function():
    import inspect
    previous_frame = inspect.currentframe().f_back.f_back
    (filename, line_number,
     function_name, lines, index) = inspect.getframeinfo(previous_frame)
    return function_name


class BPYLoggingManager(metaclass=SingletonMeta):
    def __init__(self):
        self.loggers: Dict[str, BPYLogger] = {}
        self.logger = self.get_logger("LOGGING")
        self.logger.debug('Using BPY logger')

    def get_logger(self, name):
        if name in self.loggers:
            return self.loggers[name]
        logger = self.loggers[name] = BPYLogger(name)
        return logger

    def set_logging_level(self, level):
        [logger.set_logging_level(level) for logger in self.loggers.values()]


class BPYLogger:
    class Filter(Filter):
        def __init__(self):
            super().__init__()
            self.function = None

        def filter(self, record: LogRecord):
            record.function = self.function or ""
            return True

    class Logger(Logger):
        pass

    def set_logging_level(self, level):
        self._logger.setLevel(level)

    def __init__(self, name):
        self.name = name
        self._bpy_file = None
        self._filter = self.Filter()

        self._logger = self.Logger(name)
        self._logger.addFilter(self._filter)
        for h in self._logger.handlers:
            self._logger.removeHandler(h)
        self._logger.handlers.clear()
        self._formatter = Formatter('[%(levelname)s]--[%(name)s:%(function)s]: %(message)s')

        self._logger.setLevel(DEBUG)

        self._bpy_logger = None

    def _add_bpy_file_logger(self):
        if bpy.data.__class__.__name__ == "_RestrictData":
            return
        self._bpy_file, new = get_log_file(self.name)
        if self._bpy_logger is not None:
            self._bpy_logger.stream = self._bpy_file
        if new:
            self._logger.handlers.clear()
            if self._bpy_logger is None:
                self._bpy_logger = StreamHandler(self._bpy_file)
                self._bpy_logger.setFormatter(self._formatter)
            self._logger.addHandler(self._bpy_logger)

            sh = StreamHandler(sys.stdout)
            sh.setFormatter(self._formatter)
            self._logger.addHandler(sh)

    def print(self, *args, sep=' ', end='\n', ):
        self._add_bpy_file_logger()
        self._filter.function = _get_caller_function()
        self._logger.info(sep.join(map(str, args)))

    def debug(self, message):
        self._add_bpy_file_logger()
        self._filter.function = _get_caller_function()
        self._logger.debug(message)

    def info(self, message):
        self._add_bpy_file_logger()
        self._filter.function = _get_caller_function()
        self._logger.info(message)

    def warn(self, message):
        self._add_bpy_file_logger()
        self._filter.function = _get_caller_function()
        self._logger.warning(message)

    def error(self, message):
        self._add_bpy_file_logger()
        self._filter.function or _get_caller_function()
        self._logger.error(message)

    def exception(self, message, exception=None):
        self._add_bpy_file_logger()
        self._filter.function or _get_caller_function()
        self._logger.exception(message, exc_info=exception)
