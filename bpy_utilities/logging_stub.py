from typing import Dict

from ..utilities.singleton import SingletonMeta


class BPYLoggingManager(metaclass=SingletonMeta):
    def __init__(self):
        self.loggers: Dict[str, BPYLogger] = {}

    def get_logger(self, name):
        logger = self.loggers[name] = BPYLogger(name)
        return logger


class BPYLogger:
    def __init__(self, name):
        self.name = name

    def log(self, log_level, message, module=None):
        print(f'[{log_level:8}]--[{f"{module}:" if module is not None else ""}{self.name}] {message}')

    def debug(self, message, module=None):
        self.log('DEBUG', message, module)

    def info(self, message, module=None):
        self.log('INFO', message, module)

    def warn(self, message, module=None):
        self.log('WARN', message, module)

    def error(self, message, module=None):
        self.log('ERROR', message, module)
