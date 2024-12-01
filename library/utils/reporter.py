from dataclasses import dataclass

from SourceIO.library.utils.singleton import SingletonMeta
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Reporter')


class SourceIOException(Exception):
    pass


class SourceIOWrappedException(Exception):
    def __init__(self, msg: str, ex: Exception):
        super().__init__(msg)
        self.ex = ex
        self.msg = msg

    def __str__(self):
        return f"{self.msg}: {self.ex!s}"


@dataclass(slots=True, frozen=True)
class SourceIOWarning:
    msg: str

    def __str__(self):
        return f"Warning: {self.msg}"


class Reporter(metaclass=SingletonMeta):
    def __init__(self):
        self._errors: list[Exception] = []
        self._warnings: list[SourceIOWarning] = []

    def clear(self) -> 'Reporter':
        self._errors.clear()
        self._warnings.clear()
        return self

    def error(self, exception: Exception):
        if isinstance(exception, SourceIOWrappedException):
            logger.exception(exception.msg, exception.ex)
        else:
            logger.exception(str(exception), exception)
        self._errors.append(exception)

    def warning(self, exception: SourceIOWarning):
        logger.warn(str(exception))
        self._warnings.append(exception)

    def warnings(self) -> list[SourceIOWarning]:
        return self._warnings

    def errors(self) -> list[Exception]:
        return self._errors

    @classmethod
    def new(cls):
        return cls().clear()

    @classmethod
    def current(cls):
        return cls()
