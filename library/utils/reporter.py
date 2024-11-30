from dataclasses import dataclass

from SourceIO.library.utils.singleton import SingletonMeta


class SourceIOException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(slots=True, frozen=True)
class SourceIOWarning:
    msg: str

    def __str__(self):
        return f"Warning: {self.msg}"


class Reporter(metaclass=SingletonMeta):
    def __init__(self):
        self._errors: list[SourceIOException] = []
        self._warnings: list[SourceIOWarning] = []

    def clear(self) -> 'Reporter':
        self._errors.clear()
        self._warnings.clear()
        return self

    def error(self, exception: SourceIOException):
        self._errors.append(exception)

    def warning(self, exception: SourceIOWarning):
        self._warnings.append(exception)

    def warnings(self) -> list[SourceIOWarning]:
        return self._warnings

    def errors(self) -> list[SourceIOException]:
        return self._errors

    @classmethod
    def new(cls):
        return cls().clear()

    @classmethod
    def current(cls):
        return cls()