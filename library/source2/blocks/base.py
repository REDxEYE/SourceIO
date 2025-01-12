import abc

from SourceIO.library.source2.utils.ntro_reader import NTROBuffer


class BaseBlock:
    custom_name: str | None = None

    @classmethod
    @abc.abstractmethod
    def from_buffer(cls, buffer: NTROBuffer) -> 'BaseBlock':
        raise NotImplementedError()
