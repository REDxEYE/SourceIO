from typing import Union, IO, BinaryIO
from io import TextIOBase, BufferedIOBase, BytesIO, StringIO
from pathlib import Path

from ...utilities.keyvalues import KVParser
from ...bpy_utilities.logging import BPYLoggingManager


class VMTParser:
    def __init__(self, file_or_string: Union[IO[str], IO[bytes], str]):
        self.logger = BPYLoggingManager().get_logger('vmt_parser')
        if isinstance(file_or_string, str):
            self._buffer = file_or_string
        elif isinstance(file_or_string, Path):
            with file_or_string.open('r') as f:
                self._buffer = f.read()
        elif isinstance(file_or_string, (TextIOBase, StringIO)):
            self._buffer = file_or_string.read()
        elif isinstance(file_or_string, (BufferedIOBase, BytesIO)):
            self._buffer = file_or_string.read().decode('latin', errors='replace')
        else:
            raise ValueError(f'Unknown input value type {type(file_or_string)}')

        # if not self._buffer.strip().endswith('}'):
        #     self.logger.debug(f'VMT has has missing closing "}}"')
        #     self._buffer += '\n}'

        self._parser = KVParser('<input>', self._buffer, single_value=True)
        self.header, self._raw_data = self._parser.parse()

    def get_vector(self, name, default=(0, 0, 0)):
        raw_value = self._raw_data.get(name, None)
        if raw_value is None:
            return default, None

        if raw_value[0] == '{':
            converter = int
            pass
        elif raw_value[0] == '[':
            converter = float
        else:
            return [float(raw_value)], float
            # raise ValueError(f'Not a vector value: {raw_value}')

        values = raw_value[1:-1].split()
        return tuple(map(converter, values)), converter

    def get_string(self, name, default='invalid'):
        raw_value = self._raw_data.get(name, default)
        return str(raw_value) if raw_value is not None else None

    def get_int(self, name, default=0):
        raw_value = self._raw_data.get(name, None)
        if raw_value is None:
            return default
        if raw_value and '.' in raw_value:
            return int(float(raw_value))
        return int(raw_value)

    def get_float(self, name, default=0.0):
        raw_value = self._raw_data.get(name, None)
        if raw_value is None:
            return default
        return float(raw_value)

    def get_subblock(self, name, default=None):
        if default is None:
            default = dict()
        raw_value = self._raw_data.get(name, None)
        return raw_value or default

    def get_param(self, name, default=None):
        raw_value = self._raw_data.get(name, None)
        return str(raw_value) if raw_value else default

    def apply_patch(self, data):
        self._raw_data.update(data)
        return self

    def get_raw_data(self):
        return self._raw_data
