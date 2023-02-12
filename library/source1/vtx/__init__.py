from pathlib import Path
from typing import BinaryIO, Union

from ...utils import Buffer, FileBuffer
from .v6.vtx import Vtx as Vtx6
from .v7.vtx import Vtx as Vtx7


def open_vtx(filepath_or_object: Union[Path, str, Buffer]):
    buffer: Buffer
    if isinstance(filepath_or_object, (Path, str)):
        buffer = FileBuffer(filepath_or_object)
    elif isinstance(filepath_or_object, Buffer):
        buffer = filepath_or_object
    else:
        raise NotImplementedError(f"Unsupported type of input: {filepath_or_object}")
    version = buffer.read_int32()
    buffer.seek(0)
    if version == 6:
        return Vtx6.from_buffer(filepath_or_object)
    elif version == 7:
        return Vtx7.from_buffer(filepath_or_object)
