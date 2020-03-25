from typing import List

from .header_block import InfoBlock
from ..source2 import ValveFile
from ...byte_io_mdl import ByteIO
from .dummy import DataBlock


class RERL(DataBlock):
    def __init__(self, valve_file: ValveFile, info_block):
        super().__init__(valve_file, info_block)
        self.resource_entry_offset = 0
        self.resource_count = 0
        self.resources = []  # type: List[RERLResource]

    def __repr__(self):
        return '<External resources list count:{}>'.format(self.resource_count)

    def print_resources(self):
        for res in self.resources:
            print('\t', res)

    def read(self):
        reader = self.reader
        entry = reader.tell()
        self.resource_entry_offset = reader.read_int32()
        self.resource_count = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + self.resource_entry_offset)
            for n in range(self.resource_count):
                resource = RERLResource()
                resource.read(reader)
                self.resources.append(resource)
        self.empty = False


class RERLResource:

    def __init__(self):
        self.r_id = 0
        self.resource_name_offset = 0
        self.resource_name = ''

    def __repr__(self):
        return '<External resource "{}">'.format(self.resource_name)

    def read(self, reader: ByteIO):
        self.r_id = reader.read_int64()
        entry = reader.tell()
        self.resource_name_offset = reader.read_int64()
        if self.resource_name_offset:
            self.resource_name = reader.read_from_offset(entry + self.resource_name_offset, reader.read_ascii_string)
