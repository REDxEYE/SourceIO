from typing import List

from ...utilities.byte_io_mdl import ByteIO
from .dummy import DataBlock


class RERL(DataBlock):
    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.resources = []  # type: List[RERLResource]

    def __repr__(self):
        return f'<External resources list count:{len(self.resources)}>'

    def print_resources(self):
        for res in self.resources:
            print('\t', res)

    def read(self):
        reader = self.reader
        entry = reader.tell()
        resource_entry_offset = reader.read_int32()
        resource_count = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + resource_entry_offset)
            for n in range(resource_count):
                resource = RERLResource()
                resource.read(reader)
                self.resources.append(resource)


class RERLResource:

    def __init__(self):
        self.resource_hash = 0
        self.r_id = 0
        self.resource_name = ''

    def __repr__(self):
        return '<External resource "{}">'.format(self.resource_name)

    def read(self, reader: ByteIO):
        self.resource_hash = reader.read_uint32()
        self.r_id = reader.read_uint32()
        entry = reader.tell()
        resource_name_offset = reader.read_int64()
        if resource_name_offset:
            self.resource_name = reader.read_from_offset(entry + resource_name_offset, reader.read_ascii_string)
