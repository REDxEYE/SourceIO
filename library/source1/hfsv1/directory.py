from dataclasses import dataclass

from .file import CompressionMethod, File
from .xor_key import xor_decode

#
# class Directory:
#
#     def __init__(self):
#         self.version = 0
#         self.extract_version = 0
#         self.bit_flag = 0
#         self.compression_method = 0
#         self.mod_file_time = 0
#         self.mod_file_date = 0
#         self.checksum = 0
#         self.compressed_size = 0
#         self.decompressed_size = 0
#         self.partition_number = 0
#         self.internal_attributes = 0
#         self.attributes = 0
#         self.data_offset = 0
#         self.filename = ''
#         self.extra = ''
#         self.comment = ''
#         self.file = File()
#
#     def read(self, reader: Buffer):
#         assert reader.read_int32() == 0x02014648
#         self.version = reader.read_int16()
#         self.extract_version = reader.read_int16()
#         self.bit_flag = reader.read_int16()
#         self.compression_method = CompressionMethod(reader.read_int16())
#         self.mod_file_time = reader.read_int16()
#         self.mod_file_date = reader.read_int16()
#         self.checksum = reader.read_int32()
#         self.compressed_size = reader.read_int32()
#         self.decompressed_size = reader.read_int32()
#         length = reader.read_int16()
#         extra_length = reader.read_int16()
#         comment_length = reader.read_int16()
#         self.partition_number = reader.read_int16()
#         self.internal_attributes = reader.read_int16()
#         self.attributes = reader.read_int32()
#         self.data_offset = reader.read_uint32()
#         pos = reader.tell()
#         self.filename = xor_decode(reader.read(length), key_offset=pos).decode('utf-8')
#         pos = reader.tell()
#         self.extra = xor_decode(reader.read(extra_length), key_offset=pos).decode('utf-8')
#         pos = reader.tell()
#         self.comment = xor_decode(reader.read(comment_length), key_offset=pos).decode('utf-8')
#         with reader.save_current_pos():
#             reader.seek(self.data_offset)
#             self.file.read(reader)
#
#     def __repr__(self):
#         return f'<HFSFile "{self.filename}">'
