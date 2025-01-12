from .kv3_block import KVBlock


class AseqBlock(KVBlock):
    @staticmethod
    def _struct_name():
        return 'SequenceGroupResourceData_t'
