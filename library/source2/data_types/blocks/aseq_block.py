from .kv3_block import KVBlock


class AseqBlock(KVBlock):
    @staticmethod
    def _get_struct(ntro):
        return ntro.struct_by_name('SequenceGroupResourceData_t')
