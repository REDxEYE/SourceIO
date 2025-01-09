from .kv3_block import KVBlock


class AgrpBlock(KVBlock):
    @staticmethod
    def _get_struct(ntro):
        return ntro.struct_by_name('AnimationGroupResourceData_t')
