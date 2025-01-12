from .kv3_block import KVBlock


class AgrpBlock(KVBlock):
    @staticmethod
    def _struct_name():
        return 'AnimationGroupResourceData_t'
