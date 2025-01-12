from .kv3_block import KVBlock


class PhysBlock(KVBlock):
    @staticmethod
    def _struct_name():
        return "VPhysXAggregateData_t"