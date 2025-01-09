from .kv3_block import KVBlock


class PhysBlock(KVBlock):
    @staticmethod
    def _get_struct(ntro):
        return ntro.struct_by_name("VPhysXAggregateData_t")