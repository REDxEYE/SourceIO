from typing import Type

from .base import BaseBlock


def get_block_class(name) -> Type['BaseBlock']:
    from .agrp_block import AgrpBlock
    from .aseq_block import AseqBlock
    from .dummy import DummyBlock
    from .kv3_block import KVBlock
    from .morph_block import MorphBlock
    from .phys_block import PhysBlock
    from .resource_edit_info import ResourceEditInfo, ResourceEditInfo2
    from .resource_external_reference_list import ResourceExternalReferenceList
    from .resource_introspection_manifest import ResourceIntrospectionManifest
    from .vertex_index_buffer import VertexIndexBuffer

    if name == "NTRO":
        return ResourceIntrospectionManifest
    elif name == "REDI":
        return ResourceEditInfo
    elif name == "RED2":
        return ResourceEditInfo2
    elif name == "RERL":
        return ResourceExternalReferenceList
    elif name == 'ASEQ':
        return AseqBlock
    elif name == 'MDAT':
        return KVBlock
    elif name == 'PHYS':
        return PhysBlock
    elif name == 'AGRP':
        return AgrpBlock
    elif name == 'DATA':
        return KVBlock
    elif name == 'CTRL':
        return KVBlock
    elif name == 'INSG':
        return KVBlock
    elif name == 'MRPH':
        return MorphBlock
    elif name == 'MBUF':
        return VertexIndexBuffer
    elif name == 'VBIB':
        return VertexIndexBuffer
    else:
        return BaseBlock
