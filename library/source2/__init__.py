from pathlib import Path

from ..utils import Buffer, FileBuffer


def load_compiled_resource_from_path(path: Path):
    return load_compiled_resource(FileBuffer(path), path)


def load_compiled_resource(buffer: Buffer, path: Path):
    if not buffer:
        return None
    file_type = path.suffix
    if file_type == '.vtex_c':
        return CompiledTextureResource.from_buffer(buffer, path)
    if file_type == '.vmat_c':
        return CompiledMaterialResource.from_buffer(buffer, path)
    if file_type == '.vwrld_c':
        return CompiledWorldResource.from_buffer(buffer, path)
    if file_type == '.vmdl_c':
        return CompiledModelResource.from_buffer(buffer, path)
    if file_type == '.vphy_c':
        return CompiledPhysicsResource.from_buffer(buffer, path)
    if file_type == '.vmorf_c':
        return CompiledMorphResource.from_buffer(buffer, path)
    return CompiledResource.from_buffer(buffer, path)


from .resource_types import CompiledMorphResource, CompiledPhysicsResource
# Recursive import bypass
from .resource_types.compiled_material_resource import \
    CompiledMaterialResource
from .resource_types.compiled_model_resource import CompiledModelResource
from .resource_types.compiled_texture_resource import CompiledTextureResource
from .resource_types.compiled_world_resource import CompiledWorldResource
from .resource_types.resource import CompiledResource
