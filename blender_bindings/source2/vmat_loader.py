from SourceIO.blender_bindings.utils.bpy_utils import get_or_create_material
from SourceIO.blender_bindings.material_loader.material_loader import Source2MaterialLoader
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source2 import CompiledMaterialResource
from SourceIO.library.utils.tiny_path import TinyPath


def load_material(content_manager:ContentManager,material_resource: CompiledMaterialResource, material_path: TinyPath, tinted: bool = False):
    source_material = Source2MaterialLoader(content_manager, material_resource, material_path.stem, tinted)
    material = get_or_create_material(material_path.stem, material_path.as_posix())
    source_material.create_material(material)
