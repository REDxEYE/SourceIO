from pathlib import Path

from ..utils.bpy_utils import find_or_create_material
from ...blender_bindings.material_loader.material_loader import \
    Source2MaterialLoader
from ...library.source2 import CompiledMaterialResource


def load_material(material_resource: CompiledMaterialResource, material_path: Path, tinted: bool = False):
    source_material = Source2MaterialLoader(material_resource, material_path.stem, tinted)
    material = find_or_create_material(material_path.stem, material_path.as_posix())
    source_material.create_material(material)
