from pathlib import Path

from ...blender_bindings.material_loader.material_loader import \
    Source2MaterialLoader
from ...library.source2 import CompiledMaterialResource


def load_material(material_resource: CompiledMaterialResource, material_path: Path, tinted: bool = False):
    source_material = Source2MaterialLoader(material_resource, material_path.stem, tinted)
    source_material.create_material()
