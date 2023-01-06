from pathlib import Path

from SourceIO.blender_bindings.material_loader.material_loader import \
    Source2MaterialLoader
from SourceIO.library.source2 import CompiledMaterialResource


def load_material(material_resource: CompiledMaterialResource, material_path: Path):
    source_material = Source2MaterialLoader(material_resource, material_path.stem)
    source_material.create_material()
