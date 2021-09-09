from pathlib import Path

from ..blocks import DATA
from . import ValveCompiledResource
from ...bpy_utilities.material_loader.material_loader import Source2MaterialLoader


class ValveCompiledMaterial(ValveCompiledResource):

    def __init__(self, path_or_file):
        super().__init__(path_or_file)

    def load(self):
        data_block: DATA = self.get_data_block(block_name='DATA')[0]
        source_material = Source2MaterialLoader(data_block.data, Path(data_block.data['m_materialName']).stem,
                                                self.available_resources)
        source_material.create_material()
