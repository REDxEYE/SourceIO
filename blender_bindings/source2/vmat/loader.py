from pathlib import Path

from ....library.source2.data_blocks import DATA
from ....library.source2.resource_types import ValveCompiledMaterial


class ValveCompiledMaterialLoader(ValveCompiledMaterial):
    def load(self):
        from ...material_loader.material_loader import Source2MaterialLoader
        data_block: DATA = self.get_data_block(block_name='DATA')[0]
        source_material = Source2MaterialLoader(data_block.data, Path(data_block.data['m_materialName']).stem,
                                                self.available_resources)
        source_material.create_material()
