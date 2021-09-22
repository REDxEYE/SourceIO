from pathlib import Path

from ...resource_types import ValveCompiledResource
from ....shared.content_providers.content_manager import ContentManager


class ValveCompiledModel(ValveCompiledResource):

    def __init__(self, path_or_file):
        super().__init__(path_or_file)
        if isinstance(path_or_file, (Path, str)):
            ContentManager().scan_for_content(path_or_file)
        data_block = self.get_data_block(block_name='DATA')
        assert len(data_block) == 1
        data_block = data_block[0]
        self.name = data_block.data['m_name']
        self.materials = []
