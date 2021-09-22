import numpy as np

from .....logger import SLoggingManager
from ...data_blocks import TEXR
from ...data_blocks.texture_block import VTexFormat
from ...resource_types import ValveCompiledResource


logger = SLoggingManager().get_logger("Source2::Texture")


class ValveCompiledTexture(ValveCompiledResource):
    data_block_class = TEXR

    def __init__(self, path_or_file):
        super().__init__(path_or_file)

    def load(self, flip: bool):
        data_block: TEXR = self.get_data_block(block_name='DATA')[0]
        data_block.read_image(flip)
        if data_block.format == VTexFormat.RGBA16161616F:
            pixel_data = data_block.image_data
        else:
            pixel_data = np.divide(np.frombuffer(data_block.image_data, np.uint8), 255, dtype=np.float32)

        return pixel_data
