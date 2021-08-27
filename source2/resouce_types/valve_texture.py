# noinspection PyUnresolvedReferences
import bpy
import numpy as np

from ..blocks import TEXR
from ..blocks.texture_block import VTexFormat
from . import ValveCompiledResource


class ValveCompiledTexture(ValveCompiledResource):
    data_block_class = TEXR

    def __init__(self, path_or_file):
        super().__init__(path_or_file)

    def load(self, name, flip: bool):
        print(f'Loading {name} texture')
        if name + '.tga' in bpy.data.images:
            print('Using already loaded texture')
            return bpy.data.images[f'{name}.tga']
        data_block: TEXR = self.get_data_block(block_name='DATA')[0]
        data_block.read_image(flip)
        if data_block.image_data is None:
            image = bpy.data.images.new(
                name + '.tga',
                width=data_block.width,
                height=data_block.height,
                alpha=True
            )
            return image
        if data_block.format == VTexFormat.RGBA16161616F:
            pixel_data = data_block.image_data
        else:
            pixel_data = np.divide(np.frombuffer(data_block.image_data, np.uint8), 255, dtype=np.float32)

        image = bpy.data.images.new(
            name + '.tga',
            width=data_block.width,
            height=data_block.height,
            alpha=True
        )

        image.alpha_mode = 'CHANNEL_PACKED'
        if data_block.format == VTexFormat.RGBA16161616F:
            image.use_generated_float = True
            image.file_format = 'HDR'
        else:
            image.file_format = 'TARGA'
        # image.filepath_raw = f'{name}.tga'

        if pixel_data.shape[0] > 0:
            if bpy.app.version > (2, 83, 0):
                image.pixels.foreach_set(pixel_data.tolist())
            else:
                image.pixels[:] = pixel_data.tolist()
        image.pack()
        del pixel_data
        return image
