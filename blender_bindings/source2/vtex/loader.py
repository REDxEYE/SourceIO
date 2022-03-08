import bpy

from ....library.source2.data_blocks import TEXR
from ....library.source2.data_blocks.texture_block import VTexFormat
from ....library.source2.resource_types import ValveCompiledTexture

from ....logger import SLoggingManager

logger = SLoggingManager().get_logger("Source2::Texture")


class ValveCompiledTextureLoader(ValveCompiledTexture):
    data_block_class = TEXR

    def __init__(self, path_or_file):
        super().__init__(path_or_file)

    def import_texture(self, name, flip: bool):
        logger.info(f'Loading {name} texture')
        if name + '.tga' in bpy.data.images:
            logger.info('Using already loaded texture')
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

        pixel_data = self.load(flip)

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
