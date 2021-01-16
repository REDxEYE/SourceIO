from pathlib import Path

# noinspection PyUnresolvedReferences
import bpy
import numpy as np

from ..blocks import TEXR
from ..source2 import ValveFile


class ValveTexture:

    def __init__(self, vtex_path, valve_file=None):
        if valve_file:
            self.valve_file = valve_file
        else:
            self.valve_file = ValveFile(vtex_path)
            self.valve_file.read_block_info()

    def load(self, flip: bool):
        name = Path(self.valve_file.filename).stem
        print(f'Loading {name} texture')
        data_block: TEXR = self.valve_file.get_data_block(block_name='DATA')[0]
        data_block.read_image(flip)
        pixel_data = np.divide(np.frombuffer(data_block.image_data, np.uint8), 255, dtype=np.float32)
        if name + '.tga' in bpy.data.images:
            return bpy.data.images[f'{name}.tga']
        image = bpy.data.images.new(
            name + '.tga',
            width=data_block.width,
            height=data_block.height,
            alpha=True
        )
        image.alpha_mode = 'CHANNEL_PACKED'
        image.file_format = 'TARGA'
        image.filepath_raw = str(self.valve_file.filepath.with_name(image.name).with_suffix('.tga'))
        if pixel_data.shape[0] > 0:
            if bpy.app.version > (2, 83, 0):
                image.pixels.foreach_set(pixel_data.tolist())
            else:
                image.pixels[:] = pixel_data.tolist()
        image.save()
        del pixel_data
        return image
