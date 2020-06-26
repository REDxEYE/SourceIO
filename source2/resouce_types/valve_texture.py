from pathlib import Path

import bpy
import numpy as np

from ..source2 import ValveFile


class ValveTexture:

    def __init__(self, vtex_path, valve_file=None):
        if valve_file:
            self.valve_file = valve_file
        else:
            self.valve_file = ValveFile(vtex_path)
            self.valve_file.read_block_info()

    def load(self, flip: bool, split_alpha: bool):
        name = Path(self.valve_file.filename).stem
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        data_block.read_image(flip)
        if split_alpha:
            rgb, alpha = data_block.get_rgb_and_alpha()
        else:
            rgb = np.divide(list(data_block.image_data),255)
            alpha = None

        image = bpy.data.images.new(
            name + '_RGB.tga',
            width=data_block.width,
            height=data_block.height)
        image.filepath_raw = str(self.valve_file.filepath.with_name(image.name).with_suffix('.tga'))
        image.pixels = rgb
        image.save()
        # image.pack()
        if (
                alpha is not None
                and (np.sum(alpha[0::4]) + np.sum(alpha[1::4]) + np.sum(alpha[2::4]))
                > 10
        ):
            image = bpy.data.images.new(
                name + '_A.tga',
                width=data_block.width,
                height=data_block.height)
            image.filepath_raw = str(self.valve_file.filepath.with_name(image.name).with_suffix('.tga'))
            image.pixels = alpha
            image.save()
            # image.pack()

        return (
            name + '_RGB.tga',
            name + '_A.tga' if not (alpha is None or split_alpha) else None,
        )
