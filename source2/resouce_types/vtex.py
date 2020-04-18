from pathlib import Path

import bpy


from SourceIO.source2.source2 import ValveFile


class Vtex:

    def __init__(self, vtex_path):
        self.valve_file = ValveFile(vtex_path)
        self.valve_file.read_block_info()

    def load(self, flip: bool):
        name = Path(self.valve_file.filename).stem
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        data_block.read_image(flip)
        rgb, alpha = data_block.get_rgb_and_alpha()

        image = bpy.data.images.new(
            name + '_RGB.tga',
            width=data_block.width,
            height=data_block.height)
        image.pixels = rgb
        image.pack()
        if alpha is not None:
            image = bpy.data.images.new(
                name + '_A.tga',
                width=data_block.width,
                height=data_block.height)
            image.pixels = alpha
            image.pack()
