from pathlib import Path
from typing import Any, Dict

import zlib
import bpy
import numpy as np

from ...utils.texture_utils import check_texture_cache, create_and_cache_texture
from ....library.shared.content_providers.content_manager import ContentManager
from ....library.source1.vmt import VMT
from ....library.utils import MemoryBuffer
from ...source1.vtf import import_texture
from ..shader_base import ShaderBase


def combine_troika_texture_to_vtf(header: MemoryBuffer, body: MemoryBuffer) -> MemoryBuffer:
    header_id = header.read_fourcc()
    assert header_id == "TTH"
    ver_hi, ver_lo = header.read_fmt("2B")
    assert (ver_hi, ver_lo) == (1, 0)
    vtf_mips_nr = header.read_uint8()
    unrec_ar_flag = header.read_uint8()
    vtf_chunk_size = header.read_uint32()
    mips_flags_data = header.read_fmt(f"{vtf_mips_nr}d")
    vtf_file_size = header.read_uint32()
    ttz_file_size = header.read_uint32()
    vtf_chunk = header.read_fmt(f"{vtf_chunk_size}B")
    assert header.remaining() == 0
    decompressed_body = zlib.decompress(body.data, 0)
    return MemoryBuffer(bytes(vtf_chunk) + bytes(decompressed_body))

    
class Source1ShaderBase(ShaderBase):
    def __init__(self, vmt):
        super().__init__()
        self.load_bvlg_nodes()
        self._vmt: VMT = vmt
        self.textures = {}

    def load_texture(self, texture_name: str, texture_path: Path):
        image = check_texture_cache(texture_path / texture_name)
        if image is not None:
            return image
        if bpy.data.images.get(texture_name, False):
            self.logger.debug(f'Using existing texture {texture_name}')
            return bpy.data.images.get(texture_name)

        content_manager = ContentManager()
        texture_file = content_manager.find_texture(texture_path / texture_name)
        if texture_file is None:
            texture_header = content_manager.find_tth(texture_path / texture_name)
            texture_body = content_manager.find_ttz(texture_path / texture_name)
            if texture_header is None or texture_body is None:
                self.logger.warning('Found VtMB .tth but not .ttz or vice versa, cannot import texture')
                return None
            texture_file = combine_troika_texture_to_vtf(texture_header, texture_body)
        if texture_file is not None:
            return import_texture(texture_path / texture_name, texture_file)
        return None

    @staticmethod
    def convert_ssbump(image: bpy.types.Image):
        if image.get('ssbump_converted', None):
            return image
        buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
        image.pixels.foreach_get(buffer)
        buffer[0::4] *= 0.5
        buffer[0::4] += 0.33
        buffer[1::4] *= 0.5
        buffer[1::4] += 0.33
        buffer[2::4] *= 0.2
        buffer[2::4] += 0.8
        image.pixels.foreach_set(buffer.ravel())
        image.pack()
        del buffer
        image['ssbump_converted'] = True
        return image

    @staticmethod
    def convert_normalmap(image: bpy.types.Image):
        if image.get('normalmap_converted', None):
            return image

        buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
        image.pixels.foreach_get(buffer)

        buffer[1::4] = np.subtract(1, buffer[1::4])
        image.pixels.foreach_set(buffer.ravel())
        image.pack()
        del buffer
        image['normalmap_converted'] = True
        return image
