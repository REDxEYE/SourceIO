import bpy
import numpy as np

from SourceIO.blender_bindings.material_loader.shader_base import ShaderBase
from SourceIO.blender_bindings.source1.vtf import import_texture
from SourceIO.blender_bindings.utils.texture_utils import check_texture_cache
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.blender_bindings.source1.vtf import import_texture, import_texture_tth
from SourceIO.library.source1.vmt import VMT
from SourceIO.library.utils.tiny_path import TinyPath


class Source1ShaderBase(ShaderBase):
    def __init__(self, content_manager: ContentManager, vmt: VMT):
        super().__init__()
        self.content_manager = content_manager
        self.load_bvlg_nodes()
        if (dx90 := (vmt.get('>=dx90') or vmt.get('>=DX90'))):
            # unravel it, because we want dx90 properties
            for key, value in dx90.items():
                vmt[key] = value
        self._vmt: VMT = vmt
        self.textures = {}

    def _color_property(self, name: str, default=None, length: int = 4, filler: float = 1.0):
        """Read a VMT vector/color property.

        Integer-syntax values (``{255 128 0}``) are normalized to 0..1, single
        floats are broadcast to greyscale, and the result is padded/truncated to
        ``length``. Returns ``None`` when the property is absent and no default
        was supplied.
        """
        value, value_type = self._vmt.get_vector(name, default)
        if value is None:
            return None
        divider = 255 if value_type is int else 1
        value = [component / divider for component in value]
        if len(value) == 1:
            value = [value[0]] * 3
        return self.ensure_length(value, length, filler)

    def _texture_property(self, name: str, default_color: tuple[float, float, float, float],
                          *, is_data: bool = False, normal_map: bool = False, ssbump: bool = False):
        """Load a VMT texture property, or ``None`` if it is not set.

        ``normal_map`` flips the green channel, ``ssbump`` converts from
        self-shadowed bump space; both imply non-color data.
        """
        texture_path = self._vmt.get_string(name, None)
        if texture_path is None:
            return None
        image = self.load_texture_or_default(texture_path, default_color)
        if ssbump:
            image = self.convert_ssbump(image)
        if normal_map:
            image = self.convert_normalmap(image)
        if is_data or normal_map or ssbump:
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
        return image

    def _bool_property(self, name: str, default: int = 0) -> bool:
        return self._vmt.get_int(name, default) == 1

    def load_texture(self, texture_name: TinyPath, texture_path: TinyPath | None = None):
        if texture_path is None or texture_path == texture_name:
            full_path = texture_name
        else:
            full_path = texture_path / texture_name
        image = check_texture_cache(full_path)
        if image is not None:
            return image
        if texture_path is not None and texture_path.is_absolute():  # Absolute paths shouldn't even be here! This path is invalid
            return None
        texture_file = self.content_manager.find_file("materials" / (full_path + ".vtf"))

        if texture_file is not None:
            return import_texture(full_path, texture_file)

        texture_header_file = self.content_manager.find_file("materials"  / (full_path + ".tth"))
        texture_data_file = self.content_manager.find_file("materials" /  (full_path + ".ttz"))
        if texture_header_file is not None and texture_data_file is not None:
            return import_texture_tth(full_path, texture_header_file, texture_data_file)
        return None

    @staticmethod
    def convert_ssbump(image: bpy.types.Image):
        if image.get('ssbump_converted', None):
            return image

        # from https://github.com/rob5300/ssbumpToNormal-Win/blob/main/SSBumpToNormal.cs
        bumpBasisTranspose = np.array([
            [0.81649661064147949, -0.40824833512306213, -0.40824833512306213],
            [0.0, 0.70710676908493042, -0.7071068286895752],
            [0.57735025882720947, 0.57735025882720947, 0.57735025882720947]
        ])

        buffer = np.zeros((image.size[0] * image.size[1], 4), np.float32)
        image.pixels.foreach_get(buffer.ravel())

        dots = np.dot(buffer[:, :3], bumpBasisTranspose.T)
        dots *= 0.5
        dots += 0.5

        buffer[:, :3] = dots

        image.pixels.foreach_set(buffer.ravel())
        image.pack()
        del buffer
        image['ssbump_converted'] = True
        return image

    @staticmethod
    def convert_normalmap(image: bpy.types.Image):
        if image.get('normalmap_converted', None):
            return image

        buffer = np.zeros((image.size[0], image.size[1], 4), np.float32)
        image.pixels.foreach_get(buffer.ravel())

        buffer[:, :, 1] = np.subtract(1, buffer[:, :, 1])
        image.pixels.foreach_set(buffer.ravel())
        image.pack()
        del buffer
        image['normalmap_converted'] = True
        return image
