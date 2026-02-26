from typing import Any

import bpy
import numpy as np

from SourceIO.blender_bindings.material_loader.shader_base import ShaderBase, Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.source1.vtf import import_texture
from SourceIO.blender_bindings.utils.texture_utils import check_texture_cache
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.blender_bindings.source1.vtf import import_texture, import_texture_tth
from SourceIO.library.source1.vmt import VMT
from SourceIO.library.utils.tiny_path import TinyPath


class Source1ShaderBase(ShaderBase):
    def __init__(self, content_manager: ContentManager, vmt):
        super().__init__()
        self.content_manager = content_manager
        self.load_bvlg_nodes()
        if (dx90 := (vmt.get('>=dx90') or vmt.get('>=DX90'))):
            for key, value in dx90.items():
                vmt[key] = value
        self._vmt: VMT = vmt
        self.textures = {}

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        """Fallback shader: creates a basic Principled BSDF with $basetexture if available."""
        self.logger.warn(f'Using fallback shader for "{self._vmt.shader}", creating basic Principled BSDF')
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        material_output.location = [250, 0]
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, 'Fallback BSDF')
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            basetexture = self.load_texture_or_default(texture_path, (0.5, 0.5, 0.5, 1.0))
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
            basetexture_node.image = basetexture
            self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])

        bumpmap_path = self._vmt.get_string('$bumpmap', None)
        if bumpmap_path is not None:
            bumpmap = self.load_texture_or_default(bumpmap_path, (0.5, 0.5, 1.0, 1.0))
            bumpmap.colorspace_settings.is_data = True
            bumpmap.colorspace_settings.name = 'Non-Color'
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
            bumpmap_node.image = bumpmap
            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)
            self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])

    def load_texture(self, texture_name: str, texture_path: TinyPath):
        image = check_texture_cache(texture_path / texture_name)
        if image is not None:
            return image
        if texture_path.is_absolute(): # Absolute paths shouldn't even be here! This path is invalid
            return None
        texture_file = self.content_manager.find_file("materials" / texture_path / (texture_name + ".vtf"))

        if texture_file is not None:
            return import_texture(texture_path / texture_name, texture_file)

        texture_header_file = self.content_manager.find_file("materials" / texture_path / (texture_name + ".tth"))
        texture_data_file = self.content_manager.find_file("materials" / texture_path / (texture_name + ".ttz"))
        if texture_header_file is not None and texture_data_file is not None:
            return import_texture_tth(texture_path / texture_name, texture_header_file, texture_data_file)
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
