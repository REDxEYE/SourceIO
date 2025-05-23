from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4, is_blender_4_3


class Refract(Source1ShaderBase):
    SHADER: str = 'refract'

    @property
    def bumpmap(self):
        texture_path = self._vmt.get_string('$normalmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def color2(self):
        color_value, value_type = self._vmt.get_vector('$color2', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        elif len(color_value) > 3:
            color_value = color_value[:3]
        return color_value

    @property
    def bluramount(self):
        value = self._vmt.get_float('$bluramount', 0)
        return value

    @property
    def color(self):
        color_value, value_type = self._vmt.get_vector('$color', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        elif len(color_value) > 3:
            color_value = color_value[:3]
        return color_value

    @property
    def refracttint(self):
        color_value, value_type = self._vmt.get_vector('$refracttint', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return color_value

    def create_nodes(self, material:bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        if not is_blender_4_3():
            self.bpy_material.blend_method = 'OPAQUE'
            self.bpy_material.shadow_method = 'NONE'
        self.bpy_material.use_screen_refraction = True
        self.bpy_material.use_backface_culling = True
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)

        shader_mix = self.create_node(Nodes.ShaderNodeMixShader)
        light_path = self.create_node(Nodes.ShaderNodeLightPath)
        transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(light_path.outputs['Is Camera Ray'], shader_mix.inputs[0])
        self.connect_nodes(transparent.outputs['BSDF'], shader_mix.inputs[1])
        self.connect_nodes(shader.outputs['BSDF'], shader_mix.inputs[2])
        self.connect_nodes(shader_mix.outputs[0], material_output.inputs['Surface'])

        basetexture = self.basetexture
        if basetexture:
            self.create_and_connect_texture_node(basetexture, shader.inputs['Base Color'], name='$basetexture')
        bumpmap = self.bumpmap
        if bumpmap:
            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)
            self.create_and_connect_texture_node(bumpmap, normalmap_node.inputs['Color'], name='$bumpmap')

            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])
            if is_blender_4():
                shader.inputs['Transmission Weight'].default_value = 1.0
            else:
                shader.inputs['Transmission'].default_value = 1.0
            shader.inputs['Roughness'].default_value = self.bluramount
