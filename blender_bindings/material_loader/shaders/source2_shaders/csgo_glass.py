from pprint import pformat

import bpy

from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class CSGOGlass(Source2ShaderBase):
    SHADER: str = 'csgo_glass.vfx'

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("csgo_glass.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        material_data = self._material_resource
        data, = material_data.get_data_block(block_name='DATA')
        self.logger.info(pformat(dict(data)))

        if self._have_texture("g_tGlassTintColor"):
            color_texture = self._get_texture("g_tGlassTintColor", (1, 1, 1, 1))
            self.connect_nodes(color_texture.outputs[0], shader.inputs["TintColor"])

        if self._have_texture("g_tGlassDust"):
            color_texture = self._get_texture("g_tGlassDust", (1, 1, 1, 1))
            self.connect_nodes(color_texture.outputs[0], shader.inputs["TextureDust"])
            self.connect_nodes(color_texture.outputs[1], shader.inputs["TextureRoughness"])

        if self._have_texture("g_tNormal"):
            normal_texture = self._get_texture("g_tNormal", (0.5, 0.5, 1, 1), True, True)
            self.connect_nodes(normal_texture.outputs[0], shader.inputs["TextureNormal"])

        self.bpy_material.use_screen_refraction = True
        old_engine = bpy.context.scene.render.engine
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.use_ssr = True
        bpy.context.scene.eevee.use_ssr_refraction = True
        bpy.context.scene.render.engine = old_engine
