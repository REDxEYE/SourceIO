from typing import Optional

import bpy

from .....library.goldsrc.mdl_v10.structs.texture import MdlTextureFlag
from ...shader_base import Nodes
from ..goldsrc_shader_base import GoldSrcShaderBase


class GoldSrcShaderMode5(GoldSrcShaderBase):
    SHADER: str = 'goldsrc_shader_mode5'

    def create_nodes(self, material_name: str, rad_info=None, model_name: Optional[str] = None):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        basetexture = self.load_texture(material_name, model_name)
        basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
        basetexture_node.image = basetexture
        basetexture_node.id_data.nodes.active = basetexture_node

        if rad_info is not None:
            self._emit_surface(basetexture_node, rad_info)
            return

        vertex_color_alpha = self.create_node(Nodes.ShaderNodeVertexColor)
        vertex_color_alpha.layer_name = "RENDER_AMOUNT"

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeEmission, self.SHADER)

        self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Color'])

        math_mul = self.create_node(Nodes.ShaderNodeMath)
        math_mul.operation = 'MULTIPLY'
        self.connect_nodes(basetexture_node.outputs['Alpha'], math_mul.inputs[0])
        self.connect_nodes(vertex_color_alpha.outputs['Color'], math_mul.inputs[1])

        if self._valve_material.flags & MdlTextureFlag.CHROME:
            uvs_node = self.create_node(Nodes.ShaderNodeTexCoord)
            self.connect_nodes(uvs_node.outputs['Reflection'], basetexture_node.inputs['Vector'])

        shader_mix_node = self.create_node(Nodes.ShaderNodeMixShader)

        self.connect_nodes(math_mul.outputs[0], shader_mix_node.inputs['Fac'])
        self.connect_nodes(shader.outputs['Emission'], shader_mix_node.inputs[2])
        transparent_node = self.create_node(Nodes.ShaderNodeBsdfTransparent)
        self.connect_nodes(transparent_node.outputs['BSDF'], shader_mix_node.inputs[1])

        self.connect_nodes(shader_mix_node.outputs[0], material_output.inputs['Surface'])
