from typing import Optional

import bpy

from .....library.goldsrc.mdl_v10.structs.texture import MdlTextureFlag
from ...shader_base import Nodes
from ..goldsrc_shader_base import GoldSrcShaderBase


class GoldSrcShader(GoldSrcShaderBase):
    SHADER: str = 'goldsrc_shader'

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
        else:
            material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
            self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
            self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
            if self._valve_material.flags & MdlTextureFlag.CHROME:
                shader.inputs['Specular'].default_value = 0.5
                shader.inputs['Metallic'].default_value = 1
                uvs_node = self.create_node(Nodes.ShaderNodeTexCoord)
                self.connect_nodes(uvs_node.outputs['Reflection'], basetexture_node.inputs['Vector'])
            if self._valve_material.flags & MdlTextureFlag.FULL_BRIGHT:
                shader.inputs['Emission Strength'].default_value = 1
                self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])
            else:
                shader.inputs['Specular'].default_value = 0
