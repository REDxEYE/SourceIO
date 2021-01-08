import numpy as np
import bpy

from ..shader_base import ShaderBase, Nodes
from ....bpy_utils import BPYLoggingManager
from ....goldsrc.mdl.structs.texture import StudioTexture, MdlTextureFlag

log_manager = BPYLoggingManager()


class GoldSrcShader(ShaderBase):
    SHADER: str = 'goldsrc_shader'

    def __init__(self, goldsrc_material: StudioTexture):
        self.logger = log_manager.get_logger(f'{self.SHADER}_handler')
        self.bpy_material: bpy.types.Material = None
        self._vavle_material: StudioTexture = goldsrc_material

    def create_nodes(self, material_name: str):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.load_texture()
        basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
        basetexture_node.image = basetexture
        self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
        if self._vavle_material.flags & MdlTextureFlag.CHROME:
            shader.inputs['Specular'].default_value = 0.5
            shader.inputs['Metallic'].default_value = 1
            uvs_node = self.create_node(Nodes.ShaderNodeTexCoord)
            self.connect_nodes(uvs_node.outputs['Reflection'], basetexture_node.inputs['Vector'])
        if self._vavle_material.flags & MdlTextureFlag.FULL_BRIGHT:
            shader.inputs['Emission Strength'].default_value = 1
            self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])
        else:
            shader.inputs['Specular'].default_value = 0

    def load_texture(self, **kwargs):
        model_texture_info = self._vavle_material
        model_texture = bpy.data.images.get(model_texture_info.name, None)
        if model_texture is None:
            model_texture = bpy.data.images.new(
                model_texture_info.name,
                width=model_texture_info.width,
                height=model_texture_info.height,
                alpha=False
            )

            if bpy.app.version > (2, 83, 0):
                model_texture.pixels.foreach_set(model_texture_info.data.flatten().tolist())
            else:
                model_texture.pixels[:] = model_texture_info.data.flatten().tolist()

            model_texture.pack()
        return model_texture
