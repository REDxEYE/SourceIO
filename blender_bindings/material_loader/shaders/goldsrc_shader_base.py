from typing import Optional, Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ShaderBase, ExtraMaterialParameters
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3
from SourceIO.library.models.mdl.v10.structs.texture import StudioTexture


class GoldSrcShaderBase(ShaderBase):
    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        self.bpy_material = material
        if self.bpy_material is None:
            self.logger.error('Failed to get or create material')
            return 'UNKNOWN'

        if self.bpy_material.get('source_loaded'):
            return 'LOADED'
        self.logger.info(f'Creating material {repr(material.name)}')

        self.bpy_material.use_nodes = True
        self.clean_nodes()
        if not is_blender_4_3():
            self.bpy_material.blend_method = 'OPAQUE'
            self.bpy_material.shadow_method = 'OPAQUE'
        self.bpy_material.use_screen_refraction = False
        self.bpy_material.refraction_depth = 0.2
        self.bpy_material['source_loaded'] = True

    SHADER: str = 'goldsrc_shader'

    def __init__(self, goldsrc_material: StudioTexture):
        super().__init__()
        self._valve_material: StudioTexture = goldsrc_material

    def _emit_surface(self, basetexture, rad_info):
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader_emit = self.create_node(Nodes.ShaderNodeEmission)
        shader_emit.inputs['Strength'].default_value = rad_info[3]

        color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
        color_mix.blend_type = 'MULTIPLY'
        color_mix.inputs['Fac'].default_value = 1.0

        self.connect_nodes(color_mix.outputs['Color'], shader_emit.inputs['Color'])
        self.connect_nodes(basetexture.outputs['Color'], color_mix.inputs['Color1'])
        color_mix.inputs['Color2'].default_value = (*rad_info[:3], 1.0)

        self.connect_nodes(shader_emit.outputs['Emission'], material_output.inputs['Surface'])

    def load_texture(self, material_name, model_name: Optional[str] = None, **kwargs):
        model_texture_info = self._valve_material
        if model_name:
            texture_name = f"{model_name}_{material_name}"
        else:
            texture_name = material_name
        model_texture = bpy.data.images.get(texture_name, None)
        if model_texture is None:
            model_texture = bpy.data.images.new(
                texture_name,
                width=model_texture_info.width,
                height=model_texture_info.height,
                alpha=True
            )

            model_texture.pixels.foreach_set(model_texture_info.data.ravel())

            model_texture.pack()
        return model_texture
