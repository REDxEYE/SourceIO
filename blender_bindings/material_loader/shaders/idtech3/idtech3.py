from typing import Optional

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4, is_blender_4_3
from SourceIO.blender_bindings.utils.texture_utils import check_texture_cache
from SourceIO.library.utils.tiny_path import TinyPath


class IdTech3Shader(ShaderBase):
    SHADER: str = 'idtech3_shader'

    def create_nodes(self, material, material_data: dict):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        if not material_data["textures"]:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        shader_output = shader.outputs['BSDF']
        texture_input = shader.inputs['Base Color']
        textures = material_data["textures"]

        while textures:
            texture = textures.pop(0)
            texture_path = None
            for k, v in texture.items():
                if "map" in k and k != "animMap":
                    texture_path = v
                    break

            if texture_path is not None:
                if texture_path.startswith("$"):
                    continue
                basetexture = self.load_texture(texture_path)
                basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
                basetexture_node.image = basetexture
                basetexture_node.id_data.nodes.active = basetexture_node
                if texture_input is not None:
                    if texture.get("alphaFunc", "") == "GE128":
                        if not is_blender_4_3():
                            self.bpy_material.blend_method = 'HASHED'
                            self.bpy_material.shadow_method = 'HASHED'

                        mix_node = self.create_node(Nodes.ShaderNodeMixShader)
                        self.connect_nodes(basetexture_node.outputs[1], mix_node.inputs[0])
                        transparency_node = self.create_node(Nodes.ShaderNodeBsdfTransparent)
                        self.connect_nodes(transparency_node.outputs[0], mix_node.inputs[1])
                        self.connect_nodes(shader_output, mix_node.inputs[2])
                        shader_output = mix_node.outputs[0]
                    elif texture.get("alphaFunc", "") == "LT128":
                        if not is_blender_4_3():
                            self.bpy_material.blend_method = 'HASHED'
                            self.bpy_material.shadow_method = 'HASHED'

                        mix_node = self.create_node(Nodes.ShaderNodeMixShader)
                        self.connect_nodes(basetexture_node.outputs[1], mix_node.inputs[0])
                        transparency_node = self.create_node(Nodes.ShaderNodeBsdfTransparent)
                        self.connect_nodes(transparency_node.outputs[0], mix_node.inputs[2])
                        self.connect_nodes(shader_output, mix_node.inputs[1])
                        shader_output = mix_node.outputs[0]
                    if False:
                        pass
                    elif False:
                        mix_node = self.create_node(Nodes.ShaderNodeMixRGB)
                        self.connect_nodes(mix_node.outputs[0], texture_input)
                        texture_input = mix_node.inputs[2]

                        self.connect_nodes(basetexture_node.outputs['Color'], mix_node.inputs[2])
                    else:
                        self.connect_nodes(basetexture_node.outputs['Color'], texture_input)
        # if rad_info is not None:
        #     self._emit_surface(basetexture_node, rad_info)
        #     return
        # else:
        self.connect_nodes(shader_output, material_output.inputs['Surface'])

        if is_blender_4():
            shader.inputs['Specular IOR Level'].default_value = 0
        else:
            shader.inputs['Specular'].default_value = 0

    def load_texture(self, texture_name) -> Optional[bpy.types.Image]:
        texture_name = TinyPath(texture_name)
        image = check_texture_cache(texture_name)
        if image is not None:
            return image
        model_texture = bpy.data.images.get(texture_name, None)
        if model_texture is None:
            texture_buffer = self.content_manager.find_file(texture_name + ".png")
            if texture_buffer is None:
                texture_buffer = self.content_manager.find_file(texture_name + ".jpg")
                if texture_buffer is None:
                    texture_buffer = self.content_manager.find_file(texture_name + ".jpeg")

            if texture_buffer:
                model_texture = bpy.data.images.new(
                    texture_name,
                    width=8,
                    height=8,
                    alpha=True
                )
                model_texture.source = "FILE"
                model_texture.file_format = "PNG"
                texture_data = texture_buffer.read(-1)
                model_texture.pack(data=texture_data, data_len=len(texture_data))
                model_texture.reload()
                return model_texture
            return None
        return model_texture
