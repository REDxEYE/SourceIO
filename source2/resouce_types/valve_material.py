from pathlib import Path

from ..source2 import ValveFile
from .valve_texture import ValveTexture
import bpy


class ValveMaterial:

    def __init__(self, vtex_path):
        self.valve_file = ValveFile(vtex_path)
        self.valve_file.read_block_info()
        self.valve_file.check_external_resources()

    def load(self, flip_textures,split_alpha, override=True):
        textures = {}
        params = {}
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        if data_block:
            bl_material = bpy.data.materials.get(self.valve_file.filepath.stem, False) or bpy.data.materials.new(
                self.valve_file.filepath.stem)
            for tex in data_block.data['m_textureParams']:
                texture = self.valve_file.get_child_resource(tex['m_pValue'])
                if texture is not None:
                    print(f"Loading {texture.filepath.stem} texture")
                    tex_file = ValveTexture('', valve_file=texture)
                    rgb_tex_name, alpha_tex_name = tex_file.load(flip_textures,split_alpha)
                    textures[tex['m_name']] = tex_file, rgb_tex_name, alpha_tex_name
                else:
                    textures[tex['m_name']] = None
                    print(f"missing {tex['m_pValue']} texture")
            for vec_param in data_block.data['m_vectorParams']:
                vec_name = vec_param['m_name']
                vec_value = vec_param['m_value']
                params[vec_name] = vec_value

            if override:
                bl_material.use_nodes = True
                nodes = bl_material.node_tree.nodes

                def link(output_socket, input_socket):
                    bl_material.node_tree.links.new(output_socket, input_socket)

                output_node = nodes.get('Material Output', None) or nodes.new('ShaderNodeOutputMaterial')
                shader_node = nodes.get('Principled BSDF', None) or nodes.new('ShaderNodeBsdfPrincipled')

                # bpy.ops.node.add_node(type="ShaderNodeMixRGB", use_transform=True)

                if textures.get("g_tColor", False):
                    tex = nodes.new('ShaderNodeTexImage')
                    tex.image = bpy.data.images.get(textures.get('g_tColor')[1], "NONE")
                    if params.get('g_vColorTint', None):
                        mix = nodes.new('ShaderNodeMixRGB')
                        mix.blend_type = 'MULTIPLY'
                        mix.inputs['Fac'].default_value = 1.0
                        mix.inputs['Color2'].default_value = params.get('g_vColorTint').as_list

                        link(tex.outputs["Color"], mix.inputs['Color1'])
                        link(mix.outputs["Color"], shader_node.inputs['Base Color'])

                    else:
                        link(tex.outputs["Color"], shader_node.inputs['Base Color'])

                if textures.get("g_tNormal", False):
                    tex = nodes.new('ShaderNodeTexImage')
                    tex.image = bpy.data.images.get(textures.get('g_tNormal')[1], "NONE")

                    tex.image.colorspace_settings.is_data = True
                    tex.image.colorspace_settings.name = 'Non-Color'

                    sep = nodes.new('ShaderNodeSeparateRGB')
                    combine = nodes.new('ShaderNodeCombineRGB')
                    normal = nodes.new('ShaderNodeNormalMap')

                    link(tex.outputs["Color"], sep.inputs['Image'])

                    link(sep.outputs["R"], combine.inputs['R'])
                    link(sep.outputs["G"], combine.inputs['G'])
                    combine.inputs['B'].default_value = 1.0

                    link(combine.outputs['Image'], normal.inputs['Color'])
                    link(normal.outputs['Normal'], shader_node.inputs['Normal'])

                    link(sep.outputs["B"], shader_node.inputs['Roughness'])
