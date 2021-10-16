from .detail import DetailSupportMixin
from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase
import bpy


class Lightmapped4WayBlend(DetailSupportMixin):
    SHADER = 'lightmapped_4wayblend'

    @property
    def basetexture(self):
        texture_path = self._vavle_material.get_param('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.0, 0.3, 1.0))
        return None

    @property
    def basetexture2(self):
        texture_path = self._vavle_material.get_param('$basetexture2', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.3, 0.0, 1.0))
        return None

    @property
    def basetexture3(self):
        texture_path = self._vavle_material.get_param('$basetexture3', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.3, 0.0, 1.0))
        return None

    @property
    def basetexture4(self):
        texture_path = self._vavle_material.get_param('$basetexture4', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.3, 0.0, 1.0))
        return None

    @property
    def bumpmap(self):
        texture_path = self._vavle_material.get_param('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def bumpmap2(self):
        texture_path = self._vavle_material.get_param('$bumpmap2', None) or self._vavle_material.get_param(
            '$basenormalmap2', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def bumpmap3(self):
        texture_path = self._vavle_material.get_param('$basenormalmap3', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def bumpmap4(self):
        texture_path = self._vavle_material.get_param('$basenormalmap4', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    def lumstart(self, N = 1):
        return self._vavle_material.get_float(f"$texture{N}_lumstart", 0)

    def lumend(self, N = 1):
        return self._vavle_material.get_float(f"$texture{N}_lumend", 1)

    def lumblendfactor(self, N = 1):
        return self._vavle_material.get_float(f"$lumblendfactor{N}", 0)

    def blendstart(self, N = 1):
        return self._vavle_material.get_float(f"$texture{N}_blendstart", 0)

    def blendend(self, N = 1):
        return self._vavle_material.get_float(f"$texture{N}_blendend", 1)

    def getblends(self):
        keyvalues = {}
        keyvalues['$texture1_lumstart'] = self.lumstart(1)
        keyvalues['$texture1_lumend'] = self.lumend(1)
        for i in range(2, 5):
            keyvalues[f'$texture{i}_lumstart' ] = self.lumstart(i)
            keyvalues[f'$texture{i}_lumend'] = self.lumend(i)
            keyvalues[f'$lumblendfactor{i}'] = self.lumblendfactor(i)
            keyvalues[f'$texture{i}_blendstart'] = self.blendstart(i)
            keyvalues[f'$texture{i}_blendend'] = self.blendend(i)
        return keyvalues

    def putblends(self, blends : dict, group : Nodes.ShaderNodeGroup):
        for key, value in blends.items():
            group.inputs[key].default_value = value


    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        if (self.use_bvlg):
            shader = self.create_node_group("LightMappedGeneric", name="LightMappedGeneric")
        else:
            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture1 = self.basetexture
        basetexture2 = self.basetexture2
        basetexture3 = self.basetexture3
        basetexture4 = self.basetexture4
        normal_texture1 = self.bumpmap
        normal_texture2 = self.bumpmap2
        normal_texture3 = self.bumpmap3
        normal_texture4 = self.bumpmap4
        bases = [None, None, None, None]
        normals = [None, None, None, None]
        if basetexture1:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture1')
            basetexture_node.image = basetexture1
            bases[0] = basetexture_node
        if basetexture2:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture2')
            basetexture_node.image = basetexture2
            bases[1] = basetexture_node
        if basetexture3:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture3')
            basetexture_node.image = basetexture3
            bases[2] = basetexture_node
        if basetexture4:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture4')
            basetexture_node.image = basetexture4
            bases[3] = basetexture_node

        if normal_texture1:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap1')
            bumpmap_node.image = normal_texture1
            normals[0] = bumpmap_node
        if normal_texture2:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap2')
            bumpmap_node.image = normal_texture2
            normals[1] = bumpmap_node
        if normal_texture3:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap3')
            bumpmap_node.image = normal_texture3
            normals[2] = bumpmap_node
        if normal_texture4:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap4')
            bumpmap_node.image = normal_texture4
            normals[3] = bumpmap_node

        vertex_color = self.create_node(Nodes.ShaderNodeVertexColor)
        vertex_color.layer_name = 'multiblend'

        if (self.use_bvlg):
            blend = self.create_node_group("4wayBlend", name="4wayBlend")

            self.connect_nodes(vertex_color.outputs['Color'], blend.inputs['Vertex Color'])

            if bases[0]:
                self.connect_nodes(bases[0].outputs['Color'], blend.inputs['$basetexture'])
            if normals[0]:
                self.connect_nodes(normals[0].outputs['Color'], blend.inputs['$bumpmap'])
            for i in range(1, 4):
                if bases[i]:
                    self.connect_nodes(bases[i].outputs['Color'], blend.inputs[f'$basetexture{i+1}'])
                if normals[i]:
                    self.connect_nodes(normals[i].outputs['Color'], blend.inputs[f'$bumpmap{i+1}'])

            self.connect_nodes(blend.outputs['$bumpmap [texture]'], shader.inputs['$bumpmap [texture]'])
            if self.detail:
                self.handle_detail(shader.inputs['$basetexture [texture]'], blend.outputs['$basetexture [texture]'])
            else:
                self.connect_nodes(blend.outputs['$basetexture [texture]'], shader.inputs['$basetexture [texture]'])
            self.connect_nodes(blend.outputs['$bumpmap [texture]'], shader.inputs['$bumpmap [texture]'])
            self.putblends(self.getblends(), blend)


        else:
            color_mixer = self.create_node(Nodes.ShaderNodeGroup)
            color_mixer.node_tree = self.get_or_create_4way_mix_group()
            normal_mixer = self.create_node(Nodes.ShaderNodeGroup)
            normal_mixer.node_tree = self.get_or_create_4way_mix_group()

            self.connect_nodes(vertex_color.outputs['Color'], color_mixer.inputs['MultiBlend Mask'])
            self.connect_nodes(vertex_color.outputs['Alpha'], color_mixer.inputs['MultiBlend Alpha'])
            if bases[0]:
                self.connect_nodes(bases[0].outputs['Color'], color_mixer.inputs['Color1'])
            if bases[1]:
                self.connect_nodes(bases[1].outputs['Color'], color_mixer.inputs['Color2'])
            if bases[2]:
                self.connect_nodes(bases[2].outputs['Color'], color_mixer.inputs['Color3'])
            if bases[3]:
                self.connect_nodes(bases[3].outputs['Color'], color_mixer.inputs['Color4'])

            self.connect_nodes(vertex_color.outputs['Color'], normal_mixer.inputs['MultiBlend Mask'])
            self.connect_nodes(vertex_color.outputs['Alpha'], normal_mixer.inputs['MultiBlend Alpha'])
            if normals[0]:
                self.connect_nodes(normals[0].outputs['Color'], normal_mixer.inputs['Color1'])
            if normals[1]:
                self.connect_nodes(normals[1].outputs['Color'], normal_mixer.inputs['Color2'])
            if normals[2]:
                self.connect_nodes(normals[2].outputs['Color'], normal_mixer.inputs['Color3'])
            if normals[3]:
                self.connect_nodes(normals[3].outputs['Color'], normal_mixer.inputs['Color4'])

            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

            self.connect_nodes(normal_mixer.outputs['Color'], normalmap_node.inputs['Color'])

            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])
            self.connect_nodes(color_mixer.outputs['Color'], shader.inputs['Base Color'])


    def get_or_create_4way_mix_group(self):
        mixer_group = bpy.data.node_groups.get("4way_mixer", None)
        if mixer_group is None:
            mixer_group = bpy.data.node_groups.new("4way_mixer", "ShaderNodeTree")
            nodes = mixer_group.nodes
            links = mixer_group.links
            link_nodes = links.new

            group_inputs = mixer_group.nodes.new('NodeGroupInput')
            group_inputs.location = (-350, 0)
            mixer_group.inputs.new('NodeSocketColor', 'MultiBlend Mask')
            mixer_group.inputs.new('NodeSocketColor', 'MultiBlend Alpha')
            mixer_group.inputs.new('NodeSocketColor', 'Color1')
            mixer_group.inputs.new('NodeSocketColor', 'Color2')
            mixer_group.inputs.new('NodeSocketColor', 'Color3')
            mixer_group.inputs.new('NodeSocketColor', 'Color4')

            # create group outputs
            group_outputs = mixer_group.nodes.new('NodeGroupOutput')
            group_outputs.location = (300, 0)
            mixer_group.outputs.new('NodeSocketColor', 'Color')

            split = nodes.new(Nodes.ShaderNodeSeparateRGB)
            link_nodes(group_inputs.outputs['MultiBlend Mask'], split.inputs['Image'])

            color_mix_0 = nodes.new(Nodes.ShaderNodeMixRGB)
            color_mix_0.blend_type = 'MIX'

            link_nodes(group_inputs.outputs['Color1'], color_mix_0.inputs['Color1'])
            link_nodes(group_inputs.outputs['Color2'], color_mix_0.inputs['Color2'])
            link_nodes(split.outputs['R'], color_mix_0.inputs['Fac'])
            color_mix_1 = nodes.new(Nodes.ShaderNodeMixRGB)
            color_mix_1.blend_type = 'MIX'
            link_nodes(color_mix_0.outputs['Color'], color_mix_1.inputs['Color1'])
            link_nodes(group_inputs.outputs['Color3'], color_mix_1.inputs['Color2'])
            link_nodes(split.outputs['G'], color_mix_1.inputs['Fac'])
            color_mix_2 = nodes.new(Nodes.ShaderNodeMixRGB)
            color_mix_2.blend_type = 'MIX'
            link_nodes(color_mix_1.outputs['Color'], color_mix_2.inputs['Color1'])
            link_nodes(group_inputs.outputs['Color4'], color_mix_2.inputs['Color2'])
            link_nodes(split.outputs['B'], color_mix_2.inputs['Fac'])

            link_nodes(color_mix_2.outputs['Color'], group_outputs.inputs['Color'])
        return mixer_group
