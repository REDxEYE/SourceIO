import numpy as np
from typing import Iterable
import bpy

from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class VertexLitGeneric(Source1ShaderBase):
    SHADER: str = 'vertexlitgeneric'

    @property
    def bumpmap(self):
        texture_path = self._vavle_material.get_param('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def basetexture(self):
        texture_path = self._vavle_material.get_param('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def selfillummask(self):
        texture_path = self._vavle_material.get_param('$selfillummask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def phongexponenttexture(self):
        texture_path = self._vavle_material.get_param('$phongexponenttexture', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color2(self):
        color_value, value_type = self._vavle_material.get_vector('$color2', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def color(self):
        color_value, value_type = self._vavle_material.get_vector('$color', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def translucent(self):
        return self._vavle_material.get_int('$translucent', 0) == 1

    @property
    def alphatest(self):
        return self._vavle_material.get_int('$alphatest', 0) == 1

    @property
    def alphatestreference(self):
        return self._vavle_material.get_float('$alphatestreference', 0.5)
         
    @property
    def allowalphatocoverage(self):
        return self._vavle_material.get_int('$allowalphatocoverage', 0) == 1

    @property
    def additive(self):
        return self._vavle_material.get_int('$additive', 0) == 1

    @property
    def phong(self):
        return self._vavle_material.get_int('$phong', 0) == 1

    @property
    def selfillum(self):
        return self._vavle_material.get_int('$selfillum', 0) == 1

    @property
    def basealphaenvmapmask(self):
        return self._vavle_material.get_int('$basealphaenvmapmask', 1) == 1

    @property
    def basemapalphaphongmask(self):
        return self._vavle_material.get_int('$basemapalphaphongmask', 0) == 1

    @property
    def normalmapalphaphongmask(self):
        return self._vavle_material.get_int('$normalmapalphaphongmask', 1) == 1

    @property
    def envmap(self):
        return self._vavle_material.get_string('$envmap', None) is not None

    @property
    def envmapmask(self):
        texture_path = self._vavle_material.get_param('$envmapmask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1, 1, 1, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def envmaptint(self):
        color_value, value_type = self._vavle_material.get_vector('$envmaptint', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]

        return self.ensure_length(color_value, 4, 1.0)

    @property
    def phongfresnelranges(self):
        value, value_type = self._vavle_material.get_vector('$phongfresnelranges', None)
        if value is not None:
            divider = 255 if value_type is int else 1
            value = list(map(lambda a: a / divider, value))
            return self.ensure_length(value, 3, 0.1)
        return None

    @property
    def phongexponent(self):
        return self._vavle_material.get_float('$phongexponent', None)

    @property
    def phongboost(self):
        return self._vavle_material.get_float('$phongboost', 1)

    @property
    def phongalbedotint(self):
        return self._vavle_material.get_int('$phongalbedotint', 1) == 1

    @property
    def phongtint(self):
        color_value, value_type = self._vavle_material.get_vector('$phongtint', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        material_output.location = [250, 0 ]
        parentnode = material_output


        if self.use_bvlg_status:
            self.do_arrange = False
            if self.alphatest:
                alphatest_node = self.create_node(Nodes.ShaderNodeGroup, "$alphatest")
                alphatest_node.node_tree = bpy.data.node_groups.get("$alphatest")
                parentnode = alphatest_node
                material_output.location = [450, 0]
                alphatest_node.location = [250, 0]
                alphatest_node.inputs['$alphatestreference [value]'].default_value = self.alphatestreference
                alphatest_node.inputs['$allowalphatocoverage [boolean]'].default_value = self.allowalphatocoverage
                self.connect_nodes(alphatest_node.outputs['BSDF'], material_output.inputs['Surface'])

            group_node = self.create_node(Nodes.ShaderNodeGroup, self.SHADER)
            group_node.width = group_node.bl_width_max
            group_node.location = [-200, 0]
            group_node.node_tree = bpy.data.node_groups.get("VertexLitGeneric")
            self.connect_nodes(group_node.outputs['BSDF'], parentnode.inputs[0])
            if self.basetexture:
                basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
                basetexture_node.location = [-800, 0]
                basetexture_node.image = self.basetexture
                self.connect_nodes(basetexture_node.outputs['Color'], group_node.inputs['$basetexture [texture]'])
                if self.basealphaenvmapmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])
                if self.basemapalphaphongmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['phongmask [bumpmap texture alpha]'])
                if self.alphatest:
                    self.connect_nodes(basetexture_node.outputs['Alpha'], alphatest_node.inputs['Alpha [basemap texture alpha]'])
            if self.color or self.color2:
                group_node.inputs['$color2 [RGB field]'].default_value = self.color or self.color2

            if self.envmap:
                group_node.inputs['$envmap [boolean]'].default_value = 1
                if self.envmaptint:
                    group_node.inputs['$envmaptint [RGB field]'].default_value = self.envmaptint

            if self.bumpmap:
                bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
                bumpmap_node.location = [-800, -220]
                bumpmap_node.image = self.bumpmap
                self.connect_nodes(bumpmap_node.outputs['Color'], group_node.inputs['$bumpmap [texture]'])

                if self.normalmapalphaphongmask and not self.basemapalphaphongmask:
                    self.connect_nodes(bumpmap_node.outputs['Alpha'],
                                       group_node.inputs['phongmask [bumpmap texture alpha]'])

            if self.phong:
                group_node.inputs['$phong [bool]'].default_value = 1
                if self.phongboost:
                    group_node.inputs['$phongboost [value]'].default_value = self.phongboost
                if self.phongexponent:
                    group_node.inputs['$phongexponent [value]'].default_value = self.phongexponent
                elif self.phongexponenttexture:
                    phongexponent_group_node = self.create_node(Nodes.ShaderNodeGroup, "$phongexponenttexture")
                    phongexponent_group_node.node_tree = bpy.data.node_groups.get("$phongexponenttexture splitter")
                    phongexponenttexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$phongexponenttexture')
                    phongexponenttexture_node.location = [-800, -470]
                    phongexponenttexture_node.image = self.phongexponenttexture
                    self.connect_nodes(phongexponenttexture_node.outputs['Color'], phongexponent_group_node.inputs['$phongexponenttexture [texture]'])
                    self.connect_nodes(phongexponenttexture_node.outputs['Alpha'], phongexponent_group_node.inputs['alpha'])
                    self.connect_nodes(phongexponent_group_node.outputs['$phongexponent [value]'], group_node.inputs['$phongexponent [value]'])
                    self.connect_nodes(phongexponent_group_node.outputs['rimlight mask'], group_node.inputs['rimlight mask'])
                    phongexponent_group_node.location = [-500, -300]

                    if self.phongalbedotint is not None and not self.phongtint:
                        phongexponent_group_node.location = [-550, -300]
                        phongalbedo_node = self.create_node(Nodes.ShaderNodeGroup, "$phongalbedotint")
                        phongalbedo_node.location = [-350, -345]
                        phongalbedo_node.node_tree = bpy.data.node_groups.get("$phongalbedotint")
                        self.connect_nodes(phongexponent_group_node.outputs['phongalbedotint amount'], phongalbedo_node.inputs['phongalbedotint amount'])
                        self.connect_nodes(phongalbedo_node.outputs['$phongtint [RGB field]'], group_node.inputs['$phongtint [RGB field]'])
                        if self.basetexture is not None:
                            self.connect_nodes(basetexture_node.outputs['Color'], phongalbedo_node.inputs['$basetexture [texture]'])
                else:
                    group_node.inputs['$phongexponent [value]'].default_value = 10

                if self.phongtint is not None:
                    group_node.inputs['$phongtint [RGB field]'].default_value = self.phongtint

                if self.phongfresnelranges:
                    group_node.inputs['$phongfresnelranges [value field]'].default_value = self.phongfresnelranges

            if self.selfillum:
                group_node.inputs['$selfillum [bool]'].default_value = 1
                if self.selfillummask:
                    selfillummask_node = self.create_node(Nodes.ShaderNodeTexImage, '$selfillummask')
                    selfillummask_node.location = [-500, -510]
                    selfillummask_node.image = self.selfillummask
                    self.connect_nodes(selfillummask_node.outputs['Color'], group_node.inputs['$selfillummask [texture alpha]'])
                elif self.basetexture is not None:
                    self.connect_nodes(basetexture_node.outputs['Alpha'], group_node.inputs['$selfillummask [texture alpha]'])



        else:
            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
            self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

            basetexture = self.basetexture
            if basetexture:
                basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
                basetexture_node.image = basetexture

                if self.color or self.color2:
                    color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                    color_mix.blend_type = 'MULTIPLY'
                    self.connect_nodes(basetexture_node.outputs['Color'], color_mix.inputs['Color1'])
                    color_mix.inputs['Color2'].default_value = (self.color or self.color2)
                    color_mix.inputs['Fac'].default_value = 1.0
                    self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Base Color'])
                else:
                    self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
                if self.translucent or self.alphatest:
                    self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])

                if self.additive:
                    basetexture_invert_node = self.create_node(Nodes.ShaderNodeInvert)
                    basetexture_additive_mix_node = self.create_node(Nodes.ShaderNodeMixRGB)
                    self.insert_node(basetexture_node.outputs['Color'], basetexture_additive_mix_node.inputs['Color1'],
                                     basetexture_additive_mix_node.outputs['Color'])
                    basetexture_additive_mix_node.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)

                    self.connect_nodes(basetexture_node.outputs['Color'], basetexture_invert_node.inputs['Color'])
                    self.connect_nodes(basetexture_invert_node.outputs['Color'], shader.inputs['Transmission'])
                    self.connect_nodes(basetexture_invert_node.outputs['Color'],
                                       basetexture_additive_mix_node.inputs['Fac'])

            bumpmap = self.bumpmap
            if bumpmap:
                bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
                bumpmap_node.image = bumpmap

                normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

                self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
                self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])

            if self.selfillum:
                selfillummask = self.selfillummask
                basetexture_node = self.get_node('$basetexture')
                if selfillummask is not None:
                    selfillummask_node = self.create_node(Nodes.ShaderNodeTexImage, '$selfillummask')
                    selfillummask_node.image = selfillummask
                    if 'Emission Strength' in shader.inputs:
                        self.connect_nodes(selfillummask_node.outputs['Color'], shader.inputs['Emission Strength'])

                elif 'Emission Strength' in shader.inputs:
                    self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Emission Strength'])
                self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])

            if not self.phong:
                shader.inputs['Specular'].default_value = 0
            elif self.phongboost is not None:
                shader.inputs['Specular'].default_value = self.clamp_value(self.phongboost / 64)
            phongexponenttexture = self.phongexponenttexture
            if self.phongexponent is not None and phongexponenttexture is None:
                shader.inputs['Roughness'].default_value = self.clamp_value(self.phongexponent / 256)
            elif self.phongexponenttexture is not None:
                phongexponenttexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$phongexponenttexture')
                phongexponenttexture_node.image = phongexponenttexture
                phongexponenttexture_split_node = self.create_node(Nodes.ShaderNodeSeparateRGB)
                self.connect_nodes(phongexponenttexture_node.outputs['Color'],
                                   phongexponenttexture_split_node.inputs['Image'])

                phongexponenttexture_r_invert_node = self.create_node(Nodes.ShaderNodeInvert)
                self.connect_nodes(phongexponenttexture_split_node.outputs['R'],
                                   phongexponenttexture_r_invert_node.inputs['Color'])
                self.connect_nodes(phongexponenttexture_split_node.outputs['G'],
                                   shader.inputs['Metallic'])

                self.connect_nodes(phongexponenttexture_r_invert_node.outputs['Color'], shader.inputs['Roughness'])
