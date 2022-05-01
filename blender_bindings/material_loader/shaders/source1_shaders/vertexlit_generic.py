import json

import bpy

from .detail import DetailSupportMixin
from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class VertexLitGeneric(DetailSupportMixin, Source1ShaderBase):
    SHADER: str = 'vertexlitgeneric'

    @property
    def bumpmap(self):
        texture_path = self._vmt.get_string('$bumpmap', None)
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
    def decaltexture(self):
        texture_path = self._vmt.get_string('$decaltexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def basetexturetransform(self):
        return self._vmt.get_transform_matrix('$basetexturetransform',
                                              {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0),
                                               'translate': (0, 0, 0)})

    @property
    def selfillummask(self):
        texture_path = self._vmt.get_string('$selfillummask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def phongexponenttexture(self):
        texture_path = self._vmt.get_string('$phongexponenttexture', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color2(self):
        color_value, value_type = self._vmt.get_vector('$color2', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def color(self):
        color_value, value_type = self._vmt.get_vector('$color', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def translucent(self):
        return self._vmt.get_int('$translucent', 0) == 1

    @property
    def alphatest(self):
        return self._vmt.get_int('$alphatest', 0) == 1

    @property
    def alphatestreference(self):
        return self._vmt.get_float('$alphatestreference', 0.5)

    @property
    def allowalphatocoverage(self):
        return self._vmt.get_int('$allowalphatocoverage', 0) == 1

    @property
    def additive(self):
        return self._vmt.get_int('$additive', 0) == 1

    @property
    def phong(self):
        return self._vmt.get_int('$phong', 0) == 1

    @property
    def selfillum(self):
        return self._vmt.get_int('$selfillum', 0) == 1

    @property
    def basealphaenvmapmask(self):
        return self._vmt.get_int('$basealphaenvmapmask', 1) == 1

    @property
    def basemapalphaphongmask(self):
        return self._vmt.get_int('$basemapalphaphongmask', 0) == 1

    @property
    def normalmapalphaphongmask(self):
        return self._vmt.get_int('$normalmapalphaphongmask', 1) == 1

    @property
    def normalmapalphaenvmapmask(self):
        return self._vmt.get_int('$normalmapalphaenvmapmask', 0) == 1

    @property
    def blendtintbybasealpha(self):
        return self._vmt.get_int('$blendtintbybasealpha', 0) == 1

    @property
    def envmap(self):
        return self._vmt.get_string('$envmap', None) is not None

    @property
    def envmapmask(self):
        texture_path = self._vmt.get_string('$envmapmask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1, 1, 1, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def envmaptint(self):
        color_value, value_type = self._vmt.get_vector('$envmaptint', [1.0, 1.0, 1.0])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]

        return self.ensure_length(color_value, 4, 1.0)

    @property
    def phongfresnelranges(self):
        value, value_type = self._vmt.get_vector('$phongfresnelranges', None)
        if value is not None:
            divider = 255 if value_type is int else 1
            value = list(map(lambda a: a / divider, value))
            return self.ensure_length(value, 3, 0.1)
        return None

    @property
    def phongexponent(self):
        value = self._vmt.get_float('$phongexponent', None)
        return value

    @property
    def phongboost(self):
        value = self._vmt.get_float('$phongboost', 1)
        return value

    @property
    def phongalbedotint(self):
        return self._vmt.get_int('$phongalbedotint', 1) == 1

    @property
    def phongtint(self):
        color_value, value_type = self._vmt.get_vector('$phongtint', None)
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

        if 'proxies' in self._vmt:
            proxies = self._vmt.get('proxies')
            for proxy_name, proxy_data in proxies.items():
                if proxy_name == 'selectfirstifnonzero':
                    result_var = proxy_data.get('resultvar')
                    src1_var = proxy_data.get('srcvar1')
                    src2_var = proxy_data.get('srcvar2')
                    src1_value, src1_type = self._vmt.get_vector(src1_var, [0])
                    if all([val > 0 for val in src1_value]):
                        self._vmt[result_var] = self._vmt[src1_var]
                    else:
                        self._vmt[result_var] = self._vmt[src2_var]

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        material_output.location = [250, 0]
        parentnode = material_output

        if self.alphatest or self.translucent:
            if self.translucent:
                self.bpy_material.blend_method = 'BLEND'
            else:
                self.bpy_material.blend_method = 'HASHED'
            self.bpy_material.shadow_method = 'HASHED'
        uv = None
        if self.use_bvlg_status:
            self.do_arrange = False
            if self.alphatest or self.translucent:
                alphatest_node = self.create_node_group("$alphatest", [250, 0])
                parentnode = alphatest_node
                material_output.location = [450, 0]
                alphatest_node.inputs['$alphatestreference [value]'].default_value = self.alphatestreference
                alphatest_node.inputs['$allowalphatocoverage [boolean]'].default_value = self.allowalphatocoverage
                self.connect_nodes(alphatest_node.outputs['BSDF'], material_output.inputs['Surface'])

            group_node = self.create_node_group("VertexLitGeneric", [-200, 0])
            self.connect_nodes(group_node.outputs['BSDF'], parentnode.inputs[0])
            if self.basetexture:
                basetexture_node = self.create_and_connect_texture_node(self.basetexture,
                                                                        group_node.inputs['$basetexture [texture]'],
                                                                        name='$basetexture')
                basetexture_node.location = [-800, 0]
                if self.basetexturetransform:
                    uv, self.uv_map = self.handle_transform(self.basetexturetransform, basetexture_node.inputs[0])
                albedo = basetexture_node.outputs['Color']
                if self.basealphaenvmapmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])
                if self.basemapalphaphongmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['phongmask [bumpmap texture alpha]'])
                if self.alphatest:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       alphatest_node.inputs['Alpha [basemap texture alpha]'])

                if self.detail:
                    albedo, detail = self.handle_detail(group_node.inputs['$basetexture [texture]'], albedo, UV=uv)

            if self.color or self.color2:
                group_node.inputs['$color2 [RGB field]'].default_value = self.color or self.color2

            if self.envmap:
                group_node.inputs['$envmap [boolean]'].default_value = 1
                if self.envmaptint:
                    group_node.inputs['$envmaptint [RGB field]'].default_value = self.envmaptint

            if self.bumpmap:
                bumpmap_node = self.create_and_connect_texture_node(self.bumpmap,
                                                                    group_node.inputs['$bumpmap [texture]'],
                                                                    name='$bumpmap')
                bumpmap_node.location = [-800, -220]
                if self.normalmapalphaenvmapmask:
                    self.connect_nodes(bumpmap_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])
                elif self.normalmapalphaphongmask and not self.basemapalphaphongmask:
                    self.connect_nodes(bumpmap_node.outputs['Alpha'],
                                       group_node.inputs['phongmask [bumpmap texture alpha]'])

            if self.phong:
                group_node.inputs['$phong [bool]'].default_value = 1
                if self.phongboost:
                    group_node.inputs['$phongboost [value]'].default_value = self.phongboost
                if self.phongexponent:
                    group_node.inputs['$phongexponent [value]'].default_value = self.phongexponent
                elif self.phongexponenttexture:
                    phongexponent_group_node = self.create_node_group('$phongexponenttexture splitter', [-500, -300])
                    self.connect_nodes(phongexponent_group_node.outputs['$phongexponent [value]'],
                                       group_node.inputs['$phongexponent [value]'])
                    self.connect_nodes(phongexponent_group_node.outputs['rimlight mask'],
                                       group_node.inputs['rimlight mask'])
                    phongexponenttexture_node = self.create_and_connect_texture_node(self.phongexponenttexture,
                                                                                     phongexponent_group_node.inputs[
                                                                                         '$phongexponenttexture [texture]'],
                                                                                     phongexponent_group_node.inputs[
                                                                                         'alpha'],
                                                                                     name='$phongexponenttexture',
                                                                                     UV=uv)
                    phongexponenttexture_node.location = [-800, -470]

                    if self.phongalbedotint is not None and not self.phongtint:
                        phongexponent_group_node.location = [-550, -300]
                        phongalbedo_node = self.create_node_group("$phongalbedotint", [-350, -345])
                        self.connect_nodes(phongexponent_group_node.outputs['phongalbedotint amount'],
                                           phongalbedo_node.inputs['phongalbedotint amount'])
                        self.connect_nodes(phongalbedo_node.outputs['$phongtint [RGB field]'],
                                           group_node.inputs['$phongtint [RGB field]'])
                        if self.basetexture is not None:
                            self.connect_nodes(basetexture_node.outputs['Color'],
                                               phongalbedo_node.inputs['$basetexture [texture]'])
                else:
                    group_node.inputs['$phongexponent [value]'].default_value = 10

                if self.phongtint is not None:
                    group_node.inputs['$phongtint [RGB field]'].default_value = self.phongtint

                if self.phongfresnelranges:
                    group_node.inputs['$phongfresnelranges [value field]'].default_value = self.phongfresnelranges

            if self.selfillum:
                group_node.inputs['$selfillum [bool]'].default_value = 1
                if self.selfillummask:
                    selfillummask_node = self.create_and_connect_texture_node(self.selfillummask, group_node.inputs[
                        '$selfillummask [texture alpha]'], UV=uv)
                    selfillummask_node.location = [-500, -510]
                elif self.basetexture is not None:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['$selfillummask [texture alpha]'])
        else:
            f = bpy.data.texts.new(material_name)
            f.write(json.dumps(self._vmt.data.to_dict(), indent=1))

            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
            shader_output = shader.outputs['BSDF']

            base_texture = self.basetexture
            if base_texture is not None:
                basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
                basetexture_node.image = base_texture
                basetexture_node.id_data.nodes.active = basetexture_node

            else:
                basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
                basetexture_node.image = self.get_missing_texture(f'missing_$basetexture_{material_name}')
                basetexture_node.id_data.nodes.active = basetexture_node

            base_texture_color_output = basetexture_node.outputs['Color']
            base_texture_alpha_output = basetexture_node.outputs['Alpha']

            if self.selfillum and not self.selfillummask:
                math_node = self.create_node(Nodes.ShaderNodeMath)
                math_node.operation = 'MULTIPLY'
                self.connect_nodes(base_texture_alpha_output, math_node.inputs[0])
                math_node.inputs[1].default_value = 5
                self.connect_nodes(math_node.outputs[0], shader.inputs['Emission Strength'])
                self.connect_nodes(base_texture_color_output, shader.inputs['Emission'])

            detail = self.detail
            if detail is not None:
                detail_node = self.create_node(Nodes.ShaderNodeTexImage, '$detail')
                detail_node.image = detail

                detail_scale = self.detailscale
                if detail_scale[0] != 1.0 or detail_scale[1] != 1.0:
                    mapping_node = self.add_uv_mapping(detail_scale)
                    self.connect_nodes(mapping_node.outputs[0], detail_node.inputs[0])
                detail_color_output = detail_node.outputs['Color']
                detail_alpha_output = detail_node.outputs['Alpha']
            else:
                detail_color_output = None
                detail_alpha_output = None

            decal_texture = self.decaltexture
            if decal_texture is not None:
                decal_texture_node = self.create_node(Nodes.ShaderNodeTexImage, '$decaltexture')
                decal_texture_node.image = decal_texture

                uv_node = self.add_uv_node(uv_layer_name='UV_1')
                self.connect_nodes(uv_node.outputs[0], decal_texture_node.inputs[0])
                decal_color_output = decal_texture_node.outputs['Color']
                decal_alpha_output = decal_texture_node.outputs['Alpha']
            else:
                decal_color_output = None
                decal_alpha_output = None

            bumpmap = self.bumpmap
            if bumpmap:
                bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
                bumpmap_node.image = bumpmap

                normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

                self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])

                normal_output = normalmap_node.outputs['Normal']
                normal_alpha_output = bumpmap_node.outputs['Alpha']
            else:
                normal_output = None
                normal_alpha_output = None

            if (self.color or self.color2) or self.blendtintbybasealpha:
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.blend_type = 'MULTIPLY'
                self.connect_nodes(base_texture_color_output, color_mix.inputs['Color1'])
                color_mix.inputs['Color2'].default_value = (self.color or self.color2) or (1, 1, 1, 1)
                if self.blendtintbybasealpha:
                    self.connect_nodes(base_texture_alpha_output, color_mix.inputs['Fac'])
                else:
                    color_mix.inputs['Fac'].default_value = 1.0

                base_texture_color_output = color_mix.outputs['Color']

            if detail_color_output and detail_alpha_output:
                detail_mode = self.detailmode
                if detail_mode == 1:
                    color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                    color_mix.blend_type = 'ADD'
                    color_mix.inputs['Fac'].default_value = self.detailfactor

                    self.connect_nodes(base_texture_color_output, color_mix.inputs['Color1'])
                    self.connect_nodes(detail_color_output, color_mix.inputs['Color2'])
                    base_texture_color_output = color_mix.outputs['Color']
                elif detail_mode == 2:
                    color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                    color_mix.blend_type = 'OVERLAY'

                    self.connect_nodes(base_texture_alpha_output, color_mix.inputs['Fac'])
                    self.connect_nodes(base_texture_color_output, color_mix.inputs['Color1'])
                    self.connect_nodes(detail_color_output, color_mix.inputs['Color2'])
                    base_texture_color_output = color_mix.outputs['Color']
                elif detail_mode == 8 or detail_mode == -1:
                    color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                    color_mix.blend_type = 'MULTIPLY'
                    color_mix.inputs['Fac'].default_value = self.detailfactor

                    self.connect_nodes(base_texture_color_output, color_mix.inputs['Color1'])
                    self.connect_nodes(detail_color_output, color_mix.inputs['Color2'])
                    base_texture_color_output = color_mix.outputs['Color']

            if decal_color_output and decal_alpha_output:
                math_node = self.create_node(Nodes.ShaderNodeMath)
                math_node.operation = 'MULTIPLY'
                self.connect_nodes(decal_alpha_output, math_node.inputs[0])
                math_node.inputs[1].default_value = 2.0

                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.blend_type = 'MIX'
                self.connect_nodes(math_node.outputs[0], color_mix.inputs['Fac'])

                self.connect_nodes(base_texture_color_output, color_mix.inputs['Color1'])
                self.connect_nodes(decal_color_output, color_mix.inputs['Color2'])
                base_texture_color_output = color_mix.outputs['Color']

            if self.translucent or self.alphatest:
                self.connect_nodes(base_texture_alpha_output, shader.inputs['Alpha'])

            self.connect_nodes(shader.inputs['Base Color'], base_texture_color_output)

            if normal_output:
                self.connect_nodes(normal_output, shader.inputs['Normal'])

            if self.normalmapalphaenvmapmask and normal_alpha_output:
                self.connect_nodes(shader.inputs['Metallic'], normal_alpha_output)
            elif self.envmap:
                shader.inputs['Metallic'].default_value = sum(self.envmaptint[:3]) / 3

            if self.normalmapalphaphongmask and normal_alpha_output:

                if self.phongexponent:
                    math_node = self.create_node(Nodes.ShaderNodeMath)
                    math_node.operation = 'MULTIPLY'
                    self.connect_nodes(normal_alpha_output, math_node.inputs[0])
                    math_node.inputs[1].default_value = self.clamp_value(self.phongexponent / 256) * 20
                    roughness_output = math_node.outputs[0]
                else:
                    roughness_output = normal_alpha_output

                invert_node = self.create_node(Nodes.ShaderNodeInvert)
                self.connect_nodes(roughness_output, invert_node.inputs[1])
                roughness_output = invert_node.outputs[0]
            elif self.phongexponent is not None and self.phongexponenttexture is None:
                value_node = self.create_node(Nodes.ShaderNodeValue)
                value_node.outputs[0].default_value = self.clamp_value(self.phongexponent / 256) * 25
                roughness_output = value_node.outputs[0]
            else:
                value_node = self.create_node(Nodes.ShaderNodeValue)
                value_node.outputs[0].default_value = 0.75
                roughness_output = value_node.outputs[0]

            self.connect_nodes(roughness_output, shader.inputs['Roughness'])

            if not self.phong:
                shader.inputs['Specular'].default_value = 0
            elif self.phongboost is not None:
                shader.inputs['Specular'].default_value = self.clamp_value(self.phongboost / 64)

            if self.phongtint is not None:
                layer_weight_node = self.create_node(Nodes.ShaderNodeLayerWeight)
                glossy_node = self.create_node(Nodes.ShaderNodeBsdfGlossy)
                self.connect_nodes(normal_output, glossy_node.inputs['Normal'])
                self.connect_nodes(roughness_output, glossy_node.inputs['Roughness'])
                glossy_node.inputs['Color'].default_value = self.phongtint
                mix_node = self.create_node(Nodes.ShaderNodeMixShader)
                self.connect_nodes(layer_weight_node.outputs[0], mix_node.inputs[0])
                self.connect_nodes(shader_output, mix_node.inputs[1])
                self.connect_nodes(glossy_node.outputs['BSDF'], mix_node.inputs[2])
                shader_output = mix_node.outputs[0]

            self.connect_nodes(shader_output, material_output.inputs['Surface'])

            # basetexture = self.basetexture
            # if basetexture:
            #     basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
            #     basetexture_node.image = basetexture
            #     basetexture_node.id_data.nodes.active = basetexture_node
            #
            #     if self.color or self.color2:
            #         color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
            #         color_mix.blend_type = 'MULTIPLY'
            #         self.connect_nodes(basetexture_node.outputs['Color'], color_mix.inputs['Color1'])
            #         color_mix.inputs['Color2'].default_value = (self.color or self.color2)
            #         color_mix.inputs['Fac'].default_value = 1.0
            #         self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Base Color'])
            #     else:
            #         self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
            #     if self.translucent or self.alphatest:
            #         self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])
            #
            #     if self.additive:
            #         basetexture_invert_node = self.create_node(Nodes.ShaderNodeInvert)
            #         basetexture_additive_mix_node = self.create_node(Nodes.ShaderNodeMixRGB)
            #         self.insert_node(basetexture_node.outputs['Color'], basetexture_additive_mix_node.inputs['Color1'],
            #                          basetexture_additive_mix_node.outputs['Color'])
            #         basetexture_additive_mix_node.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)
            #
            #         self.connect_nodes(basetexture_node.outputs['Color'], basetexture_invert_node.inputs['Color'])
            #         self.connect_nodes(basetexture_invert_node.outputs['Color'], shader.inputs['Transmission'])
            #         self.connect_nodes(basetexture_invert_node.outputs['Color'],
            #                            basetexture_additive_mix_node.inputs['Fac'])
            #
            # bumpmap = self.bumpmap
            # if bumpmap:
            #     bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
            #     bumpmap_node.image = bumpmap
            #
            #     normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)
            #
            #     self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
            #     self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])
            #
            # if self.selfillum and basetexture:
            #     basetexture_node = self.get_node('$basetexture')
            #     if basetexture_node:
            #         selfillummask = self.selfillummask
            #         if selfillummask is not None:
            #             selfillummask_node = self.create_node(Nodes.ShaderNodeTexImage, '$selfillummask')
            #             selfillummask_node.image = selfillummask
            #             if 'Emission Strength' in shader.inputs:
            #                 self.connect_nodes(selfillummask_node.outputs['Color'], shader.inputs['Emission Strength'])
            #
            #         else:
            #             if 'Emission Strength' in shader.inputs:
            #                 self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Emission Strength'])
            #         self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])
            #
            # if not self.phong:
            #     shader.inputs['Specular'].default_value = 0
            # elif self.phongboost is not None:
            #     shader.inputs['Specular'].default_value = self.clamp_value(self.phongboost / 64)
            # phongexponenttexture = self.phongexponenttexture
            # if self.phongexponent is not None and phongexponenttexture is None:
            #     shader.inputs['Roughness'].default_value = self.clamp_value(self.phongexponent / 256)
            # elif self.phongexponenttexture is not None:
            #     phongexponenttexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$phongexponenttexture')
            #     phongexponenttexture_node.image = phongexponenttexture
            #     phongexponenttexture_split_node = self.create_node(Nodes.ShaderNodeSeparateRGB)
            #     self.connect_nodes(phongexponenttexture_node.outputs['Color'],
            #                        phongexponenttexture_split_node.inputs['Image'])
            #
            #     phongexponenttexture_r_invert_node = self.create_node(Nodes.ShaderNodeInvert)
            #     self.connect_nodes(phongexponenttexture_split_node.outputs['R'],
            #                        phongexponenttexture_r_invert_node.inputs['Color'])
            #     self.connect_nodes(phongexponenttexture_split_node.outputs['G'],
            #                        shader.inputs['Metallic'])
            #
            #     self.connect_nodes(phongexponenttexture_r_invert_node.outputs['Color'], shader.inputs['Roughness'])
