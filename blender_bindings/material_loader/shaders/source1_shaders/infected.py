from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4
from .detail import DetailSupportMixin


class Infected(DetailSupportMixin, Source1ShaderBase):
    SHADER: str = 'infected'

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
            image = self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None
    
    @property
    def gradienttexture(self):
        texture_path = self._vmt.get_string('$gradienttexture', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
            return image
        return None
    
    @property
    def detail(self):
        texture_path = self._vmt.get_string('$detail', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
            return image
        return None
    
    @property
    def detailscale(self):
        value = self._vmt.get_float('$detailscale', 1.0)
        return value

    @property
    def basetexturetransform(self):
        return self._vmt.get_transform_matrix('$basetexturetransform',
                                              {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0),
                                               'translate': (0, 0, 0)})

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
    def phong(self):
        return self._vmt.get_int('$phong', 0) == 1
    
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
        value = self._vmt.get_float('$defaultphongexponent', None)
        return value

    @property
    def phongboost(self):
        value = self._vmt.get_float('$phongboost', 1)
        return value

    @property
    def phongtint(self):
        color_value, value_type = self._vmt.get_vector('$phongtint', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: (a / divider) ** 2.2 if value_type is int else 1, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)
    ### BLOOD ###

    @property
    def bloodcolor(self):
        color_value, value_type = self._vmt.get_vector('$bloodcolor', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        #color_value = list(map(lambda a: (a / divider) ** (2.2 if value_type is int else 1), color_value))
        #print(f'\n\n{color_value}\n\n')
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)
    
    @property
    def bloodspecboost(self):
        value = self._vmt.get_float('$bloodspecboost', 1.0)
        return value

    @property
    def bloodphongexponent(self):
        value = self._vmt.get_int('$bloodphongexponent', 200)
        return value
    
    @property
    def bloodmaskrange(self):
        value, value_type = self._vmt.get_vector('$bloodmaskrange', None)
        if value is not None:
            divider = 255 if value_type is int else 1
            value = list(map(lambda a: a / divider, value))
            return self.ensure_length(value, 3, 0.0)
        return None
    
    ###############

    ### EYEGLOW ###

    @property
    def eyeglow(self):
        value = self._vmt.get_int('$eyeglow', 0.0)
        return value
    
    @property
    def eyeglowcolor(self):
        color_value, value_type = self._vmt.get_vector('$eyeglowcolor', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: (a / divider) ** (2.2 if value_type is int else 1), color_value))
        print(f'EYEGLOW {color_value}')
        #if len(color_value) == 1:
        #    color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def eyeglowflashlightboost(self):
        value = self._vmt.get_float('$eyeglowflashlightboost', 1.0)
        return value
    
    ###############

    @property
    def sheetindex(self):
        value = self._vmt.get_int('$sheetindex', -1)
        return value
    
    @property
    def skintintgradient(self):
        value = self._vmt.get_int('$skintintgradient', -1)
        return value
    
    @property
    def colortintgradient(self):
        value = self._vmt.get_int('$colortintgradient', -1)
        return value
    
    @property
    def skinphongexponent(self):
        value = self._vmt.get_int('$skinphongexponent', 16)
        return value

    @property
    def eyeglowflashlightboost(self):
        value = self._vmt.get_float('$eyeglowflashlightboost', 1.0)
        return value

    @property
    def disablevariation(self):
        value = self._vmt.get_int('$disablevariation', 0)
        return value

    def create_nodes(self, material):
        self.do_arrange = False
        print(f"BVLG: {self.use_bvlg_status}")
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        material_output.location = [400, -140]
        parentnode = material_output

        uv = None
        if self.use_bvlg_status:
            self.do_arrange = False

            group_node = self.create_node_group("infected", [-60, -140])
            self.connect_nodes(group_node.outputs['BSDF'], parentnode.inputs[0])
            if self.basetexture:
                if not self.disablevariation:
                    basetexture_sprite_node = self.create_and_connect_texture_node(self.basetexture,
                                                                            group_node.inputs['$basetexture sprite [texture]'],
                                                                            name='$basetexture sprite')
                    
                    base_img = basetexture_sprite_node.image

                    basetexture_sprite_node.location = [-400, 20]

                    #sprite_sheet_uv = self.create_node_group('SpriteSheet', [-660, 20])
                    #sprite_sheet_uv.width=140
                    #self.connect_nodes(sprite_sheet_uv.outputs[0], basetexture_sprite_node.inputs[0])

                    randomIndex = self.create_node_group('random2x2Tile', [-840, 20])
                    randomIndex.width = 140
                    self.connect_nodes(randomIndex.outputs[0], basetexture_sprite_node.inputs[0])

                    basetexture = self.create_texture_node(self.basetexture, name='$basetexture', location=[-800, -220])
                    gradient = self.create_texture_node(self.gradienttexture, name='$gradienttexture', location=[-340, -300])
                    gradient.extension = 'CLIP'
                    gradient.interpolation = 'Closest'
                    gradientSampler = self.create_node_group('sampleColorPalette', [-540, -420])
                    gradientSampler.width = 140

                    self.connect_nodes(basetexture_sprite_node.outputs[0], group_node.inputs['$basetexture sprite [texture]'])
                    self.connect_nodes(basetexture.outputs[0], group_node.inputs['$basetexture [texture]'])
                    self.connect_nodes(basetexture.outputs[0], gradientSampler.inputs[0])
                    self.connect_nodes(basetexture.outputs[1], group_node.inputs['$basetexture alpha [float]'])
                    self.connect_nodes(basetexture.outputs[1], gradientSampler.inputs[1])
                    self.connect_nodes(gradientSampler.outputs[0], gradient.inputs[0])
                    self.connect_nodes(gradient.outputs[0], group_node.inputs['$gradienttexture'])

                    if self.sheetindex:
                        randomIndex.inputs[2].default_value = self.sheetindex

                    if self.skintintgradient:
                        gradientSampler.inputs[2].default_value = self.skintintgradient

                    if self.colortintgradient:
                        gradientSampler.inputs[3].default_value = self.colortintgradient

                    if self.detail:
                        detail = self.create_and_connect_texture_node(self.detail,
                            group_node.inputs['$detail [texture]'],
                            name='$detail')
                        detail.location = [-380, -600]
                        
                        if self.detailscale:
                            uv = self.create_node(Nodes.ShaderNodeUVMap, name='UV Map', location=[-760, -700])
                            scaler = self.create_node(Nodes.ShaderNodeVectorMath, name='$detailscale', location=[-580, -700])
                            scaler.inputs[3].default_value = self.detailscale
                            scaler.operation = 'SCALE'

                            self.connect_nodes(uv.outputs[0], scaler.inputs[0])
                            self.connect_nodes(scaler.outputs[0], detail.inputs[0])

                else:
                    basetexture_node = self.create_and_connect_texture_node(self.basetexture,
                                                                            group_node.inputs['$basetexture [texture]'],
                                                                            name='$basetexture')

                    basetexture_node.image.colorspace_settings.name = 'sRGB'
                    basetexture_node.image.colorspace_settings.is_data = False
                    basetexture_node.location = [-380, -140]

                    group_node.inputs['$disablevariation'].default_value = True

            if self.bumpmap:
                bumpmap_node = self.create_and_connect_texture_node(self.bumpmap,
                                                                    group_node.inputs['$bumpmap [texture]'],
                                                                    name='$bumpmap')
                bumpmap_node.location = [-800, -220]
                if self.normalmapalphaenvmapmask:
                    self.connect_nodes(bumpmap_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])
            
            if self.bloodcolor:
                group_node.inputs['$bloodcolor [vector]'].default_value = self.bloodcolor
                print(f'\n\nBLOODCOLOR {self.bloodcolor}\n\n')

            if self.bloodmaskrange:
                group_node.inputs['$bloodmaskrange [vector2]'].default_value = self.bloodmaskrange

            if self.bloodspecboost:
                group_node.inputs['$bloodspecboost'].default_value = self.bloodspecboost
            
            if self.bloodphongexponent:
                group_node.inputs['$bloodphongexponent'].default_value = self.bloodphongexponent

            if self.eyeglow:
                group_node.inputs['$eyeglow [bool]'].default_value = self.eyeglow

            if self.eyeglowcolor:
                group_node.inputs['$eyeglowcolor [color]'].default_value = self.eyeglowcolor

            if self.eyeglowflashlightboost:
                group_node.inputs['$eyeglowflashlightboost [float]'].default_value = self.eyeglowflashlightboost

            if self.skinphongexponent:
                group_node.inputs['$skinphongexponent [int]'].default_value = self.skinphongexponent

            if self.phong:
                group_node.inputs['$phong [bool]'].default_value = 1
                if self.phongboost:
                    group_node.inputs['$phongboost [value]'].default_value = self.phongboost
                if self.phongexponent:
                    group_node.inputs['$defaultphongexponent [value]'].default_value = self.phongexponent

                else:
                    group_node.inputs['$defaultphongexponent [value]'].default_value = 10

                if self.phongtint is not None:
                    group_node.inputs['$phongtint [RGB field]'].default_value = self.phongtint

                if self.phongfresnelranges:
                    group_node.inputs['$phongfresnelranges [value field]'].default_value = self.phongfresnelranges

        else:
            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
            self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

            basetexture = self.basetexture
            if basetexture:
                basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
                basetexture_node.image = basetexture
                basetexture_node.id_data.nodes.active = basetexture_node

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
                    if is_blender_4():
                        self.connect_nodes(basetexture_invert_node.outputs['Color'],
                                           shader.inputs['Transmission Weight'])
                    else:
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

            if self.selfillum and basetexture:
                basetexture_node = self.get_node('$basetexture')
                if basetexture_node:
                    selfillummask = self.selfillummask
                    if selfillummask is not None:
                        selfillummask_node = self.create_node(Nodes.ShaderNodeTexImage, '$selfillummask')
                        selfillummask_node.image = selfillummask
                        if 'Emission Strength' in shader.inputs:
                            self.connect_nodes(selfillummask_node.outputs['Color'], shader.inputs['Emission Strength'])
                    else:
                        if 'Emission Strength' in shader.inputs:
                            self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Emission Strength'])
                    if is_blender_4():
                        self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission Color'])
                    else:
                        self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])

            if not self.phong:
                if is_blender_4():
                    shader.inputs['Specular IOR Level'].default_value = 0
                else:
                    shader.inputs['Specular'].default_value = 0
            elif self.phongboost is not None:
                if is_blender_4():
                    shader.inputs['Specular IOR Level'].default_value = self.clamp_value(self.phongboost / 64)
                else:
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


class SDKInfected(Infected):
    SHADER: str = 'sdk_infected'
