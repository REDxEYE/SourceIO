from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4, is_blender_4_3
from .detail import DetailSupportMixin

class VertexLitGeneric(DetailSupportMixin, Source1ShaderBase):
    SHADER: str = 'vertexlitgeneric'

    @property
    def bumpmap(self):
        texture_path = self._vmt.get_string('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            if image == self.basetexture:
                return image
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
    def lightwarptexture(self):
        texture_path = self._vmt.get_string('$lightwarptexture', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
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
    def colortint_base(self):
        color_value, value_type = self._vmt.get_vector('$colortint_base', None)
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
    def selfillummaskscale(self):
        return self._vmt.get_float('$selfillummaskscale', 1.0)

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
    def selfillumtint(self):
        color_value, value_type = self._vmt.get_vector('$selfillumtint', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

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
        color_value, value_type = self._vmt.get_vector('$envmaptint', [1, 1, 1])
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
    def rimlight(self):
        value = self._vmt.get_int('$rimlight', 0)
        return value

    @property
    def rimlightexponent(self):
        value = self._vmt.get_int('$rimlightexponent', 0)
        return value

    @property
    def rimlightboost(self):
        value = self._vmt.get_float('$rimlightboost', 0)
        return value

    @property
    def phongalbedotint(self):
        return self._vmt.get_int('$phongalbedotint', 1) == 1
    
    @property
    def blendtintbybasealpha(self):
        return self._vmt.get_float('$blendtintbybasealpha', 0)
    
    @property
    def blendtintcoloroverbase(self):
        return self._vmt.get_float('$blendtintcoloroverbase', 0)
    
    @property
    def nocull(self):
        return self._vmt.get_int('$nocull', 0) == 1
    
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

    def create_nodes(self, material):
        sio_diffuse = None
        selfillum = False

        print(f"BVLG: {self.use_bvlg_status}")
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        
        if 'proxies' in self._vmt:
            proxies = self._vmt.get('proxies')
            for proxy_name, proxy_data in proxies.items():
                if proxy_name == 'selectfirstifnonzero':
                    result_var = proxy_data.get('resultvar')
                    if result_var not in self._vmt:
                        continue
                    src1_var = proxy_data.get('srcvar1')
                    src2_var = proxy_data.get('srcvar2')
                    src1_value, src1_type = self._vmt.get_vector(src1_var, [0])
                    if all([val > 0 for val in src1_value]):
                        if src1_var not in self._vmt:
                            continue
                        self._vmt[result_var] = self._vmt[src1_var]
                    else:
                        if src2_var not in self._vmt:
                            continue
                        self._vmt[result_var] = self._vmt[src2_var]


        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        material_output.location = [250, 0]
        if not is_blender_4_3() and (self.alphatest or self.translucent):
            if self.translucent:
                self.bpy_material.blend_method = 'BLEND'
                if hasattr(self.bpy_material, 'surface_render_method'):
                    self.bpy_material.surface_render_method = 'BLENDED'
            else:
                self.bpy_material.blend_method = 'HASHED'
            self.bpy_material.shadow_method = 'HASHED'
        if self.additive:
            self.bpy_material.blend_method = 'BLEND'
            self.bpy_material.surface_render_method = 'BLENDED'
        uv = None
        if self.use_bvlg_status:
            self.do_arrange = True

            self.bpy_material.use_backface_culling = not self.nocull

            group_node = self.create_node_group("VertexLitGeneric", [-200, 0])

            # so we can toggle backface culling across eevee and cycles with one boolean property
            driv = group_node.inputs['$nocull'].driver_add('default_value')
            driv.driver.expression = '1-var'
            var = driv.driver.variables.new()
            var.type = 'SINGLE_PROP'
            var.targets[0].id_type = 'MATERIAL'
            var.targets[0].id = self.bpy_material
            var.targets[0].data_path = 'use_backface_culling'

            final = group_node.outputs[0]
            
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
                #if self.alphatest or self.translucent:
                #    self.connect_nodes(basetexture_node.outputs['Alpha'],
                #                       alphatest_node.inputs['Alpha [basemap texture alpha]'])

                if self.detail:
                    albedo, detail = self.handle_detail(group_node.inputs['$basetexture [texture]'], albedo, uv_node=uv)
            elif self.color:
                group_node.inputs['$basetexture [texture]'].default_value = self.color

            if self.color2 or self.colortint_base:
                group_node.inputs['$color2 [RGB field]'].default_value = self.color2 or self.colortint_base

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

            if self.rimlight:
                group_node.inputs['$rimlight [bool]'].default_value = self.rimlight
            if self.rimlightexponent:
                group_node.inputs['$rimlightexponent [value]'].default_value = self.rimlightexponent
            if self.rimlightboost:
                group_node.inputs['$rimlightboost [value]'].default_value = self.rimlightboost

            if self.blendtintbybasealpha:
                group_node.inputs['$blendtintbybasealpha'].default_value = self.blendtintbybasealpha

            if self.blendtintcoloroverbase:
                group_node.inputs['$blendtintcoloroverbase'].default_value = self.blendtintcoloroverbase

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
                if self.selfillummaskscale:
                    group_node.inputs['$selfillummaskscale'].default_value = self.selfillummaskscale
                if self.selfillumtint:
                    group_node.inputs['$selfillumtint'].default_value = self.selfillumtint

            if self.lightwarptexture:
                material_output.target = 'EEVEE'
                cycles_output = self.create_node(Nodes.ShaderNodeOutputMaterial, name='Cycles Output')
                cycles_output.target = 'CYCLES'

                lightwarp = self.create_texture_node(self.lightwarptexture, 'lightwarptexture')
                lightwarp.extension = 'EXTEND'
                self.connect_nodes(group_node.outputs['lightwarpvector'], lightwarp.inputs[0])
                sio_diffuse = self.create_node_group('SIO-Diffuse', name='Diffuse')
                self.connect_nodes(lightwarp.outputs[0], sio_diffuse.inputs['Lightwarp'])
                self.connect_nodes(group_node.outputs['$bumpmap'], sio_diffuse.inputs['$bumpmap'])
                self.connect_nodes(group_node.outputs['Final Albedo'], sio_diffuse.inputs['Albedo'])
                self.connect_nodes(group_node.outputs['Lighting'], sio_diffuse.inputs['Light Pass'])

                # so we can toggle backface culling across eevee and cycles with one boolean property
                driv = sio_diffuse.inputs['$nocull'].driver_add('default_value')
                driv.driver.expression = '1-var'
                var = driv.driver.variables.new()
                var.type = 'SINGLE_PROP'
                var.targets[0].id_type = 'MATERIAL'
                var.targets[0].id = self.bpy_material
                var.targets[0].data_path = 'use_backface_culling'

                final = sio_diffuse.outputs['EEVEE Pass']
                final_cycles = sio_diffuse.outputs['Cycles Pass']

            if self.additive:
                if sio_diffuse:
                    add_eevee = self.create_node(Nodes.ShaderNodeAddShader)
                    add_cycles = self.create_node(Nodes.ShaderNodeAddShader)
                    transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
                    self.connect_nodes(sio_diffuse.outputs['EEVEE Pass'], add_eevee.inputs[1])
                    self.connect_nodes(sio_diffuse.outputs['Cycles Pass'], add_cycles.inputs[1])
                    self.connect_nodes(transparent.outputs[0], add_eevee.inputs[0])
                    self.connect_nodes(transparent.outputs[0], add_cycles.inputs[0])
                    final = add_eevee.outputs[0]
                    final_cycles = add_cycles.outputs[0]
                else:
                    add = self.create_node(Nodes.ShaderNodeAddShader)
                    transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
                    self.connect_nodes(final, add.inputs[1])
                    self.connect_nodes(transparent.outputs[0], add.inputs[0])
                    final = add.outputs[0]
                    
            if self.alphatest or self.translucent:
                alphatest_node = self.create_node_group("$alphatest", [250, 0])
                material_output.location = [450, 0]
                alphatest_node.inputs['$alphatestreference [value]'].default_value = self.alphatestreference
                alphatest_node.inputs['$allowalphatocoverage [boolean]'].default_value = self.allowalphatocoverage
                alphatest_node.inputs['$translucent'].default_value = self.translucent
                if self.basetexture:
                    self.connect_nodes(basetexture_node.outputs[1], alphatest_node.inputs['Alpha [basemap texture alpha]'])
                self.connect_nodes(final, alphatest_node.inputs[0])
                final = alphatest_node.outputs[0]
                if sio_diffuse:
                    alphatest_node = self.create_node_group("$alphatest", [250, 0])
                    material_output.location = [450, 0]
                    alphatest_node.inputs['$alphatestreference [value]'].default_value = self.alphatestreference
                    alphatest_node.inputs['$allowalphatocoverage [boolean]'].default_value = self.allowalphatocoverage
                    alphatest_node.inputs['$translucent'].default_value = self.translucent
                    if self.basetexture:
                        self.connect_nodes(basetexture_node.outputs[1], alphatest_node.inputs['Alpha [basemap texture alpha]'])
                    self.connect_nodes(final_cycles, alphatest_node.inputs[0])
                    final_cycles = alphatest_node.outputs[0]
            
            # Finalize outputs
            self.connect_nodes(final, material_output.inputs[0])
            if sio_diffuse:
                self.connect_nodes(final_cycles, cycles_output.inputs[0])


            '''
            END OF BVLG
            '''

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


class SDKVertexLitGeneric(VertexLitGeneric):
    SHADER: str = 'sdk_vertexlitgeneric'
