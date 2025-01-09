from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase


class UnlitGeneric(Source1ShaderBase):
    SHADER: str = 'unlittwotexture'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def texture2(self):
        texture_path = self._vmt.get_string('$texture2', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color2(self):
        color_value, value_type = self._vmt.get_vector('$color2', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        elif len(color_value) > 3:
            color_value = color_value[:3]
        return color_value

    @property
    def color(self):
        color_value, value_type = self._vmt.get_vector('$color', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        elif len(color_value) > 3:
            color_value = color_value[:3]
        return color_value
    
    @property
    def basetexturetransform(self):
        return self._vmt.get_transform_matrix('$basetexturetransform',
            {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0),
            'translate': (0, 0, 0)})
    
    @property
    def texture2transform(self):
        return self._vmt.get_transform_matrix('$texture2transform',
            {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0),
            'translate': (0, 0, 0)})

    @property
    def additive(self):
        return self._vmt.get_int('$additive', 0) == 1
    
    @property
    def translucent(self):
        return self._vmt.get_int('$translucent', 0) == 1
    
    @property
    def nocull(self):
        return self._vmt.get_int('$nocull', 0) == 1

    def create_nodes(self, material):
        self.do_arrange = True
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)

        basetexture = self.basetexture
        texture2 = self.texture2
        if self.nocull:
            self.bpy_material.use_backface_culling = False
        else:
            self.bpy_material.use_backface_culling = True
        if basetexture:
            basetexture_node = self.create_and_connect_texture_node(basetexture, name='$basetexture')
            if self.basetexturetransform:
                uv, self.uv_map = self.handle_transform(self.basetexturetransform, basetexture_node.inputs[0])
            if texture2:
                texture2_node = self.create_and_connect_texture_node(texture2, name='$texture2')
                if self.texture2transform:
                    uv, self.uv_map = self.handle_transform(self.texture2transform, texture2_node.inputs[0])
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.name = 'twotex_mult'
                color_mix.blend_type = 'MULTIPLY'
                color_mix.inputs['Fac'].default_value = 1.0
                self.connect_nodes(basetexture_node.outputs['Color'], color_mix.inputs['Color1'])
                self.connect_nodes(texture2_node.outputs['Color'], color_mix.inputs['Color2'])
                texture_output = color_mix.outputs['Color']
            else:
                texture_output = basetexture_node.outputs['Color']

            if self.color or self.color2:
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.name = 'color_mult'
                color_mix.location
                color_mix.blend_type = 'MULTIPLY'
                self.connect_nodes(texture_output, color_mix.inputs['Color1'])
                color_mix.inputs['Color2'].default_value = (*(self.color or self.color2), 1.0)
                color_mix.inputs['Fac'].default_value = 1.0
                texture_output = color_mix.outputs[0]
            else:
                pass
        if self.additive:
            self.bpy_material.blend_method = 'BLEND'
            self.bpy_material.surface_render_method = 'BLENDED'

            transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
            add_shader = self.create_node(Nodes.ShaderNodeAddShader)

            self.connect_nodes(transparent.outputs[0], add_shader.inputs[0])
            self.connect_nodes(texture_output, add_shader.inputs[1])

            texture_output = add_shader.outputs[0]
        
        if self.translucent:
            self.bpy_material.blend_method = 'BLEND'
            self.bpy_material.surface_render_method = 'BLENDED'

            mix = self.create_node(Nodes.ShaderNodeMixShader)
            transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)

            self.connect_nodes(texture_output, mix.inputs[2])
            self.connect_nodes(transparent.outputs[0], mix.inputs[1])
            self.connect_nodes(basetexture_node.outputs[1], mix.inputs[0])
            texture_output = mix.outputs[0]

        culling = self.create_node_group('BackfaceCulling')

        # so we can toggle backface culling across eevee and cycles with one boolean property
        driv = culling.inputs['$nocull'].driver_add('default_value')
        driv.driver.expression = '1-var'
        var = driv.driver.variables.new()
        var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'MATERIAL'
        var.targets[0].id = self.bpy_material
        var.targets[0].data_path = 'use_backface_culling'

        self.connect_nodes(texture_output, culling.inputs[0])
        self.connect_nodes(culling.outputs[0], material_output.inputs[0])