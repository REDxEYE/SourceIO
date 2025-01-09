from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase


class Modulate(Source1ShaderBase):
    SHADER: str = 'modulate'

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
    def additive(self):
        return self._vmt.get_int('$additive', 0) == 1
    
    @property
    def mod2x(self):
        return self._vmt.get_int('$mod2x', 0) == 1

    @property
    def translucent(self):
        return self._vmt.get_int('$translucent', 0) == 1
    
    @property
    def nocull(self):
        return self._vmt.get_int('$nocull', 0) == 1

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        self.bpy_material.blend_method = 'BLEND'
        self.bpy_material.surface_render_method = 'BLENDED'
        mult = self.create_node(Nodes.ShaderNodeVectorMath)
        mult.operation = 'MULTIPLY'
        mult.inputs[0].default_value = (1, 1, 1)
        mult.inputs[1].default_value = (1, 1, 1)
        if self.basetexture:
            basetexture_node = self.create_texture_node(self.basetexture)
            self.connect_nodes(basetexture_node.outputs[0], mult.inputs[0])
        if self.mod2x:
            mult.inputs[1].default_value = (2, 2, 2)
        self.bpy_material.use_backface_culling = not bool(self.nocull)
        transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
        self.connect_nodes(mult.outputs[0], transparent.inputs[0])
        out = self.create_node(Nodes.ShaderNodeOutputMaterial)

        culling = self.create_node_group('BackfaceCulling')
        self.connect_nodes(transparent.outputs[0], culling.inputs[0])

        # so we can toggle backface culling across eevee and cycles with one boolean property
        driv = culling.inputs['$nocull'].driver_add('default_value')
        driv.driver.expression = '1-var'
        var = driv.driver.variables.new()
        var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'MATERIAL'
        var.targets[0].id = self.bpy_material
        var.targets[0].data_path = 'use_backface_culling'

        self.connect_nodes(culling.outputs[0], out.inputs[0])