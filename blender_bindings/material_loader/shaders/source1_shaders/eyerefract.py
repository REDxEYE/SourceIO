from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase


class EyeRefract(Source1ShaderBase):
    SHADER: str = 'eyerefract'

    @property
    def iris(self):
        texture_path = self._vmt.get_string('$iris', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None
    
    @property
    def ambientoccltexture(self):
        texture_path = self._vmt.get_string('$ambientoccltexture', None)
        if texture_path:
            return self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
        
    @property
    def corneatexture(self):
        texture_path = self._vmt.get_string('$corneatexture', None)
        if texture_path:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 0.0, 1.0))
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        
    @property
    def lightwarptexture(self):
        texture_path = self._vmt.get_string('$lightwarptexture', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None
    
    @property
    def eyeballradius(self):
        return self._vmt.get_float('$eyeballradius', 0.5)
    
    @property
    def ambientocclcolor(self):
        color_value, value_type = self._vmt.get_vector('$ambientocclcolor', (0.33, 0.33, 0.33))
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)
    
    @property
    def dilation(self):
        return self._vmt.get_float('$dilation', 0.5)
    
    @property
    def parallaxstrength(self):
        return self._vmt.get_float('$parallaxstrength', 0.25)
    
    @property
    def corneabumpstrength(self):
        return self._vmt.get_float('$corneabumpstrength', 1.0)
    
    @property
    def raytracesphere(self):
        return self._vmt.get_int('$raytracesphere', 1)
    
    @property
    def irisu(self):
        mask, _ = self._vmt.get_vector('$irisu', (0.0, 1.0, 0.0, 0.0))
        return mask[:3]

    @property
    def irisv(self):
        mask, _ = self._vmt.get_vector('$irisv', (0.0, 1.0, 0.0, 0.0))
        return mask[:3]

    def create_nodes(self, material:bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        pass
        eye_template = bpy.data.materials['SIO_eye_template']
        new_eye_material: bpy.types.Material = eye_template.copy()
        new_eye_material.name = 'NEW'
        new_eye_material.use_fake_user = True
        node_tree = new_eye_material.node_tree
        nodes = node_tree.nodes

        if self.iris:
            nodes['$Iris'].image = self.iris
            nodes['fCorneaNoise'].image = self.iris
        if self.ambientoccltexture:
            nodes['$AmbientOcclTexture'].image = self.ambientoccltexture
        if self.corneatexture:
            nodes['$CorneaTexture'].image = self.corneatexture
            nodes['$CorneaTexture_1'].image = self.corneatexture
        if self.lightwarptexture:
            nodes['$lightwarptexture'].image = self.lightwarptexture
        
        nodes['$EyeballRadius'].outputs[0].default_value = self.eyeballradius
        nodes['$AmbientOcclColor'].outputs[0].default_value = self.ambientocclcolor
        nodes['$Dilation'].outputs[0].default_value = self.dilation
        nodes['$ParallaxStrength'].outputs[0].default_value = self.parallaxstrength
        nodes['$CorneaBumpStrength'].outputs[0].default_value = self.corneabumpstrength
        nodes['$RaytraceSphere'].outputs[0].default_value = self.raytracesphere
        nodes['$IrisU'].outputs[0].default_value = self.irisu
        nodes['$IrisV'].outputs[0].default_value = self.irisv
        
        wanted_name = self.bpy_material.name
        self.bpy_material.name = '!'
        
        for key, value in self.bpy_material.items():
            if isinstance(value, (float, int, bool, str)):
                pass
            elif hasattr(value, 'to_list'):
                value = value.to_list()
            elif hasattr(value, 'to_dict'):
                value = value.to_dict()
            else:
                continue
            new_eye_material[key] = value
        
        material.user_remap(new_eye_material)
        new_eye_material.name = wanted_name
        old_material = self.bpy_material
        self.bpy_material = new_eye_material
        self.do_arrange = False
        bpy.data.materials.remove(old_material)
        return new_eye_material