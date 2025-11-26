from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase


class Eyes(Source1ShaderBase):
    SHADER: str = 'eyes'

    @property
    def iris(self):
        texture_path = self._vmt.get_string('$iris', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None
    
    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None
    
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
        eye_template = bpy.data.materials['SIO_simple_eye_template']
        new_eye_material: bpy.types.Material = eye_template.copy()
        new_eye_material.name = 'NEW'
        new_eye_material.use_fake_user = True
        node_tree = new_eye_material.node_tree
        nodes = node_tree.nodes
        if self.iris:
            nodes['$Iris'].image = self.iris
        if self.basetexture:
            nodes['$Basetexture'].image = self.basetexture

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