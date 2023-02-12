from typing import Iterable

import bpy
import numpy as np

from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class Water(Source1ShaderBase):
    SHADER: str = 'water'

    @property
    def bumpmap(self):
        texture_path = self._vmt.get_string('$normalmap', None)
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
    def bluramount(self):
        value = self._vmt.get_float('$bluramount', 0)
        return value

    @property
    def color(self):
        color_value, value_type = self._vmt.get_vector('$color', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        return self.ensure_length(color_value, 4, color_value[0])

    @property
    def refracttint(self):
        color_value, value_type = self._vmt.get_vector('$refracttint', [0.85, 0.9, 0.95])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def reflecttint(self):
        color_value, value_type = self._vmt.get_vector('$reflecttint', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def abovewater(self):
        value = self._vmt.get_int('$abovewater', 0)
        return value

    def create_probe(self):
        try:
            if (bpy.context.scene.objects.get("ReflectionPlane") is not None):
                print("Probe already exists")
                return ()
            ob = bpy.context.scene.objects["world_geometry"]
            c1_slots = [id for id, mat in enumerate(ob.data.materials) if mat == self.bpy_material]
            me = ob.data
            points = [v.center for v in me.polygons if v.material_index == c1_slots[0]]
            z = -300
            for pt in points:
                z = max(pt.z, z)
            bpy.ops.object.lightprobe_add(type='PLANAR', align='WORLD', radius=300, location=(0, 0, z + 0.02),
                                          scale=(300, 300, 1))
        except Exception as e:
            print("Failed to establish water plane: " + str(e))
        return

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        self.bpy_material.blend_method = 'OPAQUE'
        self.bpy_material.shadow_method = 'NONE'
        self.bpy_material.use_screen_refraction = True
        self.bpy_material.use_backface_culling = True
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)

        # if self.abovewater:
        #    self.create_probe()
        if self.use_bvlg_status:
            group_node = self.create_node(Nodes.ShaderNodeGroup, self.SHADER)
            group_node.node_tree = bpy.data.node_groups.get("Water")
            group_node.width = group_node.bl_width_max
            self.connect_nodes(group_node.outputs['BSDF'], material_output.inputs['Surface'])
            bumpmap = self.bumpmap
            if bumpmap:
                bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$normalmap')
                bumpmap_node.image = bumpmap
                self.connect_nodes(bumpmap_node.outputs['Color'], group_node.inputs['$normalmap'])
            group_node.inputs['$reflecttint'].default_value = self.reflecttint
            group_node.inputs['$refracttint'].default_value = self.refracttint


        else:
            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
            self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
            basetexture = self.basetexture
            if basetexture:
                basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
                basetexture_node.image = basetexture
                basetexture_node.id_data.nodes.active = basetexture_node

                self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
                self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Roughness'])

            bumpmap = self.bumpmap
            if bumpmap:
                bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
                bumpmap_node.image = bumpmap

                normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

                self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
                self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])
                shader.inputs['Transmission'].default_value = 1.0
                shader.inputs['Roughness'].default_value = self.bluramount
