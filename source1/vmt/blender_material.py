import bpy
from pathlib import Path

from ..vmt.vmt import VMT
from ..vtf.import_vtf import import_texture


class BlenderMaterial:
    def __init__(self, vmt: VMT):
        vmt.parse()
        self.vmt = vmt
        self.textures = {}

    def load_textures(self):
        for key, (name, texture) in self.vmt.textures.items():
            if bpy.data.images.get(name, False):
                self.textures[key] = bpy.data.images.get(name, False)
            else:
                image = import_texture(name, texture)
                if image:
                    self.textures[key] = bpy.data.images.get(image)

    def create_material(self, material_name=None, override=True):
        if bpy.data.materials.get(material_name) and not override:
            return 'EXISTS'
        else:
            bpy.data.materials.new(material_name)
        mat = bpy.data.materials.get(material_name)
        mat.blend_method = 'HASHED'
        mat.shadow_method = 'HASHED'

        if mat.get('source1_loaded'):
            return 'LOADED'
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        diff = nodes.get('Principled BSDF', None)
        if diff:
            nodes.remove(diff)
        out = nodes.get('ShaderNodeOutputMaterial', None)
        if not out:
            out = nodes.get('Material Output', None)
        if not out:
            out = nodes.new('ShaderNodeOutputMaterial')
        out.location = (385.0, 146.0)
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (45.0, 146.0)
        mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs['Surface'])
        if self.textures.get('$basetexture', False):
            tex = nodes.new('ShaderNodeTexImage')
            tex.image = self.textures.get('$basetexture')
            tex.location = (-295.0, 146.0)
            mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs['Base Color'])
            mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs['Alpha'])
        if self.textures.get('$bumpmap', False):
            tex = nodes.new('ShaderNodeTexImage')
            tex.image = self.textures.get('$bumpmap')
            tex.location = (-635.0, 146.0)
            tex.image.colorspace_settings.is_data = True
            tex.image.colorspace_settings.name = 'Non-Color'

            normal = nodes.new("ShaderNodeNormalMap")
            normal.location = (-295.0, -125.0)
            mat.node_tree.links.new(tex.outputs["Color"], normal.inputs['Color'])
            mat.node_tree.links.new(normal.outputs["Normal"], bsdf.inputs['Normal'])
        if self.textures.get('$phongexponenttexture', False):
            tex = nodes.new('ShaderNodeTexImage')
            tex.image = self.textures.get('$phongexponenttexture')
            tex.location = (-200, 0)
            # mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs['Base Color'])

        mat['source1_loaded'] = True