import bpy
from pathlib import Path

import numpy as np

from ..content_manager import ContentManager
from ..vmt.vmt import VMT
from ..vtf.import_vtf import import_texture
from ...utilities.path_utilities import is_valid_path


class BlenderMaterial(VMT):
    def __init__(self, file_object):
        super().__init__(file_object)
        self.parse()
        self.textures = {}

    def load_textures(self):
        content_manager = ContentManager()
        for key, value in self.material_data.items():
            if key == '$envmap':
                continue
            if isinstance(value, str):
                if not is_valid_path(value) or value.replace('.', '').isdigit():
                    continue
                name = Path(value).stem
                if bpy.data.images.get(name, False):
                    print(f'Using existing texture {name}')
                    self.textures[key] = bpy.data.images.get(name)
                    continue
                texture = content_manager.find_texture(value)
                if texture:
                    print(key, value)
                    image = import_texture(name, texture)
                    if image:
                        self.textures[key] = bpy.data.images.get(image)

    def convert_ssbump(self, image):
        print(f'Converting {image.name} SSBump to normal map')
        if bpy.app.version > (2, 83, 0):
            buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
            image.pixels.foreach_get(buffer)
        else:
            buffer = np.array(image.pixels[:])
        buffer[0::4] *= 0.5
        buffer[0::4] += 0.33
        buffer[1::4] *= 0.5
        buffer[1::4] += 0.33
        buffer[2::4] *= 0.2
        buffer[2::4] += 0.8
        if bpy.app.version > (2, 83, 0):
            image.pixels.foreach_set(buffer.tolist())
        else:
            image.pixels[:] = buffer.tolist()
        image.pack()
        return image

    def create_material(self, material_name=None, override=True):
        material_name = material_name[:63]
        print(f'Creating material {repr(material_name)}, override:{override}')
        if bpy.data.materials.get(material_name) and not override:
            return 'EXISTS'
        else:
            bpy.data.materials.new(material_name)
        mat = bpy.data.materials.get(material_name)
        if mat is None:
            return
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
            basetexture_node = nodes.new('ShaderNodeTexImage')
            basetexture_node.image = self.textures.get('$basetexture')
            basetexture_node.name = '$basetexture'
            basetexture_node.location = (-295.0, 146.0)
            mat.node_tree.links.new(basetexture_node.outputs["Color"], bsdf.inputs['Base Color'])

            alpha_output = basetexture_node.outputs["Alpha"]
            if int(self.material_data.get('$basemapalphaphongmask', '0')) == 1:
                mat.node_tree.links.new(alpha_output, bsdf.inputs['Specular'])
            elif int(self.material_data.get('$basealphaenvmapmask', '0')) == 1:
                mat.node_tree.links.new(alpha_output, bsdf.inputs['Roughness'])
            elif int(self.material_data.get('$selfillum', '0')) == 1:
                mat.node_tree.links.new(basetexture_node.outputs["Color"], bsdf.inputs['Emission'])
                mat.node_tree.links.new(alpha_output, bsdf.inputs['Emission Strength'])
            elif int(self.material_data.get('$alphatest', '0')) == 1:
                mat.node_tree.links.new(alpha_output, bsdf.inputs['Alpha'])
            elif int(self.material_data.get('$translucent', '0')) == 1:
                mat.node_tree.links.new(alpha_output, bsdf.inputs['Alpha'])
        if self.textures.get('$bumpmap', False):
            bumpmap_texture = nodes.new('ShaderNodeTexImage')
            bumpmap_texture.name = '$bumpmap'
            image = self.textures.get('$bumpmap')
            if int(self.material_data.get('$ssbump', '0')):
                image = self.convert_ssbump(image)
            bumpmap_texture.image = image
            bumpmap_texture.location = (-635.0, 146.0)
            bumpmap_texture.image.colorspace_settings.is_data = True
            bumpmap_texture.image.colorspace_settings.name = 'Non-Color'

            if int(self.material_data.get('$normalmapalphaenvmapmask', '0')) == 1:
                mat.node_tree.links.new(bumpmap_texture.outputs["Alpha"], bsdf.inputs['Specular'])

            normal = nodes.new("ShaderNodeNormalMap")
            normal.location = (-295.0, -125.0)
            mat.node_tree.links.new(bumpmap_texture.outputs["Color"], normal.inputs['Color'])
            mat.node_tree.links.new(normal.outputs["Normal"], bsdf.inputs['Normal'])
        if self.textures.get('$phongexponenttexture', False):
            tex = nodes.new('ShaderNodeTexImage')
            tex.name = '$phongexponenttexture'
            tex.image = self.textures.get('$phongexponenttexture')
            tex.location = (-200, 0)

        mat.blend_method = 'HASHED'
        mat.shadow_method = 'HASHED'

        mat['source1_loaded'] = True
