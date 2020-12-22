from typing import Dict, Type

from ..vmt.valve_material import VMT
from .shader_base import ShaderBase
from .shaders import vertexlit_generic, lightmap_generic, worldvertextransition, cable,unlit_generic


class BlenderMaterial:
    def __init__(self, file_object, material_name):
        self.material_name: str = material_name[-63:]
        self.vmt: VMT = VMT(file_object)
        self._handlers: Dict[str, Type[ShaderBase]] = dict()

        sub: Type[ShaderBase]
        for sub in ShaderBase.__subclasses__():
            self._handlers[sub.SHADER] = sub

        self.vmt.parse()

    def create_material(self):
        handler: ShaderBase = self._handlers.get(self.vmt.shader, ShaderBase)(self.vmt)
        handler.create_nodes(self.material_name)
        handler.align_nodes()
        if self.vmt.shader not in self._handlers:
            print(f'Shader "{self.vmt.shader}" not currently supported by SourceIO')
        pass

    # def create_material(self, material_name=None, override=True):
    #     material_name = material_name[-63:]
    #     print(f'Creating material {repr(material_name)}, override:{override}')
    #     if bpy.data.materials.get(material_name) and not override:
    #         return 'EXISTS'
    #     else:
    #         bpy.data.materials.new(material_name)
    #     mat = bpy.data.materials.get(material_name)
    #     if mat is None:
    #         return
    #     if mat.get('source1_loaded'):
    #         return 'LOADED'
    #
    #     mat.use_nodes = True
    #     nodes = mat.node_tree.nodes
    #     diff = nodes.get('Principled BSDF', None)
    #     if diff:
    #         nodes.remove(diff)
    #     out = nodes.get('ShaderNodeOutputMaterial', None)
    #     if not out:
    #         out = nodes.get('Material Output', None)
    #     if not out:
    #         out = nodes.new('ShaderNodeOutputMaterial')
    #     out.location = (385.0, 146.0)
    #     bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    #     bsdf.location = (45.0, 146.0)
    #     mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs['Surface'])
    #
    #     if self.textures.get('$basetexture', False):
    #         basetexture_node = nodes.new('ShaderNodeTexImage')
    #         basetexture_node.image = self.textures.get('$basetexture')
    #         basetexture_node.name = '$basetexture'
    #         basetexture_node.location = (-295.0, 146.0)
    #         mat.node_tree.links.new(basetexture_node.outputs["Color"], bsdf.inputs['Base Color'])
    #
    #         alpha_output = basetexture_node.outputs["Alpha"]
    #         if int(self.material_data.get('$basemapalphaphongmask', '0')) == 1:
    #             mat.node_tree.links.new(alpha_output, bsdf.inputs['Specular'])
    #         elif int(self.material_data.get('$basealphaenvmapmask', '0')) == 1:
    #             mat.node_tree.links.new(alpha_output, bsdf.inputs['Roughness'])
    #         elif int(self.material_data.get('$selfillum', '0')) == 1:
    #             mat.node_tree.links.new(basetexture_node.outputs["Color"], bsdf.inputs['Emission'])
    #             mat.node_tree.links.new(alpha_output, bsdf.inputs['Emission Strength'])
    #         elif int(self.material_data.get('$alphatest', '0')) == 1:
    #             mat.node_tree.links.new(alpha_output, bsdf.inputs['Alpha'])
    #         elif int(self.material_data.get('$translucent', '0')) == 1:
    #             mat.node_tree.links.new(alpha_output, bsdf.inputs['Alpha'])
    #     if self.textures.get('$bumpmap', False):
    #         bumpmap_texture = nodes.new('ShaderNodeTexImage')
    #         bumpmap_texture.name = '$bumpmap'
    #         image = self.textures.get('$bumpmap')
    #         if int(self.material_data.get('$ssbump', '0')):
    #             image = self.convert_ssbump(image)
    #         bumpmap_texture.image = image
    #         bumpmap_texture.location = (-635.0, 146.0)
    #         bumpmap_texture.image.colorspace_settings.is_data = True
    #         bumpmap_texture.image.colorspace_settings.name = 'Non-Color'
    #
    #         if int(self.material_data.get('$normalmapalphaenvmapmask', '0')) == 1:
    #             mat.node_tree.links.new(bumpmap_texture.outputs["Alpha"], bsdf.inputs['Specular'])
    #
    #         normal = nodes.new("ShaderNodeNormalMap")
    #         normal.location = (-295.0, -125.0)
    #         mat.node_tree.links.new(bumpmap_texture.outputs["Color"], normal.inputs['Color'])
    #         mat.node_tree.links.new(normal.outputs["Normal"], bsdf.inputs['Normal'])
    #     if self.textures.get('$phongexponenttexture', False):
    #         tex = nodes.new('ShaderNodeTexImage')
    #         tex.name = '$phongexponenttexture'
    #         tex.image = self.textures.get('$phongexponenttexture')
    #         tex.location = (-200, 0)
    #
    #     mat.blend_method = 'HASHED'
    #     mat.shadow_method = 'HASHED'
    #
    #     mat['source1_loaded'] = True
