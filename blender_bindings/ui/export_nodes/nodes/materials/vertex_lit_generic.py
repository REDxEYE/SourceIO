from typing import List, Union

import bpy
from bpy.types import Node

from ..base_node import SourceIOModelTreeNode


class SourceIOVertexLitGenericNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOVertexLitGenericNode'
    bl_label = "Vertex lit generic Shader"

    mat: bpy.props.PointerProperty(type=bpy.types.Material, name="Material")
    base_texture: bpy.props.PointerProperty(type=bpy.types.Image, name="Base texture")
    normal_texture: bpy.props.PointerProperty(type=bpy.types.Image, name="Normal texture")

    def init(self, context):
        self.outputs.new('SourceIOMaterialSocket', "Material")

        self.inputs.new('SourceIOTextureSocket', "Base texture")
        self.inputs.new('SourceIOTextureSocket', "Normal texture")
        self.inputs.new('NodeSocketFloat', "Phong boost")
        self.inputs.new('NodeSocketFloat', "Phong exponent")

    def draw_buttons(self, context, layout):
        layout.prop(self, "mat")
