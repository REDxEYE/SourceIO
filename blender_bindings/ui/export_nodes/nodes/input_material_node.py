import bpy
from bpy.types import Node

from .base_node import SourceIOModelTreeNode


class SourceIOMaterialProto:
    def __init__(self):
        self.mat = None


class SourceIOMaterialNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOMaterialNode'
    bl_label = 'Material Node'
    mat: bpy.props.PointerProperty(type=bpy.types.Material, name="Material")

    def init(self, context):
        self.outputs.new('SourceIOMaterialSocket', "Material")

    def draw_buttons(self, context, layout):
        layout.prop(self, "mat")

    def get_value(self):
        obj = SourceIOMaterialProto()
        obj.mat = self.mat
        return obj
