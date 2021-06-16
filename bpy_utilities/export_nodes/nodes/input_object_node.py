from bpy.types import Node
import bpy

from .base_node import SourceIOModelTreeNode


class SourceIOBlankObjectProto:
    class BlankObject:
        name = "BLANK"

    @property
    def obj(self):
        return SourceIOBlankObjectProto.BlankObject


class SourceIOObjectProto:
    def __init__(self):
        self.obj = None


class SourceIOObjectNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOObjectNode'
    bl_label = 'Object Node'
    obj: bpy.props.PointerProperty(type=bpy.types.Object, name="Mesh object")
    blank: bpy.props.BoolProperty(name="Blank object")

    def init(self, context):
        self.outputs.new('SourceIOObjectSocket', "Object")

    def draw_buttons(self, context, layout):
        layout.prop(self, "blank")
        if not self.blank:
            layout.prop(self, "obj")

    def get_value(self):
        if self.blank:
            return SourceIOBlankObjectProto()
        else:
            obj = SourceIOObjectProto()
            obj.obj = self.obj
            return obj
