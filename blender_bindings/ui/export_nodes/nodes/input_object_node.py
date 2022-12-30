from typing import TextIO

import bpy
from bpy.types import Node

from .base_node import SourceIOModelTreeNode


class SourceIOBlankObjectProto:
    is_blank = True

    class BlankObject:
        name = "BLANK"

    @property
    def obj(self):
        return SourceIOBlankObjectProto.BlankObject


class SourceIOObjectProto:
    is_blank = False

    def __init__(self):
        self.obj = None


class InvalidObject(Exception):
    pass


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

    def write(self, buffer: TextIO):
        if self.blank:
            raise InvalidObject('Blank cannot be used in model keyword')
        buffer.write(f'$model model "{self.get_value().obj.name}"\n')
