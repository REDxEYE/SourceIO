import bpy
from bpy.types import NodeSocket


class SourceIOSkinSocket(NodeSocket):
    bl_idname = 'SourceIOSkinSocket'
    bl_lable = 'Skin socket'
    type = "SKIN"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 0.4, 0.8, 0.2, 1.0

class SourceIOSkinGroupSocket(NodeSocket):
    bl_idname = 'SourceIOSkinGroupSocket'
    bl_lable = 'Skin group socket'
    type = "SKINGROUP"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 1, 0.8, 0.2, 1.0
