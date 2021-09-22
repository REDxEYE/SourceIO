from bpy.types import NodeSocket
import bpy


class SourceIOBodygroupSocket(NodeSocket):
    bl_idname = 'SourceIOBodygroupSocket'
    bl_lable = 'Bodygroup socket'
    type = "BODYGROUP"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 0.5, 0.2, 0.2, 1.0
