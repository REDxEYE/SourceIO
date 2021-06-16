import bpy
from bpy.types import Node
from .base_node import SourceIOModelTreeNode


class SourceIOModelNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOModelNode'
    bl_label = "Model output"
    model_name_prop: bpy.props.StringProperty(name="Model name")

    def init(self, context):
        self.inputs.new('SourceIOObjectSocket', "Objects").link_limit = 4096
        self.inputs.new('SourceIOBodygroupSocket', "Bodygroups").link_limit = 4096
        self.inputs.new('SourceIOSkinSocket', "Skin")

    def draw_buttons(self, context, layout):
        layout.prop(self, 'model_name_prop')
        layout.operator("SourceIO.evaluate_nodetree")

    @property
    def model_name(self):
        return self.model_name_prop
