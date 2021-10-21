import bpy
from bpy.types import Node
from typing import List, Union

from .base_node import SourceIOModelTreeNode
from .input_object_node import SourceIOObjectNode


class SourceIOBodygroupProto:
    def __init__(self):
        self.name: str = ""
        self.objects: List[Union[bpy.types.Object, List[bpy.types.Object], None]] = []

    def __str__(self):
        tmp = f'"{self.name}"\n'
        for proto in self.objects:
            if type(proto) is list:
                tmp += "\t"
                for o in proto:
                    tmp += o.obj.name + "+"
                tmp += '\n'
            else:
                tmp += "\t" + proto.obj.name + '\n'
        return tmp


class SourceIOBodygroupNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOBodygroupNode'
    bl_label = "Bodygroup"
    bodygroup_name: bpy.props.StringProperty(name="Bodygroup name")

    def init(self, context):
        ob = self.inputs.new('SourceIOObjectSocket', "Objects")
        ob.link_limit = 4096

        self.outputs.new('SourceIOBodygroupSocket', "bodygroup")

    def update(self, ):
        unused = []
        for o in self.inputs:
            if (not o.is_linked) and o.bl_idname == "SourceIOObjectSocket":
                unused.append(o)
        if unused:
            for u in unused:
                self.inputs.remove(unused)
        if not unused:
            ob = self.inputs.new("SourceIOObjectSocket", "Objects")
            ob.link_limit = 4096

    def draw_buttons(self, context, layout):
        self.update()
        layout.prop(self, 'bodygroup_name')

    def draw_label(self):
        return 'Bodygroup: {}'.format(self.bodygroup_name)

    def get_value(self):
        proto = SourceIOBodygroupProto()
        proto.name = self.bodygroup_name
        for input_socket in self.inputs:
            if input_socket.is_linked:
                if len(input_socket.links) > 1:
                    merge: List[bpy.types.Object] = []
                    for link in input_socket.links:
                        obj_node: SourceIOObjectNode = link.from_node
                        merge.append(obj_node.get_value())
                    proto.objects.append(merge)
                else:
                    obj_node: SourceIOObjectNode = input_socket.links[0].from_node
                    proto.objects.append(obj_node.get_value())
        return proto
