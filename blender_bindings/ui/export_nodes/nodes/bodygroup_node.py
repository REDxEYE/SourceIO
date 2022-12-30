from typing import List, TextIO, Union

import bpy
from bpy.types import Node

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
        self.outputs.new('SourceIOBodygroupSocket', "bodygroup")

    def update(self, ):
        unused_count = 0
        total_inputs = 0
        for o in self.inputs:
            if (not o.is_linked) and o.bl_idname == "SourceIOObjectSocket":
                unused_count += 1
            else:
                total_inputs += 1
        while unused_count >= 1:
            self.inputs.remove(self.inputs[-1])
            unused_count -= 1
        if not unused_count == 0 and total_inputs <= 32:
            ob = self.inputs.new("SourceIOObjectSocket", "Objects")

    def draw_buttons(self, context, layout):
        self.update()
        layout.prop(self, 'bodygroup_name')

    def draw_label(self):
        return 'Bodygroup: {}'.format(self.bodygroup_name)

    def _get_inputs(self):
        for input_socket in self.inputs:
            if input_socket.is_linked:
                if len(input_socket.links) > 1:
                    merge: List[bpy.types.Object] = []
                    for link in input_socket.links:
                        obj_node: SourceIOObjectNode = link.from_node
                        merge.append(obj_node.get_value())
                    raise NotImplementedError('TODO: Implement "Mesh Merge" node')
                else:
                    obj_node: SourceIOObjectNode = input_socket.links[0].from_node
                    yield obj_node.get_value()

    def write(self, buffer: TextIO):
        buffer.write(f'$bodygroup "{self.bodygroup_name}"{{\n')
        for object in self._get_inputs():
            if object.is_blank:
                buffer.write('\tblank\n')
            else:
                buffer.write(f'\tstudio "{object.obj.name}"\n')
        buffer.write('}\n')
