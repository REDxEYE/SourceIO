from typing import List, Union

import bpy
from bpy.types import Node

from ..sockets import SourceIOMaterialSocket, SourceIOSkinGroupSocket
from .base_node import SourceIOModelTreeNode
from .input_material_node import SourceIOMaterialNode


class SourceIOSkinGroupProto:
    def __init__(self):
        self.name: str = ""
        self.materials: List[bpy.types.Material] = []

    def __str__(self):
        tmp = f'Skingroup\n'
        for obj in self.materials:
            tmp += "\t" + obj.name
        tmp += '\n'
        return tmp


class SourceIOSkingroupNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOSkingroupNode'
    bl_label = "Skingroup"

    def init(self, context):
        self.inputs.new('SourceIOMaterialSocket', "Material")
        self.outputs.new('SourceIOSkinGroupSocket', "Skingroup")

    def update(self, ):
        unused_count = 0
        total_inputs = 0
        for o in self.inputs:  # type:SourceIOMaterialSocket
            if (not o.is_linked and o.material is None) and o.bl_idname == "SourceIOMaterialSocket":
                unused_count += 1
            else:
                total_inputs += 1
        while unused_count >= 1:
            self.inputs.remove(self.inputs[-1])
            unused_count -= 1
        if not unused_count == 0 and total_inputs <= 32:
            self.inputs.new("SourceIOMaterialSocket", "Material")

    def draw_buttons(self, context, layout):
        self.update()

    def get_value(self):
        proto = SourceIOSkinGroupProto()
        for input_socket in self.inputs:  # type:SourceIOMaterialSocket
            if input_socket.is_linked:
                obj_node: SourceIOMaterialNode = input_socket.links[0].from_node
                proto.materials.append(obj_node.get_value().mat)
            else:
                if input_socket.material:
                    proto.materials.append(input_socket.material)
        return proto


class SourceIOSkinProto:
    def __init__(self):
        self.skins: List[SourceIOSkinGroupProto] = []

    def __str__(self):
        tmp = "Skins:\n"
        for skin in self.skins:
            tmp += '\t' + str(skin) + '\n'
        return tmp


class SourceIOSkinNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOSkinNode'
    bl_label = "Skin"

    def init(self, context):
        self.inputs.new('SourceIOSkinGroupSocket', "Skingroup")

        self.outputs.new('SourceIOSkinSocket', "Skin")

    def update(self, ):
        unused_count = 0
        for o in self.inputs:
            if (not o.is_linked) and o.bl_idname == "SourceIOSkinGroupSocket":
                unused_count += 1
        if unused_count > 1:
            for _ in range(unused_count - 1):
                self.inputs.remove(self.inputs[-1])
        if unused_count == 0:
            self.inputs.new("SourceIOSkinGroupSocket", "Skingroup")

    def draw_buttons(self, context, layout):
        self.update()

    def get_value(self):
        proto = SourceIOSkinProto()
        for input_socket in self.inputs:  # type:SourceIOSkinGroupSocket
            if input_socket.is_linked:
                obj_node: SourceIOMaterialNode = input_socket.links[0].from_node
                proto.skins.append(obj_node.get_value())
        return proto
