from typing import List

import bpy
from bpy.types import Node, NodeTree, Operator

from . import nodes


class SourceIO_OP_EvaluateNodeTree(Operator):
    bl_idname = "sourceio.evaluate_nodetree"
    bl_label = "Evaluate tree"
    tmp_file: bpy.types.Text

    def execute(self, context: bpy.types.Context):
        if not bpy.data.texts.get('qc', False):
            self.tmp_file = bpy.data.texts.new('qc')
        else:
            self.tmp_file = bpy.data.texts['qc']
        all_nodes = context.space_data.node_tree.nodes
        outputs = []  # type:List[Node]
        for node in all_nodes:  # type: Node
            if node.bl_idname == "SourceIOModelNode":
                outputs.append(node)
        for output in outputs:  # type:nodes.SourceIOModelNode
            self.traverse_tree(output)
        return {'FINISHED'}

    def traverse_tree(self, start_node: nodes.SourceIOModelNode):
        start_node.write(self.tmp_file)
        objects = start_node.inputs['Objects']
        bodygroups = start_node.inputs['Bodygroups']
        skins = start_node.inputs['Skin']
        if objects.is_linked:
            for link in objects.links:
                object_node: nodes.SourceIOObjectNode = link.from_node
                object_node.write(self.tmp_file)

        if bodygroups.is_linked:
            for link in bodygroups.links:
                bodygroup_node: nodes.SourceIOBodygroupNode = link.from_node
                bodygroup_node.write(self.tmp_file)
        if skins.is_linked:
            skin_node = skins.links[0].from_node  # type: nodes.SourceIOSkinNode
            self.tmp_file.write(str(skin_node.get_value()))
            self.tmp_file.write('\n')


class SourceIO_NT_ModelTree(NodeTree):
    bl_idname = 'sourceio.model_definition'
    bl_label = "SourceIO model definition"
    bl_icon = 'NODETREE'

    def update(self, ):
        for node in self.nodes:
            node.update()
        for link in self.links:  # type:bpy.types.NodeLink
            if link.from_socket.bl_idname != link.to_socket.bl_idname:
                self.links.remove(link)
        self.check_link_duplicates()

    def check_link_duplicates(self):
        to_remove = []
        for link in self.links:
            for link2 in self.links:
                if link == link2 or link in to_remove:
                    continue
                if link.from_node == link2.from_node and link.to_node == link2.to_node:
                    to_remove.append(link2)
                    break
        for link in to_remove:
            self.links.remove(link)
