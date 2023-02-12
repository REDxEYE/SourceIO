import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem

from . import nodes, sockets
from .model_tree_nodes import *
from .nodes import materials as material_nodes

### Node Categories ###
# Node categories are a python system for automatically
# extending the Add menu, toolbar panels and search operator.
# For more examples see release/scripts/startup/nodeitems_builtins.py



# our own base class with an appropriate poll function,
# so the categories only show up in our own tree type


class SourceIONodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'sourceio.model_definition'


# all categories in a list
node_categories = [
    # identifier, label, items list
    SourceIONodeCategory('Output', "Output", items=[
        # our basic node
        NodeItem("SourceIOModelNode"),
    ]),
    SourceIONodeCategory("Inputs", "Inputs", items=[
        NodeItem("SourceIOObjectNode"),
        NodeItem("SourceIOMaterialNode"),
    ]),
    SourceIONodeCategory("Skins", "Skins", items=[
        NodeItem("SourceIOSkingroupNode"),
        NodeItem("SourceIOSkinNode"),
    ]),
    SourceIONodeCategory("Bodygroups", "Bodygroups", items=[
        NodeItem("SourceIOBodygroupNode")
    ]),
    SourceIONodeCategory("Materials", "Materials", items=[
        NodeItem("SourceIOVertexLitGenericNode")
    ])
]

classes = (
    SourceIO_NT_ModelTree,

    nodes.SourceIOObjectNode,
    nodes.SourceIOModelNode,
    nodes.SourceIOMaterialNode,

    nodes.SourceIOBodygroupNode,
    nodes.SourceIOSkinNode,
    nodes.SourceIOSkingroupNode,

    material_nodes.SourceIOVertexLitGenericNode,

    sockets.SourceIOObjectSocket,
    sockets.SourceIOBodygroupSocket,
    sockets.SourceIOMaterialSocket,
    sockets.SourceIOSkinSocket,
    sockets.SourceIOSkinGroupSocket,
    sockets.SourceIOTextureSocket,

    SourceIO_OP_EvaluateNodeTree
)


def register_nodes():
    print('Registered nodes')
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    nodeitems_utils.register_node_categories('SourceIO_Nodes', node_categories)


def unregister_nodes():
    nodeitems_utils.unregister_node_categories('SourceIO_Nodes')

    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
