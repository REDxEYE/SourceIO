import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem

from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_1

from . import sockets
from .build_graph import classes as build_classes
from .flex_graph import (
    classes as flex_classes, register_flex_sync, unregister_flex_sync)
from .model_tree_nodes import SourceIO_NT_ModelTree
from .nodes.material_nodes import classes as material_classes
from .nodes.prop_nodes import classes as prop_classes
from .nodes.sfm_rig_nodes import (
    classes as sfm_rig_classes, sockets as sfm_rig_sockets)
from .nodes.textures import texture_ops
from .nodes.textures.texture_preview import (
    SourceIOTexturePreviewNode, SourceIO_OP_EvaluatePreview)


class SourceIONodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'sourceio.model_definition'


node_categories = [
    SourceIONodeCategory('Output', "Output", items=[
        NodeItem("SourceIOPropModelNode"),
    ]),
    SourceIONodeCategory('Geometry', "Geometry", items=[
        NodeItem("SourceIOPropGeometryNode"),
        NodeItem("SourceIOModelEntryNode"),
        NodeItem("SourceIOPropBlankNode"),
        NodeItem("SourceIOPropBodygroupNode"),
        NodeItem("SourceIOPropLodNode"),
        NodeItem("SourceIOPropCollisionNode"),
    ]),
    SourceIONodeCategory('FlexRules', "Flex rules", items=[
        NodeItem("SourceIOFlexGroupNode"),
        NodeItem("SourceIOFlexRulesNode"),
        NodeItem("SourceIOFlexControllerNode"),
        NodeItem("SourceIOVirtualControllerNode"),
        NodeItem("SourceIOFlexShapeRuleNode"),
        NodeItem("SourceIOFlexStereoRuleNode"),
        NodeItem("SourceIOFlexCorrectiveRuleNode"),
        NodeItem("SourceIOFlexConstantNode"),
        NodeItem("SourceIOFlexMathNode"),
        NodeItem("SourceIOFlexClampNode"),
        NodeItem("SourceIOFlexRemapNode"),
        NodeItem("SourceIOFlexInvertNode"),
        NodeItem("SourceIOFlexCrossfadeNode"),
        NodeItem("SourceIOFlexRawExpressionNode"),
    ]),
    SourceIONodeCategory('ModelSettings', "Model settings", items=[
        NodeItem("SourceIOPropSkinFamilyNode"),
        NodeItem("SourceIOPropAttachmentNode"),
        NodeItem("SourceIOPropHitboxNode"),
        NodeItem("SourceIOPropIdleNode"),
        NodeItem("SourceIOPropRawQcNode"),
        NodeItem("SourceIOBoneRenameNode"),
    ]),
    SourceIONodeCategory('SfmRig', "SFM rig", items=[
        NodeItem("SourceIOSfmIkRigNode"),
        NodeItem("SourceIOSfmIkChainNode"),
        NodeItem("SourceIOSfmFkChainNode"),
        NodeItem("SourceIOSfmFingerNode"),
        NodeItem("SourceIOSfmConstraintNode"),
    ]),
    SourceIONodeCategory('Materials', "Materials", items=[
        NodeItem("SourceIOVertexLitGenericVMTNode"),
        NodeItem("SourceIOUnlitGenericVMTNode"),
        NodeItem("SourceIOEyeRefractVMTNode"),
        NodeItem("SourceIOEyesVMTNode"),
        NodeItem("SourceIORefractVMTNode"),
        NodeItem("SourceIOTeethVMTNode"),
        NodeItem("SourceIOHairVMTNode"),
        NodeItem("SourceIOSkinVMTNode"),
        NodeItem("SourceIOGenericVMTNode"),
        NodeItem("SourceIOVTFOutputNode"),
    ]),
    SourceIONodeCategory('Textures', "Textures", items=[
        NodeItem("SourceIOImageSourceNode"),
        NodeItem("SourceIOExternalTextureNode"),
        NodeItem("SourceIOTextureInvertChannelsNode"),
        NodeItem("SourceIOTextureSplitChannelsNode"),
        NodeItem("SourceIOTextureCombineChannelsNode"),
        NodeItem("SourceIOTextureBrightnessContrastNode"),
        NodeItem("SourceIOTextureMathNode"),
        NodeItem("SourceIOChannelRemapNode"),
        NodeItem("SourceIONormalOpsNode"),
        NodeItem("SourceIOHeightToNormalNode"),
        NodeItem("SourceIONormalBlendNode"),
        NodeItem("SourceIOColorSpaceNode"),
        NodeItem("SourceIOPBRToPhongNode"),
        NodeItem("SourceIOTexturePreviewNode"),
    ]),
]

texture_classes = (
    texture_ops.SourceIOTextureInvertChannelsNode,
    texture_ops.SourceIOTextureSplitChannelsNode,
    texture_ops.SourceIOTextureCombineChannelsNode,
    texture_ops.SourceIOTextureBrightnessContrastNode,
    texture_ops.SourceIOTextureMathNode,
    texture_ops.SourceIOChannelRemapNode,
    texture_ops.SourceIONormalOpsNode,
    texture_ops.SourceIOHeightToNormalNode,
    texture_ops.SourceIONormalBlendNode,
    texture_ops.SourceIOPBRToPhongNode,
    texture_ops.SourceIOColorSpaceNode,
    SourceIOTexturePreviewNode,
)

drop_classes = ()
if is_blender_4_1():
    from .file_drop import (
        NODE_FH_SourceIOExportAssets, SOURCEIO_OT_DropExportAsset)
    drop_classes = (SOURCEIO_OT_DropExportAsset, NODE_FH_SourceIOExportAssets)


classes = (
    SourceIO_NT_ModelTree,
    sockets.SourceIOObjectSocket,
    sockets.SourceIOBodygroupSocket,
    sockets.SourceIOMaterialSocket,
    sockets.SourceIOSkinGroupSocket,
    sockets.SourceIOTextureSocket,
    sockets.SourceIOTextureVtfSocket,
    sockets.SourceIOTextureChannelSocket,
    *sfm_rig_sockets,  # before the model output node that uses them
    *prop_classes,
    *sfm_rig_classes[len(sfm_rig_sockets):],
    *flex_classes,
    *material_classes,
    *texture_classes,
    *drop_classes,
    *build_classes,
    SourceIO_OP_EvaluatePreview,
)


def register_nodes():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    nodeitems_utils.register_node_categories('SourceIO_Nodes', node_categories)
    register_flex_sync()


def unregister_nodes():
    nodeitems_utils.unregister_node_categories('SourceIO_Nodes')
    unregister_flex_sync()

    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
