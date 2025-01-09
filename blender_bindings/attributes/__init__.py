import bpy
from bpy.props import CollectionProperty, IntProperty, StringProperty

from SourceIO.blender_bindings.operators.flex_operators import SourceIO_PG_FlexController
from SourceIO.blender_bindings.operators.shared_operators import SOURCEIO_UL_MountedResource


def register_props():
    bpy.types.Scene.TextureCachePath = StringProperty(name="TextureCachePath", subtype="FILE_PATH")

    bpy.types.Scene.use_bvlg = bpy.props.BoolProperty(
        name="Use BVLG",
        default=True
    )
    bpy.types.Scene.use_instances = bpy.props.BoolProperty(
        name="Use instances",
        default=True
    )
    bpy.types.Scene.import_materials = bpy.props.BoolProperty(
        name="Import materials",
        default=True
    )
    bpy.types.Scene.import_physics = bpy.props.BoolProperty(
        name="Import physics",
        default=False
    )
    bpy.types.Scene.replace_entity = bpy.props.BoolProperty(
        name="Replace entity",
        default=True
    )
    bpy.types.Mesh.flex_controllers = CollectionProperty(type=SourceIO_PG_FlexController)
    bpy.types.Mesh.flex_selected_index = IntProperty(default=0)

    bpy.types.Scene.mounted_resources = CollectionProperty(type=SOURCEIO_UL_MountedResource)
    bpy.types.Scene.mounted_resources_index = IntProperty(default=0)


def unregister_props():
    del bpy.types.Scene.TextureCachePath
    del bpy.types.Mesh.flex_controllers
    del bpy.types.Mesh.flex_selected_index
    del bpy.types.Scene.use_bvlg
    del bpy.types.Scene.use_instances
    del bpy.types.Scene.replace_entity
    del bpy.types.Scene.mounted_resources
    del bpy.types.Scene.mounted_resources_index
    del bpy.types.Scene.import_materials
