import bpy
from bpy.props import CollectionProperty, IntProperty, StringProperty, PointerProperty, FloatProperty
from bpy.types import PropertyGroup

from SourceIO.blender_bindings.operators.flex_operators import SourceIO_PG_FlexController
from SourceIO.blender_bindings.operators.shared_operators import SOURCEIO_UL_MountedResource

class SourceIO_PG_props(PropertyGroup):
    TextureCachePath: StringProperty(name="TextureCachePath", subtype="FILE_PATH")
    use_bvlg: bpy.props.BoolProperty(
        name="Use BVLG",
        default=False
    )
    use_instances: bpy.props.BoolProperty(
        name="Use instances",
        default=True
    )
    import_materials: bpy.props.BoolProperty(
        name="Import materials",
        default=True
    )
    import_physics: bpy.props.BoolProperty(
        name="Import physics",
        default=False
    )
    replace_entity: bpy.props.BoolProperty(
        name="Replace entity",
        default=True
    )
    mounted_resources: CollectionProperty(type=SOURCEIO_UL_MountedResource)
    mounted_resources_index: IntProperty(default=0)

    lr_balance: FloatProperty(
        name='Left/Right Balance',
        description='Apply more pose on the left or right side',
        default=0.0,
        min=-1.0,
        max=1.0
    )

    flex_controller_index: IntProperty(options=set())


def register_props():
    bpy.utils.register_class(SourceIO_PG_props)
    bpy.types.Scene.sourceio_props = PointerProperty(type=SourceIO_PG_props)
    bpy.types.Object.flex_controllers = CollectionProperty(type=SourceIO_PG_FlexController)


def unregister_props():
    del bpy.types.Scene.sourceio_props
    bpy.utils.unregister_class(SourceIO_PG_props)
