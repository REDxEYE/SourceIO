from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, StringProperty)

from ...library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS


class SharedSettings:
    filepath: StringProperty(subtype="FILE_PATH")
    scale: FloatProperty(name="World scale", default=SOURCE1_HAMMER_UNIT_TO_METERS, precision=6)


class Source1SharedSettings:
    use_bvlg: BoolProperty(name="Use BlenderVertexLitGeneric shader", default=True, subtype='UNSIGNED')


class BSPSettings(SharedSettings):
    light_scale: FloatProperty(name="Light power scale", default=1, precision=6)
    load_props: BoolProperty(name="Load prop entities", default=True)
    load_lights: BoolProperty(name="Load light entities", default=True)
    load_decals: BoolProperty(name="Load decal entities", default=True)
    load_static_props: BoolProperty(name="Load static prop entities", default=True)
    load_triggers: BoolProperty(name="Load trigger entities", default=False)
    load_info: BoolProperty(name="Load info entities", default=False)
    load_logic: BoolProperty(name="Load logic entities", default=False)
    load_ropes: BoolProperty(name="Load rope entities", default=False)


class GoldSrcBspSettings(BSPSettings):
    import_textures: BoolProperty(name="Import materials", default=True, subtype='UNSIGNED')


class Source1BSPSettings(GoldSrcBspSettings, Source1SharedSettings):
    import_cubemaps: BoolProperty(name="Import cubemaps", default=False, subtype='UNSIGNED')


class MDLSettings(SharedSettings, Source1SharedSettings):
    write_qc: BoolProperty(name="Write QC", default=True, subtype='UNSIGNED')
    import_physics: BoolProperty(name="Import physics", default=False, subtype='UNSIGNED')
    load_refpose: BoolProperty(name="Load Ref pose", default=False, subtype='UNSIGNED')
    import_animations: BoolProperty(name="Load animations", default=False, subtype='UNSIGNED')
    unique_materials_names: BoolProperty(name="Unique material names", default=False, subtype='UNSIGNED')

    create_flex_drivers: BoolProperty(name="Create drivers for flexes", default=False, subtype='UNSIGNED')
    bodygroup_grouping: BoolProperty(name="Group meshes by bodygroup", default=True, subtype='UNSIGNED')
    import_textures: BoolProperty(name="Import materials", default=True, subtype='UNSIGNED')
