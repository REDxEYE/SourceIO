from pathlib import Path

import bpy
from bpy.props import (BoolProperty, CollectionProperty, FloatProperty,
                       IntProperty, StringProperty)

from ..utils.resource_utils import serialize_mounted_content, deserialize_mounted_content
from ...library.shared.content_providers.content_manager import ContentManager
from ...library.shared.content_providers.vpk_provider import VPKContentProvider
from ...library.source2 import (CompiledMaterialResource,
                                CompiledModelResource, CompiledTextureResource)
from ...library.source2.resource_types.compiled_world_resource import \
    CompiledMapResource
from ...library.utils import FileBuffer
from ...library.utils.math_utilities import SOURCE2_HAMMER_UNIT_TO_METERS
from ..source2.dmx.camera_loader import load_camera
from ..source2.vmat_loader import load_material
from ..source2.vmdl_loader import load_model, put_into_collections
from ..source2.vtex_loader import import_texture
from ..source2.vwrld.loader import load_map
from ..utils.utils import get_new_unique_collection
from ...library.utils.path_utilities import backwalk_file_resolver


class SOURCEIO_OT_VMDLImport(bpy.types.Operator):
    """Load Source2 VMDL"""
    bl_idname = "sourceio.vmdl"
    bl_label = "Import Source2 VMDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    # invert_uv: BoolProperty(name="Invert UV", default=True)
    import_physics: BoolProperty(name="Import physics", default=False)
    import_attachments: BoolProperty(name="Import attachments", default=False)
    lod_mask: IntProperty(name="Lod mask", default=0xFFFF, subtype="UNSIGNED")
    scale: FloatProperty(name="World scale", default=SOURCE2_HAMMER_UNIT_TO_METERS, precision=6)
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    filter_glob: StringProperty(default="*.vmdl_c", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()

        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(directory)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)

        for n, file in enumerate(self.files):
            print(f"Loading {n + 1}/{len(self.files)}")
            with FileBuffer(directory / file.name) as f:
                model_resource = CompiledModelResource.from_buffer(f, directory / file.name)
                container = load_model(model_resource, self.scale, self.lod_mask,
                                       self.import_physics, self.import_attachments)

            master_collection = get_new_unique_collection(model_resource.name, bpy.context.scene.collection)
            put_into_collections(container, Path(model_resource.name).stem, master_collection, False)

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_VMAPImport(bpy.types.Operator):
    """Load Source2 VWRLD"""
    bl_idname = "sourceio.vmap"
    bl_label = "Import Source2 VMAP file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vmap_c", options={'HIDDEN'})
    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    # invert_uv: BoolProperty(name="invert UV?", default=True)
    scale: FloatProperty(name="World scale", default=SOURCE2_HAMMER_UNIT_TO_METERS, precision=6)

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            print(f"Loading {n}/{len(self.files)}")
            cm = ContentManager()
            content_manager = ContentManager()
            if self.discover_resources:
                cm.scan_for_content(directory.parent)
                serialize_mounted_content(content_manager)
            else:
                deserialize_mounted_content(content_manager)
            cm.register_content_provider(Path(file.name).stem + ".vpk",
                                         VPKContentProvider(directory / f"{Path(file.name).stem}.vpk"))
            with FileBuffer(directory / file.name) as buffer:
                model = CompiledMapResource.from_buffer(buffer, Path(file.name))
                load_map(model, ContentManager(), self.scale)
            serialize_mounted_content(cm)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_VPK_VMAPImport(bpy.types.Operator):
    """Load Source2 VWRLD"""
    bl_idname = "sourceio.vmap_vpk"
    bl_label = "Import Source2 VMAP file from VPK"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vpk", options={'HIDDEN'})
    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    # invert_uv: BoolProperty(name="invert UV?", default=True)
    scale: FloatProperty(name="World scale", default=SOURCE2_HAMMER_UNIT_TO_METERS, precision=6)

    def execute(self, context):
        vpk_path = Path(self.filepath)
        assert vpk_path.is_file(), 'Not a file'
        cm = ContentManager()
        content_manager = ContentManager()
        if self.discover_resources:
            cm.scan_for_content(vpk_path.parent)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)
        cm.register_content_provider(vpk_path.name, VPKContentProvider(vpk_path))
        maps_folder = backwalk_file_resolver(vpk_path, "maps")
        if maps_folder is not None:
            relative_path = vpk_path.relative_to(maps_folder.parent)
            map_buffer = ContentManager().find_file(relative_path.with_suffix(".vmap_c"))
        else:
            map_buffer = ContentManager().find_file(f'maps/{vpk_path.stem}.vmap_c')
        assert map_buffer is not None, "Failed to find world file in selected VPK"

        model = CompiledMapResource.from_buffer(map_buffer, vpk_path)
        load_map(model, ContentManager(), self.scale)
        serialize_mounted_content(cm)

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_VMATImport(bpy.types.Operator):
    """Load Source2 material"""
    bl_idname = "sourceio.vmat"
    bl_label = "Import Source2 VMDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    flip: BoolProperty(name="Flip texture", default=True)
    split_alpha: BoolProperty(name="Extract alpha texture", default=True)
    filter_glob: StringProperty(default="*.vmat_c", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(directory)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)
        for n, file in enumerate(self.files):
            print(f"Loading {n + 1}/{len(self.files)}")
            with FileBuffer(directory / file.name) as f:
                material_resource = CompiledMaterialResource.from_buffer(f, directory / file.name)
                load_material(material_resource, Path(file.name))
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_VTEXImport(bpy.types.Operator):
    """Load Source Engine VTF texture"""
    bl_idname = "sourceio.vtex"
    bl_label = "Import VTEX"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH', )
    flip: BoolProperty(name="Flip texture", default=True)
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vtex_c", options={'HIDDEN'})

    def execute(self, context):
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            with FileBuffer(directory / file.name) as f:
                texture_resource = CompiledTextureResource.from_buffer(f, directory / file.name)
                import_texture(texture_resource, Path(file.name), self.flip)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_DMXCameraImport(bpy.types.Operator):
    """Load Valve DMX camera data"""
    bl_idname = "sourceio.dmx_camera"
    bl_label = "Import DMX camera"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH', )
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.dmx", options={'HIDDEN'})

    def execute(self, context):
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            load_camera(directory / file.name)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}
