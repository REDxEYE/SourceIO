from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, FloatProperty

from ..utils.utils import get_new_unique_collection
from ..source2.dmx.camera_loader import load_camera
from ..source2.vmdl.loader import put_into_collections
from ..source2.vmdl.loader import ValveCompiledModelLoader
from ..source2.vwrld.loader import ValveCompiledWorldLoader
from ..source2.vtex.loader import ValveCompiledTextureLoader
from ..source2.vmat.loader import ValveCompiledMaterialLoader
from ...library.shared.content_providers.content_manager import ContentManager
from ...library.utils.math_utilities import SOURCE2_HAMMER_UNIT_TO_METERS


class SOURCEIO_OT_VMDLImport(bpy.types.Operator):
    """Load Source2 VMDL"""
    bl_idname = "sourceio.vmdl"
    bl_label = "Import Source2 VMDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    invert_uv: BoolProperty(name="invert UV?", default=True)
    scale: FloatProperty(name="World scale", default=SOURCE2_HAMMER_UNIT_TO_METERS, precision=6)
    import_anim: BoolProperty(name="Import animations", default=False)
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    filter_glob: StringProperty(default="*.vmdl_c", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        ContentManager().scan_for_content(directory)
        for n, file in enumerate(self.files):
            print(f"Loading {n + 1}/{len(self.files)}")
            model = ValveCompiledModelLoader(str(directory / file.name), self.scale)
            model.load_mesh(self.invert_uv)
            model.load_attachments()
            master_collection = get_new_unique_collection(model.name, bpy.context.scene.collection)
            put_into_collections(model.container, Path(model.name).stem, master_collection, False)

            if self.import_anim:
                model.load_animations()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_VWRLDImport(bpy.types.Operator):
    """Load Source2 VWRLD"""
    bl_idname = "sourceio.vwrld"
    bl_label = "Import Source2 VWRLD file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vwrld_c", options={'HIDDEN'})

    invert_uv: BoolProperty(name="invert UV?", default=True)
    scale: FloatProperty(name="World scale", default=SOURCE2_HAMMER_UNIT_TO_METERS, precision=6)

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            print(f"Loading {n}/{len(self.files)}")
            ContentManager().scan_for_content((directory.parent / file.name).with_suffix('.vpk'))
            world = ValveCompiledWorldLoader(directory / file.name, invert_uv=self.invert_uv, scale=self.scale)
            world.load(file.name)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_VPK_VWRLDImport(bpy.types.Operator):
    """Load Source2 VWRLD"""
    bl_idname = "sourceio.vwrld_vpk"
    bl_label = "Import Source2 VWRLD file from VPK"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vpk", options={'HIDDEN'})

    invert_uv: BoolProperty(name="invert UV?", default=True)
    scale: FloatProperty(name="World scale", default=SOURCE2_HAMMER_UNIT_TO_METERS, precision=6)

    def execute(self, context):
        vpk_path = Path(self.filepath)
        assert vpk_path.is_file(), 'Not a file'

        ContentManager().scan_for_content(vpk_path.parent)
        ContentManager().scan_for_content(vpk_path)
        world_file = ContentManager().find_file(f'maps/{vpk_path.stem}/world.vwrld_c')
        assert world_file is not None, "Failed to find world file in selected VPK"
        world = ValveCompiledWorldLoader(world_file, invert_uv=self.invert_uv, scale=self.scale)
        world.load(vpk_path.stem)
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
    flip: BoolProperty(name="Flip texture", default=True)
    split_alpha: BoolProperty(name="Extract alpha texture", default=True)
    filter_glob: StringProperty(default="*.vmat_c", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        ContentManager().scan_for_content(directory)
        for n, file in enumerate(self.files):
            print(f"Loading {n + 1}/{len(self.files)}")
            material = ValveCompiledMaterialLoader(str(directory / file.name))
            material.load()
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
            texture = ValveCompiledTextureLoader(str(directory / file.name))
            texture.import_texture(Path(file.name).stem, self.flip)
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
