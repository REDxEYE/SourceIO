from typing import Union, Optional
from pathlib import Path

from .utilities.singleton import SingletonMeta
from .source_shared.vpk.vpk_file import open_vpk, VPKFile

import bpy
from bpy.props import (StringProperty,
                       BoolProperty,
                       CollectionProperty,
                       EnumProperty,
                       FloatProperty,
                       CollectionProperty,
                       IntProperty
                       )


class VPKFileHandle(metaclass=SingletonMeta):
    def __init__(self):
        self.current_file = None
        self.vpk_archive: Optional[VPKFile] = None
        self._current_dir: Optional[Path] = None
        self.vpk_browser_link: Optional[VPKBrowser] = None

    def dir_in(self, dir_name):
        if self._current_dir is None:
            self._current_dir = Path(dir_name)
        else:
            if dir_name == '..':
                return self.dir_out()
            elif Path(dir_name).suffix:
                return
            self._current_dir = Path(self._current_dir / dir_name)

    def dir_root(self):
        self._current_dir = None
        if self.vpk_browser_link is not None:
            self.vpk_browser_link.selected_index = -2

    def dir_out(self):
        if self._current_dir is not None:
            if len(self._current_dir.parts) == 1:
                self._current_dir = None
                self.vpk_browser_link.selected_index = -2
                return
            self._current_dir = self._current_dir.parent
        self.vpk_browser_link.selected_index = -2

    def get_current_path(self):
        return '/' + self._current_dir.as_posix() if self._current_dir is not None else '/'

    def current_dir(self):
        yield from self.vpk_archive.files_in_path(self._current_dir)

    def open_new(self, filepath):
        self.current_file = Path(filepath)
        self.vpk_archive = open_vpk(self.current_file)
        self.vpk_archive.read()


class VPKBrowserLoader(bpy.types.Operator):
    """Import whole filearchives directory."""
    bl_idname = "import_scene.vpk"
    bl_label = 'Browse VPK files'

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.vpk", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        # TODO: Validate filepath
        VPKFileHandle().open_new(self.filepath)
        VPKFileHandle().dir_root()
        VPKBrowser.bl_label = Path(self.filepath).name
        bpy.ops.ui.vpk_browser('INVOKE_DEFAULT')
        return {'FINISHED'}


class VPKEntry(bpy.types.PropertyGroup):
    name: StringProperty()
    selected: BoolProperty(name="")


class VPKButtonUp(bpy.types.Operator):
    bl_idname = "ui.vpk_browser_up"
    bl_label = "Up"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        VPKFileHandle().dir_out()
        return {'FINISHED'}


class VPKBrowser(bpy.types.Operator):
    bl_idname = "ui.vpk_browser"
    bl_label = "VPK-browser"
    bl_options = {'INTERNAL'}

    current_dir: CollectionProperty(type=VPKEntry)
    selected_index: IntProperty(default=0)  # -1 - No-op, -2 - update view
    cur_path: StringProperty(name='Cur')

    def invoke(self, context, event):
        VPKFileHandle().vpk_browser_link = self
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        box = self.layout.box()
        row = box.row()
        row.label(text=VPKFileHandle().get_current_path())
        row.operator(VPKButtonUp.bl_idname, icon='LOOP_BACK', text='')
        self.layout.separator()
        if self.selected_index != -1:

            # TODO: change current directory of archive
            if len(self.current_dir) > 0 and self.selected_index != -2:
                VPKFileHandle().dir_in(self.current_dir[self.selected_index].name)
            self.current_dir.clear()
            entry = self.current_dir.add()
            entry.name = '..'
            for directory in VPKFileHandle().current_dir():
                entry = self.current_dir.add()
                entry.name = directory

            self.selected_index = -1
        self.layout.template_list("FILE_UL_VPKDirList", "", self, "current_dir", self, "selected_index")

    def execute(self, context):
        print("execute")
        return {'FINISHED'}


class FILE_UL_VPKDirList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        operator = data
        raf_entry = item

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(raf_entry, "name", text="", emboss=False, icon_value=icon)
            if '.' in raf_entry.name:
                layout.prop(raf_entry, "selected")
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)
