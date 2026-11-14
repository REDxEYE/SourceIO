"""Create export-graph source nodes from files dropped into the Node Editor."""

from pathlib import Path

import bpy


_IMAGE_EXTENSIONS = frozenset({
    '.bmp', '.dds', '.exr', '.hdr', '.jpeg', '.jpg', '.png', '.tga',
    '.tif', '.tiff', '.vtf', '.webp',
})
_QC_EXTENSIONS = frozenset({'.qc', '.qci'})


class SOURCEIO_OT_DropExportAsset(bpy.types.Operator):
    bl_idname = 'sourceio.drop_export_asset'
    bl_label = 'Add SourceIO File Nodes'
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(
        subtype='FILE_PATH', options={'SKIP_SAVE'})
    directory: bpy.props.StringProperty(
        subtype='DIR_PATH', options={'SKIP_SAVE', 'HIDDEN'})
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={'SKIP_SAVE', 'HIDDEN'})
    location: bpy.props.FloatVectorProperty(
        size=2, options={'SKIP_SAVE', 'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return (context.area is not None
                and context.area.type == 'NODE_EDITOR'
                and context.area.ui_type == 'sourceio.model_definition'
                and context.space_data.edit_tree is not None)

    def invoke(self, context, event):
        context.space_data.cursor_location_from_region(
            event.mouse_region_x, event.mouse_region_y)
        self.location = context.space_data.cursor_location
        return self.execute(context)

    def execute(self, context):
        if self.files:
            directory = Path(bpy.path.abspath(self.directory))
            paths = [directory / item.name for item in self.files]
        elif self.filepath:
            paths = [Path(bpy.path.abspath(self.filepath))]
        else:
            self.report({'ERROR'}, 'Drop an image, VTF, QC, or QCI file')
            return {'CANCELLED'}
        paths = list(dict.fromkeys(paths))
        for path in paths:
            if (path.suffix.lower() not in _IMAGE_EXTENSIONS | _QC_EXTENSIONS
                    or not path.is_file()):
                self.report({'ERROR'},
                            f'Unsupported or missing SourceIO file: {path}')
                return {'CANCELLED'}

        texts = {}
        try:
            for path in paths:
                if path.suffix.lower() in _QC_EXTENSIONS:
                    texts[path] = bpy.data.texts.load(str(path),
                                                      internal=False)
        except (OSError, RuntimeError) as exc:
            for text in texts.values():
                bpy.data.texts.remove(text)
            self.report({'ERROR'}, f'Could not load QC text: {exc}')
            return {'CANCELLED'}

        tree = context.space_data.edit_tree
        for node in tree.nodes:
            node.select = False
        x, y = self.location
        for index, path in enumerate(paths):
            if path.suffix.lower() in _QC_EXTENSIONS:
                node = tree.nodes.new('SourceIOPropRawQcNode')
                node.snippet_text = texts[path]
            else:
                node = tree.nodes.new('SourceIOExternalTextureNode')
                node.source_path = str(path)
            node.label = path.name
            node.location = (x + index * 240, y)
            node.select = True
            tree.nodes.active = node
        return {'FINISHED'}


class NODE_FH_SourceIOExportAssets(bpy.types.FileHandler):
    bl_idname = 'NODE_FH_SourceIOExportAssets'
    bl_label = 'SourceIO export graph file drop'
    bl_import_operator = SOURCEIO_OT_DropExportAsset.bl_idname
    bl_file_extensions = ';'.join(sorted(_IMAGE_EXTENSIONS | _QC_EXTENSIONS))

    @classmethod
    def poll_drop(cls, context):
        return (context.region is not None
                and context.region.type == 'WINDOW'
                and SOURCEIO_OT_DropExportAsset.poll(context))
