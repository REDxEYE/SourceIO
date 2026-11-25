"""Run `blender --factory-startup --window-maximized --python tools/screenshot_export_nodes.py -- OUTPUT_DIR`.

Use a graphical Blender session; ImageMagick's `convert` crops the window capture
around each node. The script only modifies its temporary scene and node tree.
"""

import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SourceIO.blender_bindings.ui.export_nodes import register_nodes


NODE_IDS = (
    'SourceIOFlexSequenceRuleNode',
    'SourceIOFlexShapeRuleNode',
    'SourceIOFlexStereoRuleNode',
    'SourceIOTextureResizeNode',
    'SourceIOTextureTileNode',
    'SourceIOTextureMixNode',
    'SourceIOPropModelNode',
)


def main(output):
    bpy.context.preferences.view.show_splash = False
    register_nodes()
    output.mkdir(parents=True, exist_ok=True)
    scratch = TemporaryDirectory()
    screenshot = Path(scratch.name) / 'window_screenshot.png'

    mesh = bpy.data.meshes.new('Example mesh')
    obj = bpy.data.objects.new('Face', mesh)
    bpy.context.scene.collection.objects.link(obj)
    for name in ('Basis', 'Narrow', 'Neutral', 'Wide'):
        obj.shape_key_add(name=name)
    obj.data.shape_keys.key_blocks['Wide'].slider_max = 2
    obj.data.shape_keys.key_blocks['Narrow'].slider_min = -1

    window = bpy.context.window
    area = next(area for area in window.screen.areas if area.type == 'VIEW_3D')
    area.type = 'NODE_EDITOR'
    area.ui_type = 'sourceio.model_definition'
    space = area.spaces.active
    space.show_region_ui = False
    space.show_region_toolbar = False
    tree = bpy.data.node_groups.new('Export node screenshots', area.ui_type)
    space.pin = True
    space.node_tree = tree
    region = next(region for region in area.regions if region.type == 'WINDOW')

    def configure(node):
        match node.bl_idname:
            case 'SourceIOFlexSequenceRuleNode':
                node.obj = obj
                node.use_shape_range = True
                node.shapes.add()
                for row, name in zip(node.shapes, ('Basis', 'Neutral', 'Wide')):
                    row.shape_name = name
            case 'SourceIOFlexShapeRuleNode':
                node.obj = obj
                node.shape_name = 'Wide'
                node.use_shape_range = True
            case 'SourceIOFlexStereoRuleNode':
                node.obj = obj
                node.left_shape, node.right_shape = 'Narrow', 'Wide'
                node.use_shape_range = True
            case 'SourceIOTextureTileNode':
                node.tiles_x, node.tiles_y = 2, 3
                node.resample = True
            case 'SourceIOPropModelNode':
                node.model_path = 'models/props/example.mdl'
                node.compiler_backend = 'DIRECT'

    def spawn():
        for node_id in NODE_IDS:
            node = tree.nodes.new(node_id)
            node.width = 315
            node.select = True
            tree.nodes.active = node
            configure(node)
            yield
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.node.view_selected()
            yield
            with bpy.context.temp_override(window=window, area=area, region=region):
                assert bpy.ops.screen.screenshot(filepath=str(screenshot)) == {'FINISHED'}
            with screenshot.open('rb') as image:
                image.seek(16)
                width, height = struct.unpack('>II', image.read(8))
            view = region.view2d
            left, top = view.view_to_region(*node.location, clip=False)
            right, bottom = view.view_to_region(node.location.x + node.width,
                                                node.location.y - node.dimensions.y, clip=False)
            sx, sy = width / window.width, height / window.height
            x0, x1 = round((region.x + left) * sx) - 22, round((region.x + right) * sx) + 22
            y0 = height - round((region.y + top) * sy) - 12
            y1 = height - round((region.y + bottom) * sy) + 18
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise RuntimeError(f'{node_id} does not fit in the window; maximize Blender and retry')
            subprocess.run(['convert', str(screenshot), '-crop', f'{x1-x0}x{y1-y0}+{x0}+{y0}',
                            '+repage', str(output / f'{node_id}.png')], check=True)
            print(f'Captured {node_id}', flush=True)
            tree.nodes.remove(node)

    frames = spawn()

    def step():
        try:
            next(frames)
        except StopIteration:
            scratch.cleanup()
            print('Screenshots complete', flush=True)
            return None
        return 0.45

    bpy.app.timers.register(step, first_interval=1)


if __name__ == '__main__':
    [directory] = sys.argv[sys.argv.index('--') + 1:]
    main(Path(directory).resolve())
