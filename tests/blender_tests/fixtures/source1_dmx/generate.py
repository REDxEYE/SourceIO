"""Capture reference DMX files for the native exporter contract.

Run outside Blender: python generate.py [--output DIRECTORY] [--blender EXECUTABLE]
Each case runs in a fresh headless Blender process because the exporter uses undo.
These binary files are reference behavior, not expected byte-for-byte output.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CASES = ('single', 'multi', 'rigged', 'flex', 'vertex_animation')
VERSIONS = ((2, 1), (4, 15), (5, 18))
HERE = Path(__file__).resolve().parent
SOURCEIO_PARENT = HERE.parents[4]


def _mesh(bpy, collection, name, vertices, faces, material_names):
    data = bpy.data.meshes.new(f'{name} mesh')
    data.from_pydata(vertices, (), faces)
    for material_name in material_names:
        material = bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name)
        data.materials.append(material)
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.vs.export = True
    return obj


def _scene_case(bpy, case, scene):
    if case == 'single':
        obj = _mesh(bpy, scene.collection, 'Reference Panel', ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
                    ((0, 1, 2), (0, 2, 3)), ('reference_base', 'reference_data'))
        override = bpy.data.materials.new('reference_override')
        obj.material_slots[1].link = 'OBJECT'
        obj.material_slots[1].material = override
        obj.data.polygons[1].material_index = 1
        uv = obj.data.uv_layers.new(name='UVMap')
        for loop, coords in enumerate(((0, 0), (1, 0), (1, 1), (0.5, 0), (1, 1), (0, 1))):
            uv.data[loop].uv = coords
        obj.location = (2, 3, 1)
        obj.rotation_euler[2] = 0.5
        obj.scale = (-1, 1.5, 1)
        modifier = obj.modifiers.new('Reference displacement', 'DISPLACE')
        modifier.mid_level = 0.0
        modifier.strength = 0.125
        return obj

    collection = bpy.data.collections.new(f'Reference {case}')
    scene.collection.children.link(collection)
    collection.vs.export = True
    collection.vs.automerge = False

    if case == 'multi':
        left = _mesh(bpy, collection, 'Panel Left', ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
                     ('reference_left',))
        right = _mesh(bpy, collection, 'Panel Right', ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
                      ('reference_right',))
        left.location.x = -2
        right.location.x = 2
        right.rotation_euler[2] = 0.25
        return collection

    if case == 'rigged':
        armature = bpy.data.armatures.new('Reference rig data')
        rig = bpy.data.objects.new('Reference Rig', armature)
        collection.objects.link(rig)
        rig.vs.export = True
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        root = armature.edit_bones.new('Root')
        root.head = (0, 0, 0)
        root.tail = (0, 1, 0)
        child = armature.edit_bones.new('Child')
        child.head = root.tail
        child.tail = (0, 2, 0)
        child.parent = root
        bpy.ops.object.mode_set(mode='OBJECT')
        obj = _mesh(bpy, collection, 'Weighted Panel', ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
                    ('reference_skin',))
        obj.vertex_groups.new(name='Root').add((0, 2), 0.5, 'REPLACE')
        obj.vertex_groups.new(name='Child').add((1, 2), 0.5, 'REPLACE')
        modifier = obj.modifiers.new('Armature', 'ARMATURE')
        modifier.object = rig
        return collection

    if case == 'flex':
        sys.path.insert(0, str(SOURCEIO_PARENT))
        from SourceIO.tests.blender_tests.fixtures.source1_dmx.flex_assets import render_flex_controllers
        from SourceIO.blender_bindings.source1.model_export.prop_qc import (
            FlexControllerRecipe, FlexRuleRecipe, FlexRuleSetRecipe, GeometryRecipe, ModelEntryRecipe,
            VirtualControllerRecipe)

        obj = _mesh(bpy, collection, 'Flex Face', ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),), ('reference_face',))
        obj.shape_key_add(name='Basis')
        obj.shape_key_add(name='smile').data[0].co.x += 0.2
        obj.shape_key_add(name='blink').data[0].co.z += 0.2
        geometry = GeometryRecipe(obj, 'flex')
        rules = FlexRuleSetRecipe(
            geometry,
            (FlexControllerRecipe('faceCtrl', ('smile',), True, True, -0.25, 0.8, (0.4,)),
             VirtualControllerRecipe('expression_driver')),
            (FlexRuleRecipe(obj.name, 'smile', '(faceCtrl * 0.5)'),
             FlexRuleRecipe(obj.name, 'blink', 'expression_driver')))
        text = bpy.data.texts.new('Reference flex controls')
        text.from_string(render_flex_controllers(ModelEntryRecipe('studio', geometry, flex_rule_set=rules)))
        collection.vs.flex_controller_mode = 'ADVANCED'
        collection.vs.flex_controller_source = text.name
        return collection

    if case == 'vertex_animation':
        obj = _mesh(bpy, collection, 'Animated Panel', ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
                    ('reference_animation',))
        modifier = obj.modifiers.new('Pulse', 'DISPLACE')
        modifier.mid_level = 0.0
        modifier.strength = 0.0
        modifier.keyframe_insert(data_path='strength', frame=1)
        modifier.strength = 0.3
        modifier.keyframe_insert(data_path='strength', frame=2)
        scene.frame_set(1)
        animation = collection.vs.vertex_animations.add()
        animation.name = 'pulse'
        animation.start = 1
        animation.end = 3
        animation.export_sequence = False
        return collection

    raise ValueError(case)


def _capture(case, encoding, model_format, destination):
    import bpy
    import io_scene_valvesource
    from io_scene_valvesource.utils import State

    try:
        bpy.ops.export_scene.smd.get_rna_type()
    except (AttributeError, RuntimeError, KeyError):
        io_scene_valvesource.register()
    scene = bpy.data.scenes.new('Reference Scene')
    bpy.context.window.scene = scene
    source = _scene_case(bpy, case, scene)
    scene.vs.export_format = 'DMX'
    scene.vs.dmx_encoding = str(encoding)
    scene.vs.dmx_format = str(model_format)
    scene.vs.use_kv2 = False
    scene.vs.up_axis = 'Z'
    with tempfile.TemporaryDirectory(prefix='sourceio-dmx-reference-') as folder:
        scene.vs.export_path = folder
        State.update_scene(scene)
        if isinstance(source, bpy.types.Collection):
            result = bpy.ops.export_scene.smd(collection=source.name)
        else:
            result = bpy.ops.export_scene.smd(export_scene=True)
        if 'FINISHED' not in result:
            raise RuntimeError(f'BST cancelled {case} ({encoding}/{model_format})')
        outputs = tuple(Path(folder).rglob('*.dmx'))
        if len(outputs) != 1:
            raise RuntimeError(f'Expected one DMX for {case}, got {outputs}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(outputs[0], destination)
    print(f'FIXTURE_READY {destination}', flush=True)


def main():
    if '--' in sys.argv:
        case, encoding, model_format, destination = sys.argv[sys.argv.index('--') + 1:]
        _capture(case, int(encoding), int(model_format), Path(destination))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--output', type=Path, default=HERE / 'data')
    parser.add_argument('--force', action='store_true', help='Replace existing reference files')
    args = parser.parse_args()
    for case in CASES:
        for encoding, model_format in VERSIONS:
            destination = args.output / f'{case}_{encoding}_{model_format}.dmx'
            if destination.exists() and not args.force:
                raise FileExistsError(f'{destination} exists; pass --force to replace reference files')
            command = (args.blender, '-b', '--factory-startup', '--python-exit-code', '2', '--python',
                       str(Path(__file__).resolve()), '--', case, str(encoding), str(model_format),
                       str(destination.resolve()))
            result = subprocess.run(command, cwd=SOURCEIO_PARENT, capture_output=True, text=True)
            if result.returncode:
                print(result.stdout, file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                raise RuntimeError(f'BST fixture capture failed: {case} {encoding}/{model_format}')
            print(next(line for line in result.stdout.splitlines() if line.startswith('FIXTURE_READY ')))


if __name__ == '__main__':
    main()
