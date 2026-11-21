"""Direct export: feature check, geometry baking and the compiled model, without Blender."""

from __future__ import annotations

import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from SourceIO.blender_bindings.source1.model_export.direct_build import (
    MAX_WEIGHTS, FlexControl, build_model, check_joint_collision, direct_support, flex_setup, geometry_arrays,
    parse_advanced_flex)
from SourceIO.blender_bindings.source1.model_export.dmx_types import (
    Bone, Capture, Mesh, Rig, Shape, VertexAnimationRange, face_array)
from SourceIO.blender_bindings.source1.model_export.procedural_bones import (
    HelperBoneRecipe, HelperTriggerRecipe, helper_rows)
from SourceIO.blender_bindings.source1.model_export.prop_qc import (
    BLANK, AttachmentRecipe, BodygroupRecipe, CollisionRecipe, FlexControllerRecipe, FlexRuleRecipe,
    FlexRuleSetRecipe, GeometryRecipe, HitboxRecipe, IdleRecipe, LODRecipe, ModelEntryRecipe, ModelRecipe,
    RawQcRecipe, SkinFamilyRecipe, VirtualControllerRecipe)

IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
UP = ((0., 0., 1.),) * 3
UVS = ((0., 0.), (1., 0.), (0., 1.))


def translation(x, y, z):
    return ((1., 0., 0., x), (0., 1., 0., y), (0., 0., 1., z), (0., 0., 0., 1.))


def triangle_mesh(name, weights=(), matrix=IDENTITY, material='skin', parent=None, parent_bone=None):
    return Mesh(name, matrix, parent, 'BONE' if parent_bone else None, parent_bone,
                ((0., 0., 0.), (10., 0., 0.), (0., 10., 0.)), *face_array([((0, 1, 2), UP, UVS, material)]),
                weights, ())


RIG = Rig('rig', IDENTITY, (Bone('root', None, IDENTITY), Bone('arm', 'root', translation(0, 0, 10)),
                            Bone('twist', 'arm', translation(5, 0, 10)), Bone('hand', 'arm', translation(10, 0, 10))))
SKINNED = (('arm', .25), ('root', .75))


def geometry(name):
    return GeometryRecipe(object(), name)


class DirectSupportTests(unittest.TestCase):
    def setUp(self):
        self.body = geometry('body')

    def model(self, **fields):
        return ModelRecipe('models/props/test.mdl', (ModelEntryRecipe('studio', self.body),), **fields)

    def test_each_unsupported_feature_is_named_with_what_to_do(self):
        hulls = geometry('hulls')
        clip = GeometryRecipe(object(), 'clip', vertex_animations=(VertexAnimationRange('pulse', 0, 2),))
        cases = (
            (self.model(raw_snippets=(RawQcRecipe('\n$surfaceprop "metal"\n'),)), {},
             "Raw QC '$surfaceprop \"metal\"' can't be used with direct export; use the Built-in studiomdl backend "
             "(QC route) or replace it with nodes"),
            (self.model(collision=CollisionRecipe(hulls, mode='JOINTS', joint_qc='$rootbone "pelvis"')), {},
             "$collisionjoints joint QC '$rootbone \"pelvis\"' can't be used with direct export"),
            (ModelRecipe('models/props/test.mdl', (ModelEntryRecipe('studio', clip),)), {},
             "Vertex animation 'pulse' of 'clip' isn't supported by direct export yet"),
            (self.model(), {'controls': "flex rule 'x' of 'face' has the unsupported type 'DmeFlexRuleFoo'"},
             "Advanced flex Text 'controls' can't be used with direct export: flex rule 'x' of 'face' has the "
             "unsupported type 'DmeFlexRuleFoo'"),
        )
        for model, advanced_errors, message in cases:
            with self.subTest(message=message):
                (issue,) = direct_support(model, advanced_errors=advanced_errors)
                self.assertEqual(issue.severity, 'ERROR')
                self.assertTrue(issue.message.startswith(message), issue.message)
        flexed = ModelRecipe('models/props/test.mdl', (ModelEntryRecipe(
            'studio', self.body, flex_controllers=(FlexControllerRecipe('smile', ('smile',)),)),))
        self.assertEqual(direct_support(flexed), ())
        self.assertEqual(direct_support(self.model(collision=CollisionRecipe(hulls))), ())

    def test_conflicting_flex_rules_across_objects_are_an_error(self):
        controls = (FlexControllerRecipe('smile', ('smile',)), VirtualControllerRecipe('jaw'))

        def model(*rules, attr='models'):
            entry = ModelEntryRecipe('head', self.body, flex_rule_set=FlexRuleSetRecipe(
                self.body, controls, tuple(FlexRuleRecipe(*rule) for rule in rules)))
            if attr == 'models':
                return ModelRecipe('models/props/test.mdl', (entry,))
            return ModelRecipe('models/props/test.mdl', (ModelEntryRecipe('studio', self.body),), flex_entries=(entry,))

        self.assertEqual(direct_support(model(('face', 'smile', 'smile'), ('teeth', 'Smile', 'smile'))), ())
        for attr in ('models', 'flex_entries'):
            (issue,) = direct_support(model(('face', 'smile', 'smile'), ('teeth', 'Smile', 'jaw'), attr=attr))
            self.assertEqual(issue.severity, 'ERROR')
            self.assertIn("objects give the flex 'Smile' different rules", issue.message)


class GeometryArrayTests(unittest.TestCase):
    def test_mirrored_objects_keep_outward_winding_and_normals(self):
        mirror = ((-1., 0., 0., 5.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        normals = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        mesh = Mesh('panel', mirror, None, None, None, ((0., 0., 0.), (10., 0., 0.), (0., 10., 0.)),
                    *face_array([((0, 1, 2), normals, UVS, None)]), (), ())
        arrays = geometry_arrays(Capture('panel', (mesh,), None))
        np.testing.assert_allclose(arrays.positions, [[5, 0, 0], [-5, 0, 0], [5, 10, 0]])
        self.assertEqual(arrays.triangles.tolist(), [[0, 2, 1]])
        np.testing.assert_allclose(arrays.normals, [[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_allclose(arrays.uvs, [[0, 0], [0, 1], [1, 0]])
        self.assertEqual(arrays.materials, ('no_material',))
        # Unrigged objects get a bone of their own, like SMD nodes
        self.assertEqual([name for name, _, _ in arrays.skeleton], ['panel'])
        self.assertEqual(arrays.bones.tolist(), [[0]] * 3)

    def test_skin_keeps_the_16_strongest_weights_normalized(self):
        bones = tuple(Bone(f'b{index}', None, IDENTITY) for index in range(20))
        links = tuple((f'b{index}', float(index + 1)) for index in range(20))
        mesh = triangle_mesh('body', (links, (('b3', 2.), ('b7', 2.)), (('b0', 1.),)))
        arrays = geometry_arrays(Capture('body', (mesh,), Rig('rig', IDENTITY, bones)))
        self.assertEqual(arrays.bones.shape, (3, MAX_WEIGHTS))
        kept = dict(zip(arrays.bones[0].tolist(), arrays.weights[0].tolist()))
        self.assertEqual(sorted(kept), list(range(4, 20)))
        self.assertAlmostEqual(kept[19], 20 / sum(range(5, 21)), places=6)
        self.assertEqual(sorted(zip(arrays.bones[1].tolist(), arrays.weights[1].tolist()))[-2:], [(3, .5), (7, .5)])
        np.testing.assert_allclose(arrays.weights.sum(axis=1), 1, rtol=1e-6)

    def test_bone_parented_objects_and_joint_hulls_are_rigid(self):
        hand = triangle_mesh('ring', parent='rig', parent_bone='hand')
        arrays = geometry_arrays(Capture('ring', (hand,), RIG))
        self.assertEqual(arrays.bones.tolist(), [[3]] * 3)
        self.assertEqual([(name, parent) for name, parent, _ in arrays.skeleton],
                         [('root', None), ('arm', 'root'), ('twist', 'arm'), ('hand', 'arm')])
        np.testing.assert_allclose(arrays.skeleton[2][2][:3, 3], (5, 0, 0))
        hull = triangle_mesh('hull', (SKINNED,) * 3)
        joints = geometry_arrays(Capture('hulls', (hull,), RIG), joints=True)
        self.assertEqual(joints.bones.tolist(), [[0]] * 3)
        self.assertEqual(joints.materials, ('hull',))
        split = triangle_mesh('split', (SKINNED, (('arm', 1.),), SKINNED))
        with self.assertRaisesRegex(ValueError, 'spans multiple bones'):
            geometry_arrays(Capture('hulls', (split,), RIG), joints=True)

    def test_joint_hull_weight_ties_pick_the_lower_bone_index_like_smd(self):
        from SourceIO.blender_bindings.source1.model_export.smd_builder import build_document
        tied = (('twist', .5), ('arm', .5), ('root', .25 + .25))  # listed highest index first
        capture = Capture('hulls', (triangle_mesh('hull', (tied,) * 3),), RIG)
        direct = geometry_arrays(capture, joints=True)
        smd = build_document(capture, joints=True)
        self.assertEqual(direct.bones.tolist(), [[0]] * 3)
        self.assertEqual(set(smd.triangles['corners']['bone'].ravel().tolist()), {0})


def shape(name, base, moved=None, normal=(0., 0., 1.)):
    """A Shape of a one-triangle mesh: ``moved`` maps vertex -> offset (object space)."""
    positions = [list(position) for position in base]
    for vertex, offset in (moved or {}).items():
        positions[vertex] = [a + b for a, b in zip(positions[vertex], offset)]
    return Shape(name, tuple(map(tuple, positions)), ((normal,) * 3,))


def shaped_mesh(name, shapes, matrix=IDENTITY):
    base = ((0., 0., 0.), (10., 0., 0.), (0., 10., 0.))
    return Mesh(name, matrix, None, None, None, base, *face_array([((0, 1, 2), UP, UVS, 'skin')]), (),
                tuple(shape(item[0], base, *item[1:]) for item in shapes))


def flex_capture():
    scaled = ((2., 0., 0., 1.), (0., 2., 0., 0.), (0., 0., 2., 0.), (0., 0., 0., 1.))
    face = shaped_mesh('face', (('smile', {1: (0., 0., 1.)}), ('blink.L', {0: (0., 0., 1.)}),
                                ('blink.R', {2: (0., 0., 1.)})), scaled)
    teeth = shaped_mesh('teeth', (('smile', {2: (0., 0., .5)}), ('open', {}, (1., 0., 0.))))
    return Capture('head', (face, teeth), None)


def advanced_text(targets):
    """KeyValues2 advanced flex Text: controls smile (raw smile) and jaw (no raw), the given targets."""
    from SourceIO.library.utils import datamodel_v2 as dm
    document = dm.DataModel('keyvalues2', 1, 'model', 1)
    root = document.add_element('flex', uuid='t/flex')
    operator = document.add_element('combinationOperator', 'DmeCombinationOperator', uuid='t/operator')
    root['combinationOperator'] = operator
    controls = []
    for name, raw, scales in (('smile', ['smile'], [0.5]), ('jaw', [], [])):
        control = document.add_element(name, 'DmeCombinationInputControl', uuid=f't/control/{name}')
        control['rawControlNames'] = dm.StringArray(raw)
        control['stereo'] = False
        control['eyelid'] = False
        control['flexMin'] = 0.0
        control['flexMax'] = 2.0
        control['wrinkleScales'] = dm.FloatArray(scales)
        controls.append(control)
    operator['controls'] = dm.ElementArray(controls)
    elements = []
    for kind, name, rules in targets:
        target = document.add_element(name, kind, uuid=f't/target/{name}')
        if kind == 'DmeFlexRules':
            states = []
            for rule_type, flex, expression in rules:
                rule = document.add_element(flex, rule_type, uuid=f't/rule/{name}/{flex}')
                if expression:
                    rule['expr'] = expression
                states.append(rule)
            target['deltaStates'] = dm.ElementArray(states)
        elements.append(target)
    operator['targets'] = dm.ElementArray(elements)
    return dm.dump(document, encoding='keyvalues2')


class FlexTests(unittest.TestCase):
    def test_shapes_are_sparse_model_space_deltas_merged_by_name(self):
        arrays = geometry_arrays(flex_capture(), wrinkle_scales={('face', 'smile'): .8})
        shapes = {item[0]: item[1:] for item in arrays.shapes}
        self.assertEqual(list(shapes), ['smile', 'blink.L', 'blink.R', 'open'])
        vertices, deltas, normals, wrinkle = shapes['smile']
        self.assertEqual(vertices.tolist(), [1, 5])  # the teeth follow the face's three vertices
        np.testing.assert_allclose(deltas, [[0, 0, 2], [0, 0, .5]])  # the face object is scaled by 2
        np.testing.assert_allclose(normals, 0)
        np.testing.assert_allclose(wrinkle, [.8, 0])
        vertices, deltas, normals, wrinkle = shapes['open']  # a normal-only shape
        self.assertEqual(vertices.tolist(), [3, 4, 5])
        np.testing.assert_allclose(deltas, 0)
        np.testing.assert_allclose(normals, [[1, 0, -1]] * 3)
        self.assertIsNone(wrinkle)
        self.assertEqual(arrays.balance.shape, (6,))
        self.assertTrue(((arrays.balance >= 0) & (arrays.balance <= 1)).all())
        self.assertEqual(geometry_arrays(flex_capture(), joints=True).shapes, ())

    def test_controls_follow_the_dmx_route(self):
        capture = flex_capture()
        auto = flex_setup(capture)
        self.assertEqual([(control.name, control.raw) for control in auto.controls],
                         [(name, (name,)) for name in ('smile', 'blink.L', 'blink.R', 'open')])
        self.assertEqual(auto.rules, ())
        self.assertIsNone(flex_setup(Capture('plain', (triangle_mesh('plain'),), None)))

        head = geometry('head')
        entry = ModelEntryRecipe('head', head, flex_controllers=(
            FlexControllerRecipe('blink', ('blink.L', 'blink.R'), eyelid=True, wrinkle_scales=(0., 2.)),
            FlexControllerRecipe('smile', ('smile',), stereo=True, minimum=-1.0)))
        setup = flex_setup(capture, entry)
        self.assertEqual(setup.controls[0], FlexControl('blink', ('blink.L', 'blink.R'), eyelid=True,
                                                        wrinkle_scales=(0., 2.)))
        self.assertEqual((setup.controls[1].stereo, setup.controls[1].minimum), (True, -1.0))
        self.assertEqual(dict(setup.wrinkle_scales), {('face', 'blink.R'): 2.})

        controls = (FlexControllerRecipe('smile', ('smile',)), VirtualControllerRecipe('jaw', maximum=2.0))

        def rules(*items):
            return ModelEntryRecipe('head', head, flex_rule_set=FlexRuleSetRecipe(head, controls, tuple(
                FlexRuleRecipe(*item) for item in items)))

        covered = [('face', 'smile', 'smile', .5), ('face', 'blink.L', 'smile'), ('face', 'blink.R', 'smile'),
                   ('teeth', 'smile', 'smile'), ('teeth', 'open', 'jaw * 0.5')]
        setup = flex_setup(capture, rules(*covered))
        self.assertEqual(setup.controls[1].raw, ())
        self.assertEqual(setup.rules, (('expression', 'smile', 'smile'), ('expression', 'blink.L', 'smile'),
                                       ('expression', 'blink.R', 'smile'), ('expression', 'open', 'jaw * 0.5')))
        self.assertEqual(dict(setup.wrinkle_scales), {('face', 'smile'): .5})
        with self.assertRaisesRegex(ValueError, "objects give the flex 'smile' different rules"):
            flex_setup(capture, rules(*covered[:3], ('teeth', 'smile', 'jaw'), covered[4]))
        with self.assertRaisesRegex(ValueError, 'omit captured shape targets'):
            flex_setup(capture, rules(*covered[:4]))

    def test_advanced_text_uses_its_rules_only_when_every_target_has_rules(self):
        face = Capture('face', (shaped_mesh('face', (('smile', {1: (0., 0., 1.)}), ('open', {0: (0., 0., 1.)}))),),
                       None)
        ruled = parse_advanced_flex(advanced_text((('DmeFlexRules', 'face', (
            ('DmeFlexRuleExpression', 'open', 'jaw * smile'), ('DmeFlexRulePassThrough', 'smile', ''),
            ('DmeFlexRuleLocalVar', 'spare', ''))),)))
        self.assertEqual([(control.name, control.raw, control.maximum) for control in ruled.controls],
                         [('smile', ('smile',), 2.0), ('jaw', (), 2.0)])
        setup = flex_setup(face, advanced=ruled)
        self.assertEqual(setup.rules, (('expression', 'open', 'jaw * smile'), ('pass', 'smile', ''),
                                       ('local', 'spare', '')))
        self.assertEqual(dict(setup.wrinkle_scales), {('face', 'smile'): .5})
        mixed = parse_advanced_flex(advanced_text((('DmeMesh', 'face', ()),)))
        self.assertEqual(flex_setup(face, advanced=mixed).rules, ())
        with self.assertRaisesRegex(ValueError, "unsupported type 'DmeFlexRuleFoo'"):
            parse_advanced_flex(advanced_text((('DmeFlexRules', 'face', (('DmeFlexRuleFoo', 'x', ''),)),)))
        with self.assertRaisesRegex(ValueError, "unknown controller 'nothing'"):
            flex_setup(face, advanced=parse_advanced_flex(advanced_text((('DmeFlexRules', 'face', (
                ('DmeFlexRuleExpression', 'open', 'nothing'),)),))))


def studiomdl():
    try:
        from SourceIO.library.utils.pylib import studiomdl
    except ImportError:
        return None
    return studiomdl


class JointCollisionSharingTests(unittest.TestCase):
    def test_joint_collision_geometry_cannot_be_a_model_bodygroup_or_lod_geometry(self):
        shared, other = geometry('shared'), geometry('other')
        joints = CollisionRecipe(shared, mode='JOINTS')
        cases = {
            'model': ModelRecipe('models/props/t.mdl', (ModelEntryRecipe('studio', shared),), collision=joints),
            'bodygroup': ModelRecipe('models/props/t.mdl', (ModelEntryRecipe('studio', other),), collision=joints,
                                     bodygroups=(BodygroupRecipe('g', (BLANK, shared)),)),
            'lod': ModelRecipe('models/props/t.mdl', (ModelEntryRecipe('studio', other),), collision=joints,
                               lods=(LODRecipe(other, shared, 20),)),
        }
        for name, recipe in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, r"\$collisionjoints geometry 'shared'"):
                    build_model(None, recipe, {})
        separate = ModelRecipe('models/props/t.mdl', (ModelEntryRecipe('studio', other),), collision=joints)
        check_joint_collision(separate)
        check_joint_collision(ModelRecipe('models/props/t.mdl', (ModelEntryRecipe('studio', shared),),
                                          collision=CollisionRecipe(shared)))


@unittest.skipIf(studiomdl() is None, 'this pylib build has no studiomdl')
class BuildModelTests(unittest.TestCase):
    def test_direct_weights_use_one_pose_count_for_fitting_and_testing(self):
        body = geometry('body')
        recipe = ModelRecipe('models/props/weights.mdl', (ModelEntryRecipe('studio', body),))
        native = Mock(wraps=studiomdl().Model('props/weights'))
        build_model(SimpleNamespace(Model=lambda name: native), recipe,
                    {'body': Capture('body', (triangle_mesh('body'),), None)},
                    fit_weight_poses=5)
        native.set_fit_weights.assert_called_once_with(True, 5, 5)

    def test_compiled_model_has_the_recipe_parts(self):
        from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
        from SourceIO.library.utils import MemoryBuffer

        body, sleeve, low, hull = (geometry(name) for name in ('body', 'sleeve', 'low', 'hull'))
        weights = (SKINNED, (('arm', 1.),), (('hand', 1.),))
        captures = {
            'body': Capture('body', (triangle_mesh('body', weights),), RIG),
            'sleeve': Capture('sleeve', (triangle_mesh('sleeve', weights, material='cloth'),), RIG),
            'low': Capture('low', (triangle_mesh('low', weights),), RIG),
            'hull': Capture('hull', (Mesh('hull', IDENTITY, None, None, None,
                                          ((0., 0., 0.), (10., 0., 0.), (0., 10., 0.), (0., 0., 10.)),
                                          *face_array((face, UP, UVS, None)
                                                      for face in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))),
                                          (), ()),), None),
        }
        helper = HelperBoneRecipe('twist', 'hand', (HelperTriggerRecipe(), HelperTriggerRecipe(
            tolerance=45, control_rotation=(0., 1., 0.), helper_rotation=(0., .5, 0.))))
        recipe = ModelRecipe(
            'models/props/direct.mdl', (ModelEntryRecipe('studio', body),),
            bodygroups=(BodygroupRecipe('sleeves', (BLANK, sleeve)),),
            skinfamilies=(SkinFamilyRecipe(('skin', 'cloth')), SkinFamilyRecipe(('cloth', 'skin'))),
            lods=(LODRecipe(body, low, 20),), collision=CollisionRecipe(hull, mass=12, concave=True),
            attachments=(AttachmentRecipe('grip', 'hand', (1., 2., 3.), (0., 90., 0.)),),
            hitboxes=(HitboxRecipe(1, 'arm', (-1., -1., -1.), (1., 1., 1.)),), idle=IdleRecipe(),
            procedural_bones=(helper,), scale=2.0)
        rest = {bone.name: (bone.parent, np.array(bone.matrix_local)) for bone in RIG.bones}
        model = build_model(studiomdl(), recipe, captures, material_dirs=('models/props',),
                            helpers=helper_rows(recipe.procedural_bones, rest, scale=1.0),
                            fit_weight_poses=7)
        result = model.compile()
        self.assertEqual(sorted(result.files), [f'models/props/direct.{suffix}'
                                                for suffix in ('dx90.vtx', 'mdl', 'phy', 'vvd')])
        data = result.files['models/props/direct.mdl']
        mdl = MdlV49.from_buffer(MemoryBuffer(data))
        names = [bone.name for bone in mdl.bones]
        # The unrigged hull's own bone has no render vertices and is collapsed, as on the QC route; the helper bone
        # has no vertices either but is kept for its procedural rule
        self.assertEqual(set(names), {'root', 'arm', 'twist', 'hand'})
        self.assertNotEqual(mdl.bones[names.index('twist')].procedural_rule_type, 0)
        self.assertEqual([(part.name, [model.vertex_count for model in part.models]) for part in mdl.body_parts],
                         [('studio', [3]), ('sleeves', [0, 3])])
        self.assertEqual(mdl.materials_paths, ['models/props/'])
        self.assertEqual([[material.name for material in family] for family in mdl.skin_groups],
                         [['skin', 'cloth'], ['cloth', 'skin']])
        (attachment,) = mdl.attachments
        self.assertEqual((attachment.name, names[attachment.parent_bone]), ('grip', 'hand'))
        np.testing.assert_allclose(np.asarray(attachment.matrix).reshape(3, 4)[:, 3], (2, 4, 6), atol=1e-4)
        self.assertEqual(mdl.header.hitbox_set_count, 1)
        self.assertEqual(struct.unpack_from('<i', data, mdl.header.hitbox_set_offset + 4)[0], 1)
        self.assertEqual([sequence.name for sequence in mdl.sequences], ['idle'])

    def test_compiled_flexes_follow_controls_and_rules(self):
        from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
        from SourceIO.library.utils import MemoryBuffer

        head = geometry('head')
        base = flex_capture()
        capture = Capture('head', base.meshes + (shaped_mesh('brow', (('smile_blink.L', {0: (0., 0., 1.)}),)),),
                          None)

        def compiled(entry):
            recipe = ModelRecipe('models/props/flex.mdl', (entry,), idle=IdleRecipe())
            result = build_model(studiomdl(), recipe, {'head': capture}).compile()
            mdl = MdlV49.from_buffer(MemoryBuffer(result.files['models/props/flex.mdl']))
            flexes = {mdl.flex_names[flex.flex_desc_index]: flex.vertex_anim_type.name
                      for flex in mdl.body_parts[0].models[0].meshes[0].flexes}
            return mdl, [(item.name, item.min, item.max) for item in mdl.flex_controllers], flexes

        # Controls: a 2-way slider, a stereo control with a wrinkle (L/R flexes), a corrective shape
        mdl, controllers, flexes = compiled(ModelEntryRecipe('head', head, flex_controllers=(
            FlexControllerRecipe('blink', ('blink.L', 'blink.R'), wrinkle_scales=(0., 0.)),
            FlexControllerRecipe('smile', ('smile',), stereo=True, wrinkle_scales=(1.,)),
            FlexControllerRecipe('open', ('open',)))))
        self.assertEqual(controllers, [('blink', -1.0, 1.0), ('right_smile', 0.0, 1.0), ('left_smile', 0.0, 1.0),
                                       ('open', 0.0, 1.0)])
        self.assertEqual(set(mdl.flex_names), {'smileL', 'smileR', 'blink.L', 'blink.R', 'open', 'smile_blink.LL',
                                               'smile_blink.LR'})
        self.assertEqual(flexes['smileL'], 'WRINKLE')
        self.assertEqual(flexes['open'], 'NORMAL')

        # Flex rules: a controller without shapes feeds expressions
        controls = (FlexControllerRecipe('smile', ('smile',)), VirtualControllerRecipe('jaw', maximum=2.0))
        rules = (('face', 'smile', 'smile'), ('face', 'blink.L', 'jaw'), ('face', 'blink.R', 'jaw'),
                 ('teeth', 'smile', 'smile'), ('teeth', 'open', 'jaw * 0.5'), ('brow', 'smile_blink.L', 'smile*jaw'))
        mdl, controllers, flexes = compiled(ModelEntryRecipe('head', head, flex_rule_set=FlexRuleSetRecipe(
            head, controls, tuple(FlexRuleRecipe(*rule) for rule in rules))))
        self.assertEqual(controllers, [('smile', 0.0, 1.0), ('jaw', 0.0, 2.0)])
        self.assertEqual(sorted(mdl.flex_names), sorted(['smile', 'blink.L', 'blink.R', 'open', 'smile_blink.L']))
        self.assertEqual(len(mdl.flex_rules), 5)


if __name__ == '__main__':
    unittest.main()
