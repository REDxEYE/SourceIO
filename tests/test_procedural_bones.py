"""$proceduralbones VRD rendering, parsing and import for helper bones."""

from __future__ import annotations

import unittest
from math import radians

import numpy as np
from SourceIO.blender_bindings.source1.model_export.procedural_bones import (
    HelperBoneRecipe, HelperTriggerRecipe, euler_matrix, evaluate_helper, helpers_from_vrd, parse_vrd, render_vrd,
    rest_locals)


def _matrix(angles_degrees, position):
    result = np.eye(4)
    result[:3, :3] = euler_matrix([radians(v) for v in angles_degrees])
    result[:3, 3] = position
    return result


def rig():
    """bone -> (parent, armature-space rest matrix), deliberately rolled."""
    pelvis = _matrix((0, 0, 0), (0, 0, 1))
    arm = pelvis @ _matrix((90, 10, -30), (0.2, 0, 0.3))
    return {
        'pelvis': (None, pelvis),
        'arm': ('pelvis', arm),
        'hand': ('arm', arm @ _matrix((-5, -8, 4), (0, 0.25, 0))),
        'twist': ('arm', arm @ _matrix((0, 0, 3), (0, 0.12, 0))),
        'hand.tip': ('hand', arm @ _matrix((0, 0, 0), (0, 0.35, 0))),
    }


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


TWIST = HelperBoneRecipe('twist', 'hand', (
    HelperTriggerRecipe(),
    HelperTriggerRecipe(60.0, (0.0, radians(90), 0.0), (0.0, radians(45), 0.0), (0.0, 0.1, 0.0)),
))


class RenderVrdTests(unittest.TestCase):
    def test_triggers_are_local_transforms_of_the_posed_bones(self):
        bones = rig()
        text = render_vrd((TWIST,), bones, {'hand': 'bip_hand'}, scale=10.0)
        self.assertTrue(text.startswith('<helper> twist arm arm bip_hand\n'))
        (helper,) = parse_vrd(text)
        arm = bones['arm'][1]
        # Rest trigger: hand and twist at rest relative to the arm.
        rest, posed = helper.triggers
        hand_local = np.linalg.inv(arm) @ bones['hand'][1]
        twist_local = np.linalg.inv(arm) @ bones['twist'][1]
        self.assertEqual((rest.tolerance, posed.tolerance), (90.0, 60.0))
        self.assertLess(angle_between(euler_matrix(np.radians(rest.trigger)), hand_local[:3, :3]), 1e-5)
        self.assertLess(angle_between(rest.rotation, twist_local[:3, :3]), 1e-5)
        np.testing.assert_allclose(rest.position, twist_local[:3, 3] * 10, atol=1e-6)
        # Posed trigger: rotations about each bone's own rest Y axis; the
        # helper offset is in its rest frame and scaled to Source units.
        self.assertLess(angle_between(
            euler_matrix(np.radians(posed.trigger)),
            hand_local[:3, :3] @ euler_matrix((0, radians(90), 0))), 1e-5)
        self.assertLess(angle_between(
            posed.rotation, twist_local[:3, :3] @ euler_matrix((0, radians(45), 0))), 1e-5)
        np.testing.assert_allclose(
            posed.position,
            (twist_local[:3, 3] + twist_local[:3, :3] @ (0, 0.1, 0)) * 10,
            atol=1e-6)

    def test_import_reproduces_rendered_helpers(self):
        bones = rig()
        names = {'hand': 'bip_hand', 'arm': 'bip_arm'}
        text = render_vrd((TWIST,), bones, names, scale=39.37)
        (imported,) = helpers_from_vrd(parse_vrd(text), bones, names, 39.37)
        self.assertEqual((imported.bone, imported.control), ('twist', 'hand'))
        for got, expected in zip(imported.triggers, TWIST.triggers, strict=True):
            self.assertAlmostEqual(got.tolerance, expected.tolerance)
            for field in ('control_rotation', 'helper_rotation', 'helper_location'):
                np.testing.assert_allclose(getattr(got, field), getattr(expected, field), atol=1e-7)
        # A VRD written against Blender names (compiled with $renamebone)
        # imports too.
        blender_named = render_vrd((TWIST,), bones, scale=39.37)
        self.assertEqual(helpers_from_vrd(parse_vrd(blender_named), bones, names, 39.37)[0].control, 'hand')

    def test_import_requires_the_vrd_parents_of_the_armature(self):
        text = render_vrd((TWIST,), rig()).replace('<helper> twist arm', '<helper> twist pelvis')
        with self.assertRaisesRegex(ValueError, "parent is 'pelvis' but the armature has 'arm'"):
            helpers_from_vrd(parse_vrd(text), rig())

    def test_bones_studiomdl_cannot_resolve_are_rejected(self):
        bones = rig()
        cases = (
            (HelperBoneRecipe('twist', 'hand.tip', (HelperTriggerRecipe(),)),
             {}, "'hand.tip' contains '.'"),
            (TWIST, {'arm': 'bip.arm'}, "'bip.arm' contains '.'"),
            (TWIST, {'hand': 'bip hand'}, 'without whitespace'),
            (HelperBoneRecipe('arm', 'pelvis', (HelperTriggerRecipe(),)),
             {}, "control bone 'pelvis' needs a parent"),
            (HelperBoneRecipe('arm', 'hand', (HelperTriggerRecipe(),)),
             {}, "cannot be a child of its helper"),
            (HelperBoneRecipe('twist', 'missing', (HelperTriggerRecipe(),)),
             {}, "'missing' is not in the exported armature"),
        )
        for helper, names, message in cases:
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                render_vrd((helper,), bones, names)
        with self.assertRaisesRegex(ValueError, 'more than one helper'):
            render_vrd((TWIST, TWIST), bones)
        # The hand.tip rename makes the same helper valid.
        render_vrd((cases[0][0],), bones, {'hand.tip': 'hand_tip'})

    def test_trigger_count_is_limited_to_the_engine_maximum(self):
        HelperBoneRecipe('twist', 'hand', (HelperTriggerRecipe(),) * 32)
        with self.assertRaisesRegex(ValueError, 'at most 32'):
            HelperBoneRecipe('twist', 'hand', (HelperTriggerRecipe(),) * 33)
        with self.assertRaisesRegex(ValueError, 'at least one trigger'):
            HelperBoneRecipe('twist', 'hand', ())
        with self.assertRaisesRegex(ValueError, 'positive angle'):
            HelperTriggerRecipe(0.0)

    def test_import_rejects_a_vrd_written_for_another_model_scale(self):
        text = render_vrd((TWIST,), rig(), scale=39.37)
        with self.assertRaisesRegex(ValueError, 'set Model scale to 39.37'):
            helpers_from_vrd(parse_vrd(text), rig(), scale=1.0)


class EvaluateHelperTests(unittest.TestCase):
    """The live preview follows the engine's DoQuatInterpBone."""

    def pose(self, control_delta):
        rests = rest_locals(rig())
        control = rests['hand'].copy()
        control[:3, :3] = control[:3, :3] @ euler_matrix([radians(v) for v in control_delta])
        return rests, evaluate_helper(TWIST, rests['twist'], rests['hand'], control)

    def test_at_a_trigger_the_helper_takes_its_target(self):
        rests, result = self.pose((0, 90, 0))
        self.assertEqual(result.weights, (0.0, 1.0))
        expected = rests['twist'].copy()
        expected[:3, :3] = expected[:3, :3] @ euler_matrix((0, radians(45), 0))
        expected[:3, 3] += rests['twist'][:3, :3] @ (0, 0.1, 0)
        np.testing.assert_allclose(result.matrix, expected, atol=1e-9)

    def test_between_triggers_the_targets_blend_by_tolerance(self):
        # 30 degrees from rest (tolerance 90) and 60 from the posed trigger
        # (tolerance 60): weights 2/3 and 0, so the rest target wins.
        _, result = self.pose((0, 30, 0))
        np.testing.assert_allclose(result.weights, (1.0, 0.0))
        # 50 / 40 degrees away: weights 4/9 and 1/3.
        rests, result = self.pose((0, 50, 0))
        np.testing.assert_allclose(result.weights, (4 / 7, 3 / 7))
        twist = rests['twist']
        angle = np.degrees(np.arccos(np.clip((np.trace(twist[:3, :3].T @ result.matrix[:3, :3]) - 1) / 2, -1, 1)))
        self.assertAlmostEqual(angle, 45 * 3 / 7, delta=0.5)

    def test_out_of_range_falls_back_to_the_first_trigger(self):
        rests, result = self.pose((0, -120, 0))
        self.assertEqual(result.weights, (0.0, 0.0))
        np.testing.assert_allclose(result.matrix, rests['twist'], atol=1e-9)


class ParseVrdTests(unittest.TestCase):
    def test_settings_persist_across_helpers_like_studiomdl(self):
        first, second = parse_vrd(
            '<helper> a pa pc c\n'
            '<basepos> 1 2 3\n'
            '<rotateaxis> 0 0 90\n'
            '<jointorient> 90 0 0\n'
            '<trigger> 45 0 0 0 0 0 0 1 0 0 // comment\n'
            '\n'
            '<helper> b pb pc c\n'
            '<display> 1 1 1 0\n'
            '<trigger> 90 10 20 30 0 0 0 0 0 0\n')
        self.assertEqual((first.bone, first.parent, first.control_parent, first.control), ('a', 'pa', 'pc', 'c'))
        expected = euler_matrix((0, 0, radians(90))) @ euler_matrix((radians(90), 0, 0))
        for helper, position in ((first, (2, 2, 3)), (second, (1, 2, 3))):
            (trigger,) = helper.triggers
            np.testing.assert_allclose(trigger.rotation, expected, atol=1e-12)
            self.assertEqual(trigger.position, position)
        self.assertEqual(second.triggers[0].trigger, (10, 20, 30))

    def test_unsupported_and_malformed_lines_are_rejected(self):
        for text, message in (
                ('<aimconstraint> a b c\n', 'not supported'),
                ('<helper> a b c d\n<display> 1 1 1 50\n'
                 '<trigger> 90 0 0 0 0 0 0 0 0 0\n', 'distance'),
                ('<helper> a b c d\n<trigger> 90 0 0\n', 'needs 10 numbers'),
                ('<helper> a b c\n', 'needs bone, parent'),
                ('<trigger> 90 0 0 0 0 0 0 0 0 0\n', 'before any <helper>'),
                ('<helper> a b c d\n', 'has no triggers'),
                ('<spin> 1\n', 'unknown command')):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                parse_vrd(text)


if __name__ == '__main__':
    unittest.main()
