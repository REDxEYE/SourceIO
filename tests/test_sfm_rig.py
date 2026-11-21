"""SFM IK rig script generation: bone detection, validation and a dry run
against a fake SFM API.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

import numpy as np

from SourceIO.blender_bindings.source1.model_export.sfm_rig import (
    BONE_FIELDS, ConstraintRecipe, FingerRecipe, FKChainRecipe, IKChainRecipe, IKRigRecipe, biped_limbs,
    _keyword_bone, default_pole_positions, detect_biped, render_sfm_rig, resolve_rig, script_name)


def creature():
    """bone -> (parent, rest position) for a rig built from generic parts."""
    return {
        'body': (None, (0, 0, 20)),
        'clav_X': ('body', (2, 0, 24)),
        'up_X': ('clav_X', (6, 0, 24)),
        'twist_X': ('up_X', (9, 0, 24)),
        'low_X': ('up_X', (14, -1, 24)),
        'hand_X': ('low_X', (22, 0, 24)),
        'claw_X': ('hand_X', (25, 0, 24)),
        'hip_M': ('body', (0, 2, 18)),
        'shin_M': ('hip_M', (0, 3, 10)),
        'foot_M': ('shin_M', (0, 2, 2)),
        'toe_M': ('foot_M', (0, 6, 0)),
        'tail_0': ('body', (0, -4, 20)),
        'tail_1': ('tail_0', (0, -8, 19)),
        'tail_2': ('tail_1', (0, -12, 20)),
        'eye': ('body', (0, 3, 26)),
    }


def creature_recipe(**overrides):
    values = dict(
        pelvis='body',
        chains=(
            IKChainRecipe('X', 'up_X', 'hand_X', 'ARM', collar='clav_X'),
            IKChainRecipe('M', 'hip_M', 'foot_M', 'LEG', toe='toe_M', heel_attachment='heel_mid'),
        ),
        fk_chains=(
            FKChainRecipe('tail', 'tail_0', 'tail_2'),
            FKChainRecipe('claw', 'claw_X', group='Claws'),
        ),
        constraints=(
            ConstraintRecipe('ORIENT', 'twist_X', (('rig_hand_X', 0.5), ('up_X', 0.5))),
            ConstraintRecipe('AIM', 'eye', (('rig_hand_X', 1.0),), maintain_offset=False),
            ConstraintRecipe('ROTATION', 'rig_tail_2', name='rig_curl', axis=(0.0, 0.0, 1.0), degrees=120.0,
                             default=0.25),
        ),
    )
    values.update(overrides)
    return IKRigRecipe(**values)


def applied_constraints(sfm):
    """(kind, selected dags, keywords) of every selection based constraint."""
    kinds = ('PointConstraint', 'OrientConstraint', 'ParentConstraint', 'AimConstraint')
    selection, found = [], []
    for name, args, kwargs in sfm.mock_calls:
        if name == 'ClearSelection':
            selection = []
        elif name == 'SelectDag':
            selection.append(getattr(args[0], 'name', args[0]))
        elif name in kinds and not args:
            found.append((name, tuple(selection), kwargs))
    return found


def skeleton(prefix='ValveBiped.Bip01_'):
    """bone -> (parent, rest position) for a ValveBiped style biped with a tail.
    """
    p = prefix
    bones = {p + 'Pelvis': (None, (0, 0, 40)),
             p + 'Spine': (p + 'Pelvis', (0, 0, 44)),
             p + 'Spine1': (p + 'Spine', (0, 0, 48)),
             p + 'Spine2': (p + 'Spine1', (0, 0, 52)),
             p + 'Neck1': (p + 'Spine2', (0, 0, 60)),
             p + 'Head1': (p + 'Neck1', (0, 0, 64)),
             'tail_0': (p + 'Pelvis', (0, -4, 40)),
             'tail_1': ('tail_0', (0, -8, 39)),
             'tail_2': ('tail_1', (0, -12, 40))}
    for side, x in (('R', -1), ('L', 1)):
        bones.update({
            f'{p}{side}_Clavicle': (p + 'Spine2', (2 * x, 0, 56)),
            f'{p}{side}_UpperArm': (f'{p}{side}_Clavicle', (6 * x, 0, 56)),
            f'{p}{side}_Forearm': (f'{p}{side}_UpperArm', (16 * x, -1, 56)),
            f'{p}{side}_Hand': (f'{p}{side}_Forearm', (26 * x, 0, 56)),
            f'{p}{side}_Finger1': (f'{p}{side}_Hand', (29 * x, 0, 56)),
            f'{p}{side}_Thigh': (p + 'Pelvis', (4 * x, 0, 38)),
            # Perfectly straight legs: the knees must fall back to the toe
            # direction.
            f'{p}{side}_Calf': (f'{p}{side}_Thigh', (4 * x, 0, 20)),
            f'{p}{side}_Foot': (f'{p}{side}_Calf', (4 * x, 0, 3)),
            f'{p}{side}_Toe0': (f'{p}{side}_Foot', (4 * x, 6, 0)),
        })
    return bones


def parents_of(bones):
    return {name: parent for name, (parent, _) in bones.items()}


def recipe_for(bones, chains=(), **overrides):
    """The detected biped as a rig of chains named R and L. Limb bones (``hand_r``) and pole targets (``knee_pole_r``)
    can be overridden.
    """
    found = detect_biped(parents_of(bones))
    poles = {
        (kind, side): overrides.pop(f'{word}_pole_{side.lower()}', None)
        for kind, word in (('ARM', 'elbow'), ('LEG', 'knee'))
        for side in 'RL'}
    for name in [name for name in overrides if name in found and name not in BONE_FIELDS]:
        found[name] = overrides.pop(name)
    limbs = tuple(
        IKChainRecipe(name, start, end, kind, collar=collar, toe=toe, pole=poles[kind, name])
        for kind, name, start, end, collar, toe in biped_limbs(found))
    values = {name: found[name] for name in BONE_FIELDS}
    values.update(overrides)
    return IKRigRecipe(chains=limbs + tuple(chains), **values)


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Dag:
    def __init__(self, name, position, rotation=None):
        self.name, self.position = name, position
        # World rotation, columns are the local axes.
        self.rotation = np.identity(3) if rotation is None else rotation

    def GetAbsPosition(self, out):
        out.x, out.y, out.z = self.position

    def GetAbsTransform(self):
        return self.rotation


def _rotate(matrix):
    def apply(vector, rotation, out):
        out.x, out.y, out.z = matrix(rotation) @ (vector.x, vector.y, vector.z)
    return apply


def run_script(text, bones, attachments=(), model=True, rotations=None):
    """Execute a generated script against stand-ins for ``vs``, ``sfm`` and ``sfmUtils``. ``rotations`` maps dag names
    to world rotations.
    """
    rotations = rotations or {}
    dags = {name: Dag(name, position, rotations.get(name)) for name, (_, position) in bones.items()}
    dags['RootTransform'] = Dag('RootTransform', (0, 0, 0), rotations.get('RootTransform'))
    vs = types.ModuleType('vs')
    vs.Vector, vs.Color = Vector, MagicMock()
    vs.mathlib = types.SimpleNamespace(
        VectorRotate=_rotate(lambda rotation: rotation), VectorIRotate=_rotate(lambda rotation: rotation.T))
    vs.REPARENT_LOGS_OVERWRITE, vs.AT_FLOAT = 1, 2
    sfm, utils = MagicMock(), MagicMock()
    sfm.FindDag.side_effect = dags.get
    # Handles take the transform of the bone they are created at.
    targets = {}
    sfm.GetPosition.side_effect = lambda name: list(targets[name].position)
    utils.GetDagFromNameOrObject.side_effect = lambda name: targets[name]
    sfm.GetRotation.side_effect = lambda name: targets[name].rotation

    # Like sfm.FindDag, name lookups only find bones and rig handles with a
    # control.
    findable = set(dags)

    def rig_handle(name, pos=(0, 0, 0), rot=None, posControl=True, rotControl=True):
        targets[name] = Dag(name, tuple(pos), rot)
        if posControl or rotControl:
            findable.add(name)
        return targets[name]

    def reparent(child, parent, *args):
        for item in (child, parent):
            if isinstance(item, str) and not (item.startswith('handle:') or item in findable):
                raise TypeError(f'{item!r} is not a valid dag node or name of a dag node')

    sfm.CreateRigHandle.side_effect = rig_handle
    utils.Parent.side_effect = reparent
    utils.ParentMaintainWorld.side_effect = reparent
    anim_set = sfm.GetCurrentAnimationSet.return_value
    if model:
        anim_set.gameModel.FindAttachment.side_effect = (lambda name: 1 if name in attachments else 0)
    else:
        # As a camera or light animation set in SFM: no such attribute.
        del anim_set.gameModel

    def create(name, target=None, *args, **kwargs):
        target = targets.get(target, target)
        if isinstance(target, Dag):
            targets[name] = target
        findable.add(name)
        return f'handle:{name}'

    for creator in (utils.CreateConstrainedHandle, utils.CreateOffsetHandle, utils.CreateHandleAt):
        creator.side_effect = create
    utils.CreateControlledValue.return_value = (MagicMock(), MagicMock())
    previous = sys.modules.get('vs')
    sys.modules['vs'] = vs
    try:
        exec(compile(text, 'rig.py', 'exec'), {'sfm': sfm, 'sfmUtils': utils, '__name__': 'rig'})
    finally:
        if previous is None:
            del sys.modules['vs']
        else:
            sys.modules['vs'] = previous
    return sfm, utils


class DetectionTests(unittest.TestCase):
    def test_valve_biped(self):
        found = detect_biped(parents_of(skeleton()))
        self.assertEqual(found['pelvis'], 'ValveBiped.Bip01_Pelvis')
        self.assertEqual(found['chest'], 'ValveBiped.Bip01_Spine2')
        self.assertEqual(found['head'], 'ValveBiped.Bip01_Head1')
        self.assertEqual(found['collar_r'], 'ValveBiped.Bip01_R_Clavicle')
        self.assertEqual(found['upper_arm_l'], 'ValveBiped.Bip01_L_UpperArm')
        self.assertEqual(found['hand_l'], 'ValveBiped.Bip01_L_Hand')
        self.assertEqual(found['thigh_r'], 'ValveBiped.Bip01_R_Thigh')
        self.assertEqual(found['toe_r'], 'ValveBiped.Bip01_R_Toe0')

    def test_other_naming_styles(self):
        for names in (
                dict(pelvis='Hips', chest='Spine2', head='Head', collar='{S}Shoulder', upper='{S}Arm',
                     lower='{S}ForeArm', hand='{S}Hand', thigh='{S}UpLeg', calf='{S}Leg', foot='{S}Foot',
                     toe='{S}ToeBase', sides=('Left', 'Right')),  # Mixamo
                dict(pelvis='pelvis', chest='chest', head='head', collar='{S}Collar', upper='{S}Shldr',
                     lower='{S}ForeArm', hand='{S}Hand', thigh='{S}Thigh', calf='{S}Shin', foot='{S}Foot',
                     toe='{S}Toe', sides=('l', 'r')),  # Daz
                dict(pelvis='bip_pelvis', chest='bip_spine_3', head='bip_head', collar='bip_collar_{S}',
                     upper='bip_upperArm_{S}', lower='bip_lowerArm_{S}', hand='bip_hand_{S}', thigh='bip_hip_{S}',
                     calf='bip_knee_{S}', foot='bip_foot_{S}', toe='bip_toe_{S}', sides=('L', 'R')),  # TF2
                dict(pelvis='root', chest='spine.003', head='head', collar='shoulder.{S}', upper='upper_arm.{S}',
                     lower='forearm.{S}', hand='hand.{S}', thigh='thigh.{S}', calf='shin.{S}', foot='foot.{S}',
                     toe='toe.{S}', sides=('L', 'R')),  # Blender
        ):
            parents = {names['pelvis']: None, names['chest']: names['pelvis'], names['head']: names['chest']}
            for side in names['sides']:
                bone = {key: value.replace('{S}', side) for key, value in names.items() if key != 'sides'}
                parents.update({
                    bone['collar']: bone['chest'],
                    bone['upper']: bone['collar'],
                    bone['lower']: bone['upper'],
                    bone['hand']: bone['lower'],
                    bone['hand'] + 'Thumb1': bone['hand'],
                    bone['thigh']: bone['pelvis'],
                    bone['calf']: bone['thigh'],
                    bone['foot']: bone['calf'],
                    bone['toe']: bone['foot']})
            found = detect_biped(parents)
            left = names['sides'][0]
            with self.subTest(pelvis=names['pelvis']):
                self.assertEqual(found['pelvis'], names['pelvis'])
                self.assertEqual(found['chest'], names['chest'])
                self.assertEqual(found['head'], names['head'])
                for field in ('collar', 'upper', 'hand', 'thigh', 'foot', 'toe'):
                    key = {'upper': 'upper_arm'}.get(field, field) + '_l'
                    self.assertEqual(found[key], names[field].replace('{S}', left))
                self.assertTrue(all(found.values()))

    def test_rigify_control_bones_do_not_hide_the_real_limbs(self):
        parents = {'root': None, 'spine': 'root', 'chest': 'spine', 'head': 'chest',
                   'MCH-hand_ik_parent.L': 'root', 'MCH-hand_ik_parent.R': 'root',
                   'ctrl_foot.L': 'root'}
        for side in 'LR':
            parents.update({
                f'hand_ik.{side}': 'root', f'foot_ik.{side}': 'root', f'IKFoot.{side}': 'root',
                f'DEF-shoulder.{side}': 'chest', f'DEF-upper_arm.{side}': f'DEF-shoulder.{side}',
                f'DEF-forearm.{side}': f'DEF-upper_arm.{side}', f'DEF-hand.{side}': f'DEF-forearm.{side}',
                f'DEF-thigh.{side}': 'root', f'DEF-shin.{side}': f'DEF-thigh.{side}',
                f'DEF-foot.{side}': f'DEF-shin.{side}', f'DEF-toe.{side}': f'DEF-foot.{side}'})
        self.assertEqual(_keyword_bone({'root': None, 'kick_hand.L': 'root', 'hand_ik.L': 'root'}, 'hand', 'L'),
                         'kick_hand.L')
        found = detect_biped(parents)
        for side in 'LR':
            suffix = side.lower()
            self.assertEqual(found[f'hand_{suffix}'], f'DEF-hand.{side}')
            self.assertEqual(found[f'upper_arm_{suffix}'], f'DEF-upper_arm.{side}')
            self.assertEqual(found[f'foot_{suffix}'], f'DEF-foot.{side}')
            self.assertEqual(found[f'thigh_{suffix}'], f'DEF-thigh.{side}')
            self.assertEqual(found[f'toe_{suffix}'], f'DEF-toe.{side}')

    def test_no_limbs(self):
        found = detect_biped({'root': None, 'lid': 'root'})
        self.assertFalse(any(found.values()))

    def test_biped_limbs_skip_incomplete_limbs(self):
        found = detect_biped(parents_of(skeleton()))
        found['hand_l'] = ''
        found['toe_r'] = ''
        p = 'ValveBiped.Bip01_'
        self.assertEqual(biped_limbs(found), (
            ('ARM', 'R', p + 'R_UpperArm', p + 'R_Hand', p + 'R_Clavicle', ''),
            ('LEG', 'R', p + 'R_Thigh', p + 'R_Foot', '', ''),
            ('LEG', 'L', p + 'L_Thigh', p + 'L_Foot', '', p + 'L_Toe0')))


class ResolveTests(unittest.TestCase):
    def test_layout(self):
        bones = skeleton()
        rig = resolve_rig(recipe_for(bones, chains=(IKChainRecipe('tail', 'tail_0', 'tail_2'),)), parents_of(bones))
        handles = {handle: (bone, parent) for handle, bone, parent, _ in rig.fk}
        self.assertEqual(handles['rig_spine_0'], ('ValveBiped.Bip01_Spine', 'rig_pelvis'))
        self.assertEqual(handles['rig_chest'], ('ValveBiped.Bip01_Spine2', 'rig_spine_1'))
        self.assertEqual(handles['rig_neck'], ('ValveBiped.Bip01_Neck1', 'rig_chest'))
        self.assertEqual(handles['rig_head'][1], 'rig_neck')
        self.assertEqual(handles['rig_collar_L'][1], 'rig_chest')
        self.assertEqual(handles['rig_toe_R'][1], 'rig_footHelper_R')
        limbs = {limb[2]: limb for limb in rig.limbs}
        self.assertEqual(limbs['rig_hand_R'][3:9], (
            'rig_elbow_R', 'ValveBiped.Bip01_R_UpperArm', 'ValveBiped.Bip01_R_Forearm', 'ValveBiped.Bip01_R_Hand', '',
            'rig_collar_R'))
        self.assertEqual(limbs['rig_foot_L'][7:9], ('rig_toe_L', 'rig_foot_L'))
        self.assertEqual(limbs['rig_tail'][3:9], ('rig_tail_pole', 'tail_0', 'tail_1', 'tail_2', '', 'rig_pelvis'))

    def test_pelvis_only_and_legs_only(self):
        bones = skeleton()
        rig = resolve_rig(IKRigRecipe(pelvis='ValveBiped.Bip01_Pelvis'), parents_of(bones))
        self.assertEqual([item[0] for item in rig.fk], ['rig_pelvis'])
        rig = resolve_rig(
            IKRigRecipe(pelvis='ValveBiped.Bip01_Pelvis', chains=(
                IKChainRecipe('R', 'ValveBiped.Bip01_R_Thigh', 'ValveBiped.Bip01_R_Foot', 'LEG'),)),
            parents_of(bones))
        self.assertEqual(rig.helpers, ())
        self.assertEqual(len(rig.limbs), 1)

    def test_generic_parts(self):
        rig = resolve_rig(creature_recipe(), parents_of(creature()))
        handles = {handle: (bone, parent, group) for handle, bone, parent, group in rig.fk}
        self.assertEqual(handles['rig_tail_0'], ('tail_0', 'rig_pelvis', 'RigExtra'))
        self.assertEqual(handles['rig_tail_2'], ('tail_2', 'rig_tail_1', 'RigExtra'))
        # A single bone chain has no index and hangs from the IK handle.
        self.assertEqual(handles['rig_claw'], ('claw_X', 'rig_hand_X', 'Claws'))
        self.assertEqual(handles['rig_collar_X'], ('clav_X', 'rig_pelvis', 'Arm_X'))
        limbs = {limb[2]: limb for limb in rig.limbs}
        self.assertEqual(limbs['rig_hand_X'], (
            'arm', 'X', 'rig_hand_X', 'rig_elbow_X', 'up_X', 'low_X', 'hand_X', '', 'rig_collar_X', 'Arm_X', ''))
        self.assertEqual(limbs['rig_foot_M'], (
            'leg', 'M', 'rig_foot_M', 'rig_knee_M', 'hip_M', 'shin_M', 'foot_M', 'rig_toe_M', 'rig_foot_M', 'Leg_M',
            'heel_mid'))
        self.assertEqual(rig.groups, (
            ('RigBody', '', 'top'),
            ('RigLegs', '', 'top'), ('Leg_M', 'RigLegs', 'top'),
            ('RigArms', '', 'top'), ('Arm_X', 'RigArms', 'top'),
            ('RigExtra', '', 'top'), ('Claws', '', 'top')))
        orient, aim, rotation = rig.constraints
        self.assertEqual(orient[:4], (
            'orient', ('bone', 'twist_X'), ((('handle', 'rig_hand_X'), 0.5), (('bone', 'up_X'), 0.5)), True))
        self.assertEqual(aim[:4], (
            'aim', ('bone', 'eye'), ((('handle', 'rig_hand_X'), 1.0),), False))
        self.assertEqual(rotation, (
            'rotation', ('handle', 'rig_tail_2'), (), True, 'rig_curl', (0.0, 0.0, 1.0), 120.0, 0.25, 'RigExtra'))
        self.assertEqual(rig.bones()[-2:], ('twist_X', 'eye'))

    def test_generic_part_rejections(self):
        parents = parents_of(creature())
        with self.assertRaisesRegex(ValueError, 'solved by IK'):
            resolve_rig(creature_recipe(fk_chains=(FKChainRecipe('bad', 'twist_X'),)), parents)
        with self.assertRaisesRegex(ValueError, 'descendant'):
            resolve_rig(creature_recipe(fk_chains=(FKChainRecipe('bad', 'tail_1', 'eye'),)), parents)
        slider = ConstraintRecipe('ROTATION', 'eye', name='rig_look')
        with self.assertRaisesRegex(ValueError, 'slider names.*rig_look'):
            resolve_rig(creature_recipe(constraints=(slider, slider)), parents)
        with self.assertRaisesRegex(ValueError, "target bone 'rig_nope'"):
            resolve_rig(creature_recipe(constraints=(ConstraintRecipe('POINT', 'eye', (('rig_nope', 1.0),)),)), parents)
        with self.assertRaisesRegex(ValueError, 'at least one target'):
            ConstraintRecipe('POINT', 'eye')
        with self.assertRaisesRegex(ValueError, 'distinct'):
            ConstraintRecipe('PARENT', 'eye', (('eye', 1.0),))
        with self.assertRaisesRegex(ValueError, 'positive'):
            ConstraintRecipe('PARENT', 'eye', (('body', 0.0),))
        with self.assertRaisesRegex(ValueError, 'nonzero axis'):
            ConstraintRecipe('ROTATION', 'eye', name='rig_look', axis=(0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, 'not by targets'):
            ConstraintRecipe('ROTATION', 'eye', (('body', 1.0),), name='rig_look')
        with self.assertRaisesRegex(ValueError, 'only arm chains'):
            IKChainRecipe('a', 'up_X', 'hand_X', collar='clav_X')
        with self.assertRaisesRegex(ValueError, 'only leg chains'):
            IKChainRecipe('a', 'hip_M', 'foot_M', 'ARM', toe='toe_M')

    def test_rejections(self):
        bones = skeleton()
        parents = parents_of(bones)
        with self.assertRaisesRegex(ValueError, 'not in the exported armature'):
            resolve_rig(recipe_for(bones, head='Nope'), parents)
        with self.assertRaisesRegex(ValueError, 'grandchild'):
            resolve_rig(recipe_for(bones, hand_r='ValveBiped.Bip01_R_Forearm'), parents)
        with self.assertRaisesRegex(ValueError, 'descendant'):
            resolve_rig(recipe_for(bones, chest='tail_1'), parents)
        with self.assertRaisesRegex(ValueError, 'more than one control'):
            resolve_rig(recipe_for(bones, chains=(IKChainRecipe(
                'arm2', 'ValveBiped.Bip01_R_UpperArm', 'ValveBiped.Bip01_R_Hand'),)), parents)
        with self.assertRaisesRegex(ValueError, 'used twice'):
            resolve_rig(recipe_for(bones, chains=(IKChainRecipe('hand_R', 'tail_0', 'tail_2'),)), parents)
        with self.assertRaisesRegex(ValueError, 'letters'):
            IKChainRecipe('bad name', 'a', 'b')
        with self.assertRaisesRegex(ValueError, 'Pelvis'):
            IKRigRecipe(pelvis='')


class RenderTests(unittest.TestCase):
    def test_script_name(self):
        recipe = IKRigRecipe(pelvis='p')
        self.assertEqual(script_name(recipe, 'models/props/hero_v2.mdl'), 'rig_ik_props__hero_v2')
        self.assertEqual(script_name(recipe, 'models\\props\\hero.mdl'), 'rig_ik_props__hero')
        self.assertEqual(script_name(IKRigRecipe(pelvis='p', script_name='rig_hero'), 'models/a.mdl'), 'rig_hero')

    def test_script_name_never_collides_after_sanitizing_or_truncating(self):
        recipe = IKRigRecipe(pelvis='p')
        paths = ('models/a/hero.mdl', 'models/b/hero.mdl', 'models/my char/hero.mdl', 'models/my_char/hero.mdl',
                 'models/my-char/hero.mdl', 'models/a/b_c.mdl', 'models/a_b/c.mdl', 'models/a_b_c.mdl',
                 'models/a/b/c.mdl', 'models/a__b.mdl', 'models/a/_b.mdl', 'models/a_/b.mdl', 'models/a.mdl',
                 'models/' + 'long/' * 30 + 'a.mdl', 'models/' + 'long/' * 30 + 'b.mdl')
        names = [script_name(recipe, path) for path in paths]
        self.assertEqual(len(set(names)), len(paths), names)
        for name in names:
            self.assertRegex(name, r'[A-Za-z0-9_]{1,64}\Z')
        self.assertEqual(script_name(recipe, 'models/my char/hero.mdl'), script_name(recipe, 'models/my char/hero.mdl'))

    def test_renames_and_python2_subset(self):
        bones = skeleton()
        text = render_sfm_rig(recipe_for(bones), parents_of(bones),
                              {'ValveBiped.Bip01_Pelvis': 'bip_pelvis', 'ValveBiped.Bip01_R_Hand': 'кисть "R"'},
                              'models/hero.mdl')
        self.assertIn('("rig_pelvis", "bip_pelvis", "rig_root", "RigBody")', text)
        self.assertIn(r'"кисть \"R\""', text)
        self.assertNotIn('"ValveBiped.Bip01_Pelvis"', text)
        compile(text, 'rig.py', 'exec')
        # Constructs Python 2.7 (SFM) cannot parse.
        for pattern in (r'\bf["\']', r'\bnonlocal\b', r' -> ', r':=', r'\basync\b'):
            self.assertNotRegex(text, pattern)
        with self.assertRaisesRegex(ValueError, 'merge'):
            render_sfm_rig(recipe_for(bones), parents_of(bones), {'ValveBiped.Bip01_Pelvis': 'ValveBiped.Bip01_Spine'})

    def test_dry_run(self):
        bones = skeleton()
        recipe = recipe_for(bones, chains=(IKChainRecipe('tail', 'tail_0', 'tail_2'),))
        sfm, utils = run_script(render_sfm_rig(recipe, parents_of(bones)), bones)
        sfm.BeginRig.assert_called_once()
        sfm.EndRig.assert_called_once()
        sfm.GenerateSamples.assert_called_once()
        limbs = {call.args[1]: call.args for call in utils.BuildArmLeg.call_args_list}
        self.assertEqual(set(limbs), {
            'handle:rig_hand_R', 'handle:rig_hand_L', 'handle:rig_foot_R', 'handle:rig_foot_L', 'handle:rig_tail'})
        self.assertEqual(limbs['handle:rig_foot_R'][0], 'handle:rig_knee_R')
        self.assertEqual(limbs['handle:rig_foot_R'][2].name, 'ValveBiped.Bip01_R_Thigh')
        self.assertEqual(limbs['handle:rig_foot_R'][3].name, 'ValveBiped.Bip01_R_Foot')
        offsets = {call.args[0]: call.args[2] for call in utils.CreateOffsetHandle.call_args_list}
        knee, elbow, tail = (offsets['rig_knee_R'], offsets['rig_elbow_R'], offsets['rig_tail_pole'])
        # straight leg: towards the toes
        self.assertGreater(knee.y, 0)
        self.assertAlmostEqual(knee.x, 0)
        # bent arm: where the elbow already points
        self.assertLess(elbow.y, 0)
        # generic chain: its own bend
        self.assertLess(tail.z, 0)
        # The pelvis bone follows the hips handle, every other FK bone its
        # own handle.
        driven = {call.args[1].name: call.args[0] for call in utils.CreatePointOrientConstraint.call_args_list}
        self.assertEqual(driven['ValveBiped.Bip01_Pelvis'], 'handle:rig_hips')
        self.assertEqual(driven['ValveBiped.Bip01_Head1'], 'handle:rig_head')
        self.assertEqual(driven['RootTransform'], 'handle:rig_root')
        # no heel attachments: no foot roll
        utils.CreateControlledValue.assert_not_called()
        parents = {call.args[0]: call.args[1] for call in utils.ParentMaintainWorld.call_args_list}
        self.assertEqual(parents['handle:rig_knee_L'], 'handle:rig_foot_L')
        self.assertEqual(parents['handle:rig_elbow_L'], 'handle:rig_collar_L')
        self.assertEqual(parents['handle:rig_hand_L'], 'handle:rig_root')
        self.assertEqual(parents['handle:rig_toe_L'], 'handle:rig_footHelper_L')

    def test_foot_roll(self):
        bones = skeleton()
        text = render_sfm_rig(recipe_for(bones), parents_of(bones))
        _, utils = run_script(text, bones, attachments=('pvt_heel_R',))
        self.assertEqual([call.args[0] for call in utils.CreateControlledValue.call_args_list], ['rig_footRoll_R'])
        targets = {call.args[0]: call.args[1] for call in utils.BuildArmLeg.call_args_list}
        self.assertEqual(targets['handle:rig_knee_R'], 'handle:rig_footIK_R')
        self.assertEqual(targets['handle:rig_knee_L'], 'handle:rig_foot_L')
        _, utils = run_script(
            render_sfm_rig(recipe_for(bones, foot_roll=False), parents_of(bones)), bones, attachments=('pvt_heel_R',))
        utils.CreateControlledValue.assert_not_called()

    def test_foot_roll_axis_is_the_characters_left_in_the_foot_frame(self):
        bones = skeleton()
        text = render_sfm_rig(recipe_for(bones), parents_of(bones))
        foot = 'ValveBiped.Bip01_L_Foot'
        # The foot's local X points along world +Y, not sideways.
        turned = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        lying = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        # Toes are 6 units along +Y and 3 down from the ankle. With Z up the
        # character's left is world -X, which is local +Y of the foot; with
        # the model's up along world -Y it is world +X instead.
        for root, expected in ((None, (0, 1, 0)), (lying, (0, -1, 0))):
            rotations = {foot: turned}
            if root is not None:
                rotations['RootTransform'] = root
            _, utils = run_script(text, bones, attachments=('pvt_heel_L',), rotations=rotations)
            axes = {call.args[0]: call.args[1] for call in utils.CreateRotationConstraint.call_args_list}
            for name in ('rotationConstraint_toe_L', 'rotationConstraint_heel_L'):
                axis = axes[name]
                np.testing.assert_allclose((axis.x, axis.y, axis.z), expected, atol=1e-9)

    @staticmethod
    def hand():
        """The biped with a three bone index finger on the left hand."""
        bones = skeleton()
        parent = 'ValveBiped.Bip01_L_Hand'
        for index, x in enumerate((28, 30, 32)):
            bones[f'index_{index}'] = (parent, (x, 0, 56))
            parent = f'index_{index}'
        return bones

    def test_finger_gets_posable_controls_and_one_curl_slider(self):
        bones = self.hand()
        recipe = recipe_for(bones, fingers=(FingerRecipe('index', 'index_0', 'index_2', 'Fingers', (0.0, 0.0, 2.0)),))
        rig = resolve_rig(recipe, parents_of(bones))
        fk = {handle: (bone, parent) for handle, bone, parent, _ in rig.fk}
        self.assertEqual(fk['rig_index_0'], ('index_0', 'rig_hand_L'))
        self.assertEqual(fk['rig_index_2'], ('index_2', 'rig_index_1'))
        # 90 degrees at 1 from a rest value of 0.2.
        self.assertEqual(rig.curls, (
            ('rig_curl_index', 'Fingers', 112.5, 0.2, (0.0, 0.0, 1.0), ('rig_index_0', 'rig_index_1', 'rig_index_2')),))

        # The finger bones' local X runs along world +Y.
        turned = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        recipe = recipe_for(bones, fingers=(FingerRecipe('index', 'index_0', 'index_2', 'Fingers'),))
        text = render_sfm_rig(recipe, parents_of(bones))
        _, utils = run_script(text, bones, rotations={f'index_{index}': turned for index in range(3)})
        self.assertIn('rig_curl_index', [call.args[0] for call in utils.CreateControlledValue.call_args_list])
        expressions = {call.args[0]: call.args[1] for call in utils.CreateExpression.call_args_list}
        self.assertEqual(expressions['expr_rig_curl_index'], '( slider - 0.2 ) * 112.5')

        # A curl helper goes between each finger control and its parent.
        def names(calls):
            return [tuple(getattr(arg, 'name', arg) for arg in call.args[:2]) for call in calls]

        parents = names(utils.Parent.call_args_list)
        self.assertIn(('rig_index_0_curl', 'handle:rig_hand_L'), parents)
        self.assertIn(('rig_index_1_curl', 'handle:rig_index_0'), parents)
        self.assertIn(('handle:rig_index_2', 'rig_index_2_curl'), names(utils.ParentMaintainWorld.call_args_list))
        # Helpers take their parent's orientation (the hand's is the world's
        # here), so the bone's X is the helper's +Y.
        axes = {call.args[0]: call.args[1] for call in utils.CreateRotationConstraint.call_args_list}
        axis = axes['rotationConstraint_rig_index_0_curl']
        np.testing.assert_allclose((axis.x, axis.y, axis.z), (0, 1, 0), atol=1e-9)
        # The finger controls stay in their group for posing, and still
        # drive the bones.
        grouped = {getattr(dag, 'name', dag) for call in utils.AddDagControlsToGroup.call_args_list
                   for dag in call.args[1:]}
        self.assertIn('handle:rig_index_1', grouped)
        driven = {call.args[1].name: call.args[0] for call in utils.CreatePointOrientConstraint.call_args_list}
        self.assertEqual(driven['index_1'], 'handle:rig_index_1')

    def test_finger_rejections(self):
        bones = self.hand()
        parents = parents_of(bones)
        with self.assertRaisesRegex(ValueError, 'slider names.*rig_curl_a'):
            resolve_rig(recipe_for(bones, fingers=(FingerRecipe('a', 'index_0'),), constraints=(
                ConstraintRecipe('ROTATION', 'index_2', name='rig_curl_a'),)), parents)
        with self.assertRaisesRegex(ValueError, 'more than one control'):
            resolve_rig(recipe_for(bones, fingers=(
                FingerRecipe('a', 'index_0'), FingerRecipe('b', 'index_0'))), parents)
        # A copied node with only its bones changed.
        with self.assertRaisesRegex(ValueError, 'own names; used more than once: a$'):
            resolve_rig(recipe_for(bones, fingers=(
                FingerRecipe('a', 'index_0'), FingerRecipe('a', 'index_1', 'index_2')),
                fk_chains=(FKChainRecipe('tail', 'tail_0'),)), parents)
        with self.assertRaisesRegex(ValueError, 'nonzero curl axis'):
            FingerRecipe('a', 'index_0', axis=(0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, 'below 1'):
            FingerRecipe('a', 'index_0', default=1.0)

    def test_driven_bones_are_moved_to_a_hidden_group(self):
        bones = skeleton()
        recipe = recipe_for(bones, hidden_bones=('tail_0', 'not_in_model'))
        text = render_sfm_rig(recipe, parents_of(bones), {'tail_0': 'tail_root'})
        self.assertIn('HIDDEN_BONES = ("tail_root",', text)
        bones['tail_root'] = bones['tail_0']
        sfm, utils = run_script(text, bones)
        root = (sfm.GetCurrentAnimationSet.return_value.GetRootControlGroup.return_value)
        moved = [call.args for call in utils.AddDagControlsToGroup.call_args_list
                 if call.args[0] is root.FindChildByName.return_value]
        names = {dag.name for dag in moved[-1][1:]}
        self.assertIn('ValveBiped.Bip01_R_Forearm', names)
        self.assertIn('ValveBiped.Bip01_Head1', names)
        self.assertIn('tail_root', names)
        self.assertNotIn('RootTransform', names)
        self.assertNotIn('ValveBiped.Bip01_R_Finger1', names)
        text = render_sfm_rig(recipe_for(bones, hide_driven_bones=False), parents_of(bones))
        _, utils = run_script(text, bones)
        self.assertFalse(any(isinstance(arg, Dag) for call in utils.AddDagControlsToGroup.call_args_list
                             for arg in call.args))

    def test_animation_set_without_a_model_aborts_before_the_rig(self):
        bones = skeleton()
        text = render_sfm_rig(recipe_for(bones), parents_of(bones))
        sfm, _ = run_script(text, bones, model=False)
        sfm.BeginRig.assert_not_called()

    def test_missing_bone_aborts_cleanly(self):
        bones = skeleton()
        text = render_sfm_rig(recipe_for(bones), parents_of(bones))
        del bones['ValveBiped.Bip01_R_Toe0']  # e.g. collapsed by studiomdl
        sfm, utils = run_script(text, bones)
        sfm.EndRig.assert_called_once()
        utils.CreateConstrainedHandle.assert_not_called()

    def test_generic_parts_dry_run(self):
        bones = creature()
        text = render_sfm_rig(creature_recipe(), parents_of(bones), {'twist_X': 'twist_out'})
        self.assertIn('("bone", "twist_out")', text)
        bones['twist_out'] = bones.pop('twist_X')
        sfm, utils = run_script(text, bones, attachments=('heel_mid',))
        sfm.EndRig.assert_called_once()

        targets = {call.args[0]: call.args[1:] for call in utils.BuildArmLeg.call_args_list}
        self.assertEqual(targets['handle:rig_elbow_X'][0], 'handle:rig_hand_X')
        # The custom heel attachment exists, so the leg follows the foot roll.
        self.assertEqual(targets['handle:rig_knee_M'][0], 'handle:rig_footIK_M')
        driven = {call.args[1].name: call.args[0] for call in utils.CreatePointOrientConstraint.call_args_list}
        self.assertEqual(driven['tail_1'], 'handle:rig_tail_1')
        self.assertEqual(driven['claw_X'], 'handle:rig_claw')
        self.assertEqual(driven['clav_X'], 'handle:rig_collar_X')
        parents = {call.args[0]: call.args[1] for call in utils.ParentMaintainWorld.call_args_list}
        self.assertEqual(parents['handle:rig_claw'], 'handle:rig_hand_X')
        self.assertEqual(parents['handle:rig_tail_2'], 'handle:rig_tail_1')

        self.assertEqual(applied_constraints(sfm), [
            ('OrientConstraint', ('handle:rig_hand_X', 'twist_out'), dict(mo=True, w=0.5)),
            ('OrientConstraint', ('up_X', 'twist_out'), dict(mo=True, w=0.5)),
            ('AimConstraint', ('handle:rig_hand_X', 'eye'), dict(mo=False)),
        ])
        sliders = [call.args[0] for call in utils.CreateControlledValue.call_args_list]
        self.assertEqual(sliders, ['rig_footRoll_M', 'rig_curl'])
        self.assertEqual(utils.CreateControlledValue.call_args.args[3], 0.25)
        expressions = {call.args[0]: call.args[1] for call in utils.CreateExpression.call_args_list}
        self.assertEqual(expressions['expr_rig_curl'], '( slider - 0.25 ) * 120.0')
        name, axis, slave = utils.CreateRotationConstraint.call_args.args[:3]
        self.assertEqual((name, slave), ('rotationConstraint_rig_curl', 'handle:rig_tail_2'))
        self.assertEqual((axis.x, axis.y, axis.z), (0.0, 0.0, 1.0))

        # Bones constrained directly are hidden with the handle driven ones.
        root = sfm.GetCurrentAnimationSet().GetRootControlGroup()
        hidden = [call.args[1:] for call in utils.AddDagControlsToGroup.call_args_list
                  if call.args[0] is root.FindChildByName.return_value][-1]
        names = {dag.name for dag in hidden}
        self.assertLessEqual({'twist_out', 'eye', 'low_X', 'tail_2'}, names)
        self.assertNotIn('RootTransform', names)


class PoleTargetTests(unittest.TestCase):
    CALF = 'ValveBiped.Bip01_R_Calf'

    @staticmethod
    def positions(bones):
        return {name: position for name, (_, position) in bones.items()}

    @staticmethod
    def moved(point):
        """A point after a turn, a uniform scale and a shift: the model as compiled with another up axis and scale and
        placed in a shot.
        """
        x, y, z = point
        cos, sin = 0.8, 0.6
        x, y = x * cos - y * sin, x * sin + y * cos   # around Z
        y, z = -z, y                                  # Z up to Y up
        return (2.5 * x + 100, 2.5 * y - 40, 2.5 * z + 7)

    def test_default_positions_follow_the_bend_or_the_toes(self):
        bones = skeleton()
        positions = {name: position for name, (_, position) in bones.items()}
        elbow_r, _, knee_r, knee_l = default_pole_positions(recipe_for(bones).chains, parents_of(bones), positions)
        # The elbow is already bent backwards.
        self.assertAlmostEqual(elbow_r[0], -16)
        self.assertLess(elbow_r[1], -5)
        self.assertAlmostEqual(elbow_r[2], 56)
        # Straight legs bend towards the toes, 40% of the leg length away.
        for knee, expected in ((knee_r, (-4, 14, 20)), (knee_l, (4, 14, 20))):
            for value, wanted in zip(knee, expected):
                self.assertAlmostEqual(value, wanted)

    def test_authored_pole_follows_the_bones_of_the_model(self):
        bones = skeleton()
        knee, elbow = (-9.0, 10.0, 23.0), (30.0, -12.0, 50.0)
        recipe = recipe_for(bones, knee_pole_r=knee, elbow_pole_l=elbow)
        text = render_sfm_rig(recipe, parents_of(bones), bone_positions=self.positions(bones))
        in_shot = {name: (parent, self.moved(position)) for name, (parent, position) in bones.items()}
        for model in (bones, in_shot):
            _, utils = run_script(text, model)
            calls = {call.args[0]: call for call in utils.CreateOffsetHandle.call_args_list}
            for handle, middle, pole in (
                    ('rig_knee_R', self.CALF, knee),
                    ('rig_elbow_L', 'ValveBiped.Bip01_L_Forearm', elbow)):
                dag, offset = calls[handle].args[1:3]
                self.assertEqual(dag.name, middle)
                self.assertNotIn('refSpaceDag', calls[handle].kwargs)
                wanted = pole if model is bones else self.moved(pole)
                found = (dag.position[0] + offset.x, dag.position[1] + offset.y, dag.position[2] + offset.z)
                for value, expected in zip(found, wanted):
                    self.assertAlmostEqual(value, expected, places=6)
            # The other limbs still derive their pole from the pose.
            auto = calls['rig_knee_L'].args[2]
            self.assertAlmostEqual(
                (auto.x ** 2 + auto.y ** 2 + auto.z ** 2) ** 0.5,
                14.0 if model is bones else 35.0)

    def test_rejections(self):
        bones = skeleton()
        parents = parents_of(bones)
        recipe = recipe_for(bones, knee_pole_r=(-4.0, 10.0, 20.0))
        with self.assertRaisesRegex(ValueError, 'no rest pose'):
            render_sfm_rig(recipe, parents)
        positions = self.positions(bones)
        on_joint = recipe_for(bones, knee_pole_r=(-4.0, 0.0, 20.0))
        with self.assertRaisesRegex(ValueError, 'sits on the joint'):
            render_sfm_rig(on_joint, parents, bone_positions=positions)
        line = {'p': None, 'a': 'p', 'b': 'a', 'c': 'b'}
        straight = IKRigRecipe(pelvis='p', chains=(IKChainRecipe('x', 'a', 'c', pole=(0.0, 1.0, 1.0)),))
        with self.assertRaisesRegex(ValueError, 'lies on one line'):
            render_sfm_rig(straight, line, bone_positions={
                'p': (0, 0, -1), 'a': (0, 0, 0), 'b': (0, 0, 1), 'c': (0, 0, 2)})
        with self.assertRaisesRegex(ValueError, 'three finite numbers'):
            IKChainRecipe('a', 'b', 'c', pole=(1.0, 2.0))


if __name__ == '__main__':
    unittest.main()
