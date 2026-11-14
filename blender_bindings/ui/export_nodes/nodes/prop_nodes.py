"""Prop-only graph nodes. Recipes are inert; the export operator owns every
write.

``resolve(upstream_output_socket)`` must return that socket's
node.recipe(resolve), with reroutes/cycles handled by the graph evaluator.
Multi-input link order is QC order. No legacy node output is interpreted
implicitly.
"""
from math import degrees
from pathlib import PurePosixPath
import re

import bpy
from bpy.types import Node, NodeSocket, Operator, PropertyGroup
from mathutils import Euler, Matrix

from SourceIO.blender_bindings.source1.model_export.dmx_types import (
    VertexAnimationRange)
from SourceIO.blender_bindings.source1.model_export.geometry_rig import (
    resolve_geometry_rig)
from SourceIO.blender_bindings.source1.model_export.material_assets import (
    MaterialRecipe)
from SourceIO.blender_bindings.source1.model_export.prop_qc import (
    BLANK, AttachmentRecipe, BodygroupRecipe, CollisionRecipe,
    FlexControllerRecipe, FlexRuleSetRecipe, GeometryRecipe, HitboxRecipe,
    IdleRecipe, LODRecipe, ModelEntryRecipe, ModelRecipe, RawQcRecipe,
    SkinFamilyRecipe, BoneRenameRecipe, parse_bone_table,
)
from SourceIO.blender_bindings.source1.model_export.sfm_rig import (
    IKRigRecipe)
from .base_node import SourceIOModelTreeNode as _PropNode


def _linked(socket, resolve):
    if resolve is None:
        raise ValueError(
            "Linked prop sockets require a resolve(output_socket) callback")
    # Blender displays multi-input links from the highest sort ID downward.
    if socket.is_multi_input:
        links = sorted(socket.links, key=lambda link: link.multi_input_sort_id,
                       reverse=True)
    else:
        links = socket.links
    return tuple(resolve(link.from_socket) for link in links)


def _one(node, name, resolve, expected, required=False):
    values = _linked(node.inputs[name], resolve)
    if len(values) > 1:
        raise ValueError(f"{node.name}: {name} accepts only one link")
    if not values:
        if required:
            raise ValueError(f"{node.name}: {name} is required")
        return None
    if not isinstance(values[0], expected):
        raise ValueError(f"{node.name}: {name} has the wrong recipe type")
    return values[0]


def _many(node, name, resolve, expected):
    values = _linked(node.inputs[name], resolve)
    if any(not isinstance(value, expected) for value in values):
        raise ValueError(f"{node.name}: {name} has the wrong recipe type")
    return values


def _text_or_string(text, string):
    """Contents of an optional Text block, else the inline string."""
    return text.as_string() if text else string


def _rest_matrix(obj):
    """World matrix of an object with its bone parent in the rest pose."""
    world = obj.matrix_world
    armature = obj.parent
    if armature is not None and obj.parent_type == 'BONE':
        pose_bone = armature.pose.bones.get(obj.parent_bone)
        if pose_bone is not None:
            rest = armature.matrix_world @ pose_bone.bone.matrix_local
            pose = armature.matrix_world @ pose_bone.matrix
            world = rest @ pose.inverted() @ world
    return world


def _attach_to_bone(obj, armature, bone_name, rest_world):
    """Parent an object to a bone so that it follows the pose and sits at
    ``rest_world`` while the bone is in its rest pose.
    """
    pose_bone = armature.pose.bones[bone_name]
    rest = armature.matrix_world @ pose_bone.bone.matrix_local
    pose = armature.matrix_world @ pose_bone.matrix
    # A bone parent sits at the tail of the bone.
    parent = pose @ Matrix.Translation((0, pose_bone.bone.length, 0))
    obj.parent = armature
    obj.parent_type = 'BONE'
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = parent.inverted() @ (
        pose @ rest.inverted() @ rest_world)


def _model_outputs(tree, node):
    """Model outputs the node feeds."""
    from ..build_graph import _upstream_nodes
    return [output for output in tree.nodes
            if isinstance(output, SourceIOPropModelNode)
            and node in set(_upstream_nodes(output))]


def _connected_armature(tree, node):
    """Armature of the geometry exported by the model outputs of a node."""
    from ..build_graph import GraphResolver, _upstream_nodes
    resolver = GraphResolver()
    for output in _model_outputs(tree, node):
        for geometry_node in _upstream_nodes(output):
            if not isinstance(geometry_node, SourceIOPropGeometryNode):
                continue
            try:
                geometry = resolver(geometry_node.outputs[0])
                rig = (
                    resolve_geometry_rig(geometry.source, geometry.rig)
                    if isinstance(geometry, GeometryRecipe) else None)
            except (ValueError, RuntimeError):
                continue
            if rig is not None:
                return rig
    return None


class _PropSocket(NodeSocket):
    bl_label = "Model socket"
    color = (0.35, 0.55, 0.75, 1.0)

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return self.color


class SourceIOModelEntrySocket(_PropSocket):
    bl_idname = "SourceIOModelEntrySocket"
    bl_label = "Model entry"


class SourceIOPropLodSocket(_PropSocket):
    bl_idname = "SourceIOPropLodSocket"
    bl_label = "LOD"


class SourceIOPropCollisionSocket(_PropSocket):
    bl_idname = "SourceIOPropCollisionSocket"
    bl_label = "Collision"


class SourceIOPropAttachmentSocket(_PropSocket):
    bl_idname = "SourceIOPropAttachmentSocket"
    bl_label = "Attachment"


class SourceIOPropHitboxSocket(_PropSocket):
    bl_idname = "SourceIOPropHitboxSocket"
    bl_label = "Hitbox"


class SourceIOPropIdleSocket(_PropSocket):
    bl_idname = "SourceIOPropIdleSocket"
    bl_label = "Idle sequence"


class SourceIOPropRawQcSocket(_PropSocket):
    bl_idname = "SourceIOPropRawQcSocket"
    bl_label = "Raw QC"


class SourceIOPropGeometryNode(_PropNode, Node):
    bl_idname = "SourceIOPropGeometryNode"
    bl_label = "Geometry"

    source_type: bpy.props.EnumProperty(name="Source", items=(
        ("OBJECT", "Object", ""), ("COLLECTION", "Collection", "")))
    obj: bpy.props.PointerProperty(name="Mesh object", type=bpy.types.Object)
    collection: bpy.props.PointerProperty(name="Mesh collection",
                                          type=bpy.types.Collection)
    rig: bpy.props.PointerProperty(
        name="Rig",
        description="Armature for weighted meshes without an armature "
                    "parent or modifier",
        type=bpy.types.Object, poll=lambda self, obj: obj.type == "ARMATURE")
    basename: bpy.props.StringProperty(
        name="Export basename",
        description="Leave blank to use the selected object or collection "
                    "name")
    subdir: bpy.props.StringProperty(name="Export subdirectory")
    auto_merge: bpy.props.BoolProperty(
        name="Auto-merge collection",
        description="Combine collection meshes into one DMX mesh",
        default=False)
    stereo_balance_groups: bpy.props.StringProperty(
        name="Stereo balance groups",
        description="One object_name=vertex_group per line")
    stereo_balance_groups_text: bpy.props.PointerProperty(
        name="Stereo groups Text", type=bpy.types.Text)
    vertex_animation_ranges: bpy.props.StringProperty(
        name="Vertex animation ranges",
        description="One clip start end [sequence] per line; end is "
                    "exclusive")
    vertex_animation_ranges_text: bpy.props.PointerProperty(
        name="Vertex animation Text", type=bpy.types.Text)

    def init(self, context):
        self.width = 210
        self.outputs.new("SourceIOObjectSocket", "Geometry")

    def draw_buttons(self, context, layout):
        layout.prop(self, "source_type")
        layout.prop(self, "obj" if self.source_type == "OBJECT"
                    else "collection")
        layout.prop(self, "rig")
        layout.prop(self, "basename")
        if not self.basename.strip():
            source = (self.obj if self.source_type == "OBJECT"
                      else self.collection)
            if source is not None:
                layout.label(text=f"Using {source.name} as export basename")
        layout.prop(self, "subdir")
        if self.source_type == "COLLECTION":
            layout.prop(self, "auto_merge")
        layout.prop(self, "stereo_balance_groups_text")
        if self.stereo_balance_groups_text is None:
            layout.prop(self, "stereo_balance_groups")
        layout.prop(self, "vertex_animation_ranges_text")
        if self.vertex_animation_ranges_text is None:
            layout.prop(self, "vertex_animation_ranges")

    def _balance_groups(self):
        groups = []
        seen_objects = set()
        source = _text_or_string(self.stereo_balance_groups_text,
                                 self.stereo_balance_groups)
        for line_number, raw in enumerate(source.splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            if line.count("=") != 1:
                raise ValueError(
                    f"{self.name}: stereo balance line {line_number} must "
                    f"be object_name=vertex_group")
            object_name, group_name = (part.strip()
                                       for part in line.split("="))
            if not object_name or not group_name:
                raise ValueError(
                    f"{self.name}: stereo balance line {line_number} needs "
                    f"an object and vertex group")
            if object_name in seen_objects:
                raise ValueError(
                    f"{self.name}: duplicate stereo balance object "
                    f"{object_name!r} on line {line_number}")
            seen_objects.add(object_name)
            groups.append((object_name, group_name))
        return tuple(groups)

    def _vertex_animations(self):
        animations = []
        seen_clips = set()
        source = _text_or_string(self.vertex_animation_ranges_text,
                                 self.vertex_animation_ranges)
        for line_number, raw in enumerate(source.splitlines(), 1):
            parts = raw.split()
            if not parts:
                continue
            if len(parts) not in (3, 4) or (len(parts) == 4
                                            and parts[3] != "sequence"):
                raise ValueError(
                    f"{self.name}: vertex animation line {line_number} must "
                    f"be clip start end [sequence]")
            clip = parts[0]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", clip):
                raise ValueError(
                    f"{self.name}: vertex animation line {line_number} has "
                    f"unsafe clip name {clip!r}")
            if clip in seen_clips:
                raise ValueError(
                    f"{self.name}: duplicate vertex animation clip {clip!r} "
                    f"on line {line_number}")
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"{self.name}: vertex animation line {line_number} needs "
                    f"integer start and end frames") from exc
            if start < 0 or end <= start:
                raise ValueError(
                    f"{self.name}: vertex animation line {line_number} needs "
                    f"0 <= start < end (exclusive)")
            seen_clips.add(clip)
            animations.append(
                VertexAnimationRange(clip, start, end, len(parts) == 4))
        return tuple(animations)

    def recipe(self, resolve=None):
        if self.source_type == "OBJECT":
            source = self.obj
            if source is None or source.type not in {"MESH", "CURVE"}:
                raise ValueError(
                    f"{self.name}: select a mesh or curve Object")
        else:
            source = self.collection
            if source is None or not any(obj.type in {"MESH", "CURVE"}
                                         for obj in source.all_objects):
                raise ValueError(
                    f"{self.name}: select a Collection with mesh or curve "
                    f"Objects")
        return GeometryRecipe(
            source, self.basename.strip() or source.name,
            self.subdir.strip(), self.rig,
            stereo_balance_groups=self._balance_groups(),
            vertex_animations=self._vertex_animations(),
            auto_merge=(self.auto_merge
                        if self.source_type == "COLLECTION" else False))


class SourceIO_PG_FlexController(PropertyGroup):
    raw_name: bpy.props.StringProperty(name="Shape key")
    second_raw_name: bpy.props.StringProperty(name="Second shape key")
    stereo: bpy.props.BoolProperty(
        name="Stereo",
        description="Split the controller using DMX balance values")
    eyelid: bpy.props.BoolProperty(name="Eyelid",
                                   description="Enable eyelid tracking")
    minimum: bpy.props.FloatProperty(name="Minimum", default=0.0)
    maximum: bpy.props.FloatProperty(name="Maximum", default=1.0)
    wrinkle_scale: bpy.props.FloatProperty(name="Wrinkle scale", default=0.0)
    second_wrinkle_scale: bpy.props.FloatProperty(
        name="Second wrinkle scale", default=0.0)


class SourceIOModelEntryNode(_PropNode, Node):
    bl_idname = "SourceIOModelEntryNode"
    bl_label = "Model entry"

    entry_name: bpy.props.StringProperty(name="Entry name", default="studio")
    directive: bpy.props.EnumProperty(name="Directive", items=(
        ("MODEL", "$model", "Model entry"),
        ("BODY", "$body", "Shortcut model entry"),
    ), default="MODEL")
    flex_controller_text: bpy.props.PointerProperty(
        name="DMX flex controllers",
        description="Advanced KeyValues2 flex controller/rule Text",
        type=bpy.types.Text)
    flex_controllers: bpy.props.CollectionProperty(
        type=SourceIO_PG_FlexController)
    flex_controller_index: bpy.props.IntProperty(default=0, min=0)

    def init(self, context):
        self.width = 200
        self.inputs.new("SourceIOObjectSocket", "Geometry")
        self.inputs.new("SourceIOFlexRulesSocket", "Flex Rules")
        self.outputs.new("SourceIOModelEntrySocket", "Model")

    def draw_buttons(self, context, layout):
        layout.prop(self, "entry_name")
        layout.prop(self, "directive")

    def recipe(self, resolve=None):
        flex_rules = (_one(self, "Flex Rules", resolve, FlexRuleSetRecipe)
                      if self.inputs.get("Flex Rules") else None)
        if flex_rules is not None and self.flex_controllers:
            raise ValueError(f"{self.name}: remove direct flex controllers "
                             f"before using Flex Rules")
        controls = ()
        if flex_rules is None:
            controls = tuple(self._controller(item)
                             for item in self.flex_controllers)
        return ModelEntryRecipe(
            self.entry_name.strip(),
            _one(self, "Geometry", resolve, GeometryRecipe, required=True),
            self.directive,
            (self.flex_controller_text.name
             if self.flex_controller_text else None),
            controls, flex_rules)

    @staticmethod
    def _controller(item):
        raw_names = (item.raw_name.strip(),)
        wrinkle_scales = (item.wrinkle_scale,)
        if item.second_raw_name.strip():
            raw_names += (item.second_raw_name.strip(),)
            wrinkle_scales += (item.second_wrinkle_scale,)
        return FlexControllerRecipe(
            item.name.strip(), raw_names, item.stereo, item.eyelid,
            item.minimum, item.maximum, wrinkle_scales)


class SourceIOPropBlankNode(_PropNode, Node):
    bl_idname = "SourceIOPropBlankNode"
    bl_label = "Blank bodygroup choice"

    def init(self, context):
        self.outputs.new("SourceIOObjectSocket", "Blank")

    def draw_buttons(self, context, layout):
        layout.label(text="Explicit blank choice")

    def recipe(self, resolve=None):
        return BLANK


class SourceIOPropBodygroupNode(_PropNode, Node):
    bl_idname = "SourceIOPropBodygroupNode"
    bl_label = "Bodygroup"

    bodygroup_name: bpy.props.StringProperty(name="Bodygroup name")

    def init(self, context):
        self.inputs.new("SourceIOObjectSocket", "Choices",
                        use_multi_input=True)
        self.outputs.new("SourceIOBodygroupSocket", "Bodygroup")

    def draw_buttons(self, context, layout):
        layout.prop(self, "bodygroup_name")

    def recipe(self, resolve=None):
        return BodygroupRecipe(
            self.bodygroup_name.strip(),
            _many(self, "Choices", resolve, (GeometryRecipe, type(BLANK))))


class SourceIOPropSkinFamilyNode(_PropNode, Node):
    bl_idname = "SourceIOPropSkinFamilyNode"
    bl_label = "Skin family"

    material_names: bpy.props.StringProperty(
        name="Replacement VMT names (one per line)",
        description="Ordered material names, without .vmt or directories")
    material_names_text: bpy.props.PointerProperty(
        name="Replacement names text", type=bpy.types.Text)

    def init(self, context):
        self.inputs.new("SourceIOMaterialSocket", "Materials",
                        use_multi_input=True)
        self.outputs.new("SourceIOSkinGroupSocket", "Skin family")

    def draw_buttons(self, context, layout):
        layout.prop(self, "material_names_text")
        if self.material_names_text is None:
            layout.prop(self, "material_names", text="Names")

    def recipe(self, resolve=None):
        linked = _many(self, "Materials", resolve, MaterialRecipe)
        if linked:
            return SkinFamilyRecipe(tuple(PurePosixPath(mat.vmt_path).stem
                                          for mat in linked))
        source = _text_or_string(self.material_names_text,
                                 self.material_names)
        names = tuple(line.strip() for line in source.splitlines())
        if any(not name or "/" in name or "\\" in name
               or name.lower().endswith(".vmt") for name in names):
            raise ValueError(f"{self.name}: use one bare VMT name per line")
        return SkinFamilyRecipe(names)


class SourceIOPropLodNode(_PropNode, Node):
    bl_idname = "SourceIOPropLodNode"
    bl_label = "LOD"

    distance: bpy.props.FloatProperty(name="Switch distance", default=20.0,
                                      min=0.0)
    shadow: bpy.props.BoolProperty(name="Shadow LOD", default=False)

    def init(self, context):
        self.inputs.new("SourceIOObjectSocket", "Original mesh")
        self.inputs.new("SourceIOObjectSocket", "Replacement mesh")
        self.outputs.new("SourceIOPropLodSocket", "LOD")

    def draw_buttons(self, context, layout):
        layout.prop(self, "shadow")
        if not self.shadow:
            layout.prop(self, "distance")

    def recipe(self, resolve=None):
        return LODRecipe(
            _one(self, "Original mesh", resolve, GeometryRecipe,
                 required=True),
            _one(self, "Replacement mesh", resolve, GeometryRecipe,
                 required=True),
            self.distance, self.shadow)


class SourceIOPropCollisionNode(_PropNode, Node):
    bl_idname = "SourceIOPropCollisionNode"
    bl_label = "Collision"

    mass: bpy.props.FloatProperty(name="Mass", default=1.0, min=0.0)
    concave: bpy.props.BoolProperty(name="Concave")
    mode: bpy.props.EnumProperty(name="Mode", items=(
        ("MODEL", "$collisionmodel", "Static collision geometry"),
        ("JOINTS", "$collisionjoints",
         "Per-bone convex hulls for a skinned model"),
    ), default="MODEL")
    joint_qc: bpy.props.StringProperty(
        name="Joint QC directives",
        description="Optional caller-authored QC directives inside "
                    "$collisionjoints")
    joint_qc_text: bpy.props.PointerProperty(name="Joint QC text block",
                                             type=bpy.types.Text)

    def init(self, context):
        self.inputs.new("SourceIOObjectSocket", "Geometry")
        self.outputs.new("SourceIOPropCollisionSocket", "Collision")

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode")
        layout.prop(self, "mass")
        if self.mode == "MODEL":
            layout.prop(self, "concave")
        else:
            layout.prop(self, "joint_qc_text")
            if self.joint_qc_text is None:
                layout.prop(self, "joint_qc")

    def recipe(self, resolve=None):
        return CollisionRecipe(
            _one(self, "Geometry", resolve, GeometryRecipe, required=True),
            self.mass, self.concave, self.mode,
            _text_or_string(self.joint_qc_text, self.joint_qc))


class SourceIOPropAttachmentNode(_PropNode, Node):
    bl_idname = "SourceIOPropAttachmentNode"
    bl_label = "Attachment"

    attachment_name: bpy.props.StringProperty(name="Name")
    armature: bpy.props.PointerProperty(
        name="Armature", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE",
        description="Used to pick the bone and to place the empty relative "
                    "to it")
    bone: bpy.props.StringProperty(name="Bone", default="root")
    empty: bpy.props.PointerProperty(
        name="Empty", type=bpy.types.Object,
        description="Object marking the attachment; its rest pose transform "
                    "relative to the bone replaces Position and Rotation")
    position: bpy.props.FloatVectorProperty(
        name="Position", size=3, subtype="TRANSLATION")
    rotation: bpy.props.FloatVectorProperty(
        name="Rotation", size=3, subtype="EULER",
        description="QC angles: pitch (around Y), yaw (around Z) and roll "
                    "(around X)")

    def init(self, context):
        self.width = 220
        self.outputs.new("SourceIOPropAttachmentSocket", "Attachment")

    def draw_buttons(self, context, layout):
        layout.prop(self, "attachment_name")
        layout.prop(self, "armature")
        if self.armature is not None:
            layout.prop_search(self, "bone", self.armature.data, "bones")
        else:
            layout.prop(self, "bone")
        layout.prop(self, "empty")
        if self.empty is None:
            layout.prop(self, "position")
            column = layout.column(align=True)
            column.label(text="Rotation")
            for index, label in enumerate(("Pitch", "Yaw", "Roll")):
                column.prop(self, "rotation", index=index, text=label)
            layout.operator(
                "sourceio.spawn_attachment_empty", icon='EMPTY_ARROWS'
            ).node_name = self.name

    def _empty_transform(self):
        """Position and QC angles of the empty in the space of the bone."""
        armature = self.armature
        parent = self.empty.parent
        if armature is None and parent is not None and (
                parent.type == 'ARMATURE'):
            armature = parent
        world = _rest_matrix(self.empty)
        if armature is not None:
            bone = armature.data.bones.get(self.bone.strip())
            if bone is None:
                raise ValueError(
                    f"{self.name}: bone {self.bone.strip()!r} is not in "
                    f"armature {armature.name!r}")
            world = (armature.matrix_world
                     @ bone.matrix_local).inverted() @ world
        location, rotation, _ = world.decompose()
        # QC angles are pitch, yaw, roll: turns around Y, Z and X.
        angles = rotation.to_euler('XYZ')
        return tuple(location), tuple(
            degrees(angle) for angle in (angles.y, angles.z, angles.x))

    def recipe(self, resolve=None):
        if self.empty is not None:
            position, rotation = self._empty_transform()
        else:
            position = tuple(self.position)
            rotation = tuple(degrees(angle) for angle in self.rotation)
        return AttachmentRecipe(self.attachment_name.strip(),
                                self.bone.strip(), position, rotation)


def spawn_attachment_empty(tree, node, scene):
    """Create the empty of an Attachment node at its Position and Rotation,
    parented to its bone when the armature has it.
    """
    armature = node.armature or _connected_armature(tree, node)
    bone = None
    if armature is not None:
        node.armature = armature
        bone = armature.data.bones.get(node.bone.strip())
    pitch, yaw, roll = node.rotation
    rest = Matrix.Translation(node.position) @ Euler(
        (roll, pitch, yaw), 'XYZ').to_matrix().to_4x4()
    empty = bpy.data.objects.new(
        node.attachment_name.strip() or "attachment", None)
    empty.empty_display_type = 'ARROWS'
    collections = (armature.users_collection if armature else ()) or (
        scene.collection,)
    for collection in collections:
        collection.objects.link(empty)
    if bone is not None:
        empty.empty_display_size = max(bone.length * 0.5, 1e-3)
        _attach_to_bone(empty, armature, bone.name,
                        armature.matrix_world @ bone.matrix_local @ rest)
    else:
        empty.matrix_basis = rest
    node.empty = empty
    return empty


class SourceIO_OP_SpawnAttachmentEmpty(Operator):
    bl_idname = "sourceio.spawn_attachment_empty"
    bl_label = "Create empty"
    bl_description = (
        "Create an empty at the attachment's Position and Rotation and use "
        "it instead. It is parented to the bone when the armature (the "
        "node's, or the rig of the connected model output) has that bone")
    bl_options = {'REGISTER', 'UNDO'}
    node_name: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return bool(space and space.type == 'NODE_EDITOR' and
                    space.edit_tree and
                    space.edit_tree.bl_idname == 'sourceio.model_definition')

    def execute(self, context):
        tree = context.space_data.edit_tree
        node = tree.nodes.get(self.node_name) or context.active_node
        if not isinstance(node, SourceIOPropAttachmentNode):
            self.report({'ERROR'}, 'Select an Attachment node')
            return {'CANCELLED'}
        if node.empty is not None:
            self.report({'ERROR'}, f'{node.name} already uses an empty')
            return {'CANCELLED'}
        empty = spawn_attachment_empty(tree, node, context.scene)
        if empty.parent is not None:
            self.report({'INFO'}, f'Created {empty.name} on bone '
                                  f'{empty.parent_bone}')
        else:
            self.report({'WARNING'},
                        f'Created {empty.name} unparented: no armature '
                        f'with bone {node.bone.strip()!r}')
        return {'FINISHED'}


class SourceIOPropHitboxNode(_PropNode, Node):
    bl_idname = "SourceIOPropHitboxNode"
    bl_label = "Hitbox"

    group: bpy.props.IntProperty(name="Group", min=0)
    bone: bpy.props.StringProperty(name="Bone", default="root")
    minimum: bpy.props.FloatVectorProperty(name="Minimum", size=3)
    maximum: bpy.props.FloatVectorProperty(name="Maximum", size=3)

    def init(self, context):
        self.outputs.new("SourceIOPropHitboxSocket", "Hitbox")

    def draw_buttons(self, context, layout):
        for prop in ("group", "bone", "minimum", "maximum"):
            layout.prop(self, prop)

    def recipe(self, resolve=None):
        return HitboxRecipe(self.group, self.bone.strip(),
                            tuple(self.minimum), tuple(self.maximum))


class SourceIOPropIdleNode(_PropNode, Node):
    bl_idname = "SourceIOPropIdleNode"
    bl_label = "Idle sequence"

    sequence_name: bpy.props.StringProperty(name="Sequence name",
                                            default="idle")
    fps: bpy.props.FloatProperty(name="FPS", default=1.0, min=0.001)

    def init(self, context):
        self.inputs.new("SourceIOObjectSocket", "Geometry")
        self.outputs.new("SourceIOPropIdleSocket", "Idle")

    def draw_buttons(self, context, layout):
        layout.prop(self, "sequence_name")
        layout.prop(self, "fps")

    def recipe(self, resolve=None):
        return IdleRecipe(self.sequence_name.strip(),
                          _one(self, "Geometry", resolve, GeometryRecipe),
                          self.fps)


class SourceIOBoneNamesSocket(_PropSocket):
    bl_idname = "SourceIOBoneNamesSocket"
    bl_label = "Bone names"


class SourceIOBoneRenameNode(_PropNode, Node):
    bl_idname = "SourceIOBoneRenameNode"
    bl_label = "Bone rename table"

    table: bpy.props.PointerProperty(
        name="Rename table", type=bpy.types.Text,
        description="Lines of 'Blender name = exported name'")

    def init(self, context):
        self.width = 220
        self.outputs.new("SourceIOBoneNamesSocket", "Bone names")

    def draw_buttons(self, context, layout):
        layout.prop(self, "table")
        layout.operator("sourceio.generate_bone_table",
                        icon='FILE_REFRESH').rename_node = self.name

    def recipe(self, resolve=None):
        if self.table is None:
            raise ValueError(f"{self.name}: assign or generate a rename table")
        try:
            return BoneRenameRecipe(parse_bone_table(self.table.as_string()))
        except ValueError as exc:
            raise ValueError(
                f"{self.name} ({self.table.name}): {exc}") from exc


class SourceIOPropRawQcNode(_PropNode, Node):
    bl_idname = "SourceIOPropRawQcNode"
    bl_label = "Raw QC"

    snippet: bpy.props.StringProperty(
        name="QC lines",
        description="Emitted verbatim after typed directives")
    snippet_text: bpy.props.PointerProperty(name="QC text block",
                                            type=bpy.types.Text)

    def init(self, context):
        self.outputs.new("SourceIOPropRawQcSocket", "QC")

    def draw_buttons(self, context, layout):
        layout.prop(self, "snippet_text")
        if self.snippet_text is None:
            layout.prop(self, "snippet", text="Line")

    def recipe(self, resolve=None):
        return RawQcRecipe(_text_or_string(self.snippet_text, self.snippet))


class SourceIOPropModelNode(_PropNode, Node):
    bl_idname = "SourceIOPropModelNode"
    bl_label = "Source 1 model output"

    model_path: bpy.props.StringProperty(
        name="Model path", description="Game-relative models/.../*.mdl")
    mostlyopaque: bpy.props.BoolProperty(name="$mostlyopaque", default=True)
    ambientboost: bpy.props.BoolProperty(name="$ambientboost", default=True)
    material_dirs: bpy.props.StringProperty(
        name="$cdmaterials paths",
        description="One material-root-relative path per line")
    material_dirs_text: bpy.props.PointerProperty(
        name="$cdmaterials text block", type=bpy.types.Text)
    content_root: bpy.props.StringProperty(name="Content root",
                                           subtype="DIR_PATH")
    game_root: bpy.props.StringProperty(name="Game root", subtype="DIR_PATH")
    geometry_format: bpy.props.EnumProperty(name="Geometry format", items=(
        ("DMX", "DMX", "Export DMX using SourceIO's native exporter"),
        ("SMD", "SMD", "Export SMD using SourceIO's native exporter"),
    ), default="DMX")
    dmx_encoding_version: bpy.props.EnumProperty(
        name="DMX encoding version", items=(
            ("2", "2", ""), ("3", "3", ""), ("4", "4", ""), ("5", "5", ""),
        ), default="2")
    dmx_model_version: bpy.props.EnumProperty(
        name="DMX model version", items=(
            ("1", "1", ""), ("15", "15", ""), ("18", "18", ""),
        ), default="1")
    dmx_trim_weights: bpy.props.BoolProperty(
        name="Limit weights to 3 bones", default=True,
        description="Keep the three strongest bone weights of every vertex "
                    "and renormalize, as Source 1 studiomdl skins with at "
                    "most three bones. Off writes every weight untrimmed")
    dmx_up_axis: bpy.props.EnumProperty(name="Geometry up axis", items=(
        ("Z", "Z up", "Blender coordinates"),
        ("Y", "Y up", "Rotate Blender Z-up coordinates to Y-up"),
    ), default="Z")
    native_settings_confirmed: bpy.props.BoolProperty(
        name="Native settings reviewed",
        description="Confirm up axis, collection merge and triangulation "
                    "(and DMX version for DMX graphs) after migration from "
                    "BST",
        default=False)
    compiler_backend: bpy.props.EnumProperty(name="Compiler backend", items=(
        ("COMMAND", "Custom command",
         "Invoke configured compiler command without a shell"),
        ("STUDIOMDL", "Blender Source Tools",
         "Use Blender Source Tools' Windows studiomdl.exe backend"),
    ), default="COMMAND")
    compiler_command: bpy.props.StringProperty(
        name="Compiler command",
        description="Argument template with {game} and {qc}")
    bst_engine_root: bpy.props.StringProperty(name="BST engine root",
                                              subtype="DIR_PATH")
    auto_idle: bpy.props.BoolProperty(name="Simple idle sequence",
                                      default=True)
    use_export_cache: bpy.props.BoolProperty(
        name="Use export cache", default=True,
        description="Only re-export meshes and textures whose inputs "
                    "changed since the last export (fingerprints are kept "
                    "in <model>.sourceio-cache.json next to the QC)")
    model_scale: bpy.props.FloatProperty(
        name="Model scale", default=1.0, min=1e-6, soft_min=0.001,
        soft_max=1000.0, precision=4,
        description="Multiplies every exported position, bone offset, shape "
                    "delta, attachment and hitbox (Blender units to Source "
                    "units, e.g. 39.37 for metres to inches). Not QC $scale")

    def init(self, context):
        self.width = 290
        self.native_settings_confirmed = True
        self.inputs.new("SourceIOModelEntrySocket", "Models",
                        use_multi_input=True)
        self.inputs.new("SourceIOMaterialSocket", "Materials",
                        use_multi_input=True)
        for socket_id, name in (
            ("SourceIOBodygroupSocket", "Bodygroups"),
            ("SourceIOSkinGroupSocket", "Skin families"),
            ("SourceIOPropLodSocket", "LODs"),
            ("SourceIOPropAttachmentSocket", "Attachments"),
            ("SourceIOPropHitboxSocket", "Hitboxes"),
            ("SourceIOPropRawQcSocket", "Raw QC"),
        ):
            self.inputs.new(socket_id, name, use_multi_input=True)
        self.inputs.new("SourceIOFlexRulesSocket", "Flex Rules",
                        use_multi_input=True)
        self.inputs.new("SourceIOPropCollisionSocket", "Collision")
        self.inputs.new("SourceIOPropIdleSocket", "Idle")
        self.inputs.new("SourceIOBoneNamesSocket", "Bone names")
        self.inputs.new("SourceIOSfmRigSocket", "SFM rig")

    def draw_buttons(self, context, layout):
        for prop in ("model_path", "model_scale", "mostlyopaque",
                     "ambientboost", "material_dirs_text"):
            layout.prop(self, prop)
        if self.material_dirs_text is None:
            layout.prop(self, "material_dirs")
        for prop in ("auto_idle", "content_root", "game_root",
                     "geometry_format", "use_export_cache",
                     "compiler_backend"):
            layout.prop(self, prop)
        if self.geometry_format == "DMX":
            layout.prop(self, "dmx_encoding_version")
            layout.prop(self, "dmx_model_version")
            layout.prop(self, "dmx_trim_weights")
        layout.prop(self, "dmx_up_axis")
        if not self.native_settings_confirmed:
            layout.prop(self, "native_settings_confirmed")
        layout.prop(self, "compiler_command"
                    if self.compiler_backend == "COMMAND"
                    else "bst_engine_root")
        for operator in ("sourceio.preview_export_graph",
                         "sourceio.export_model_graph",
                         "sourceio.export_model_meshes",
                         "sourceio.export_model_materials",
                         "sourceio.compile_model_graph"):
            layout.operator(operator).output_node = self.name

    def recipe(self, resolve=None):
        return collect_model(self, resolve)


def collect_model(output_node: SourceIOPropModelNode, resolve) -> ModelRecipe:
    """Resolve one output; callback receives each upstream output socket."""
    from ..flex_graph import bind_flex_rules
    models = _many(output_node, "Models", resolve, ModelEntryRecipe)
    bodygroups = _many(output_node, "Bodygroups", resolve, BodygroupRecipe)
    shared = (_many(output_node, "Flex Rules", resolve, FlexRuleSetRecipe)
              if output_node.inputs.get("Flex Rules") else ())
    models, flex_entries = bind_flex_rules(models, shared, bodygroups)
    renames = (_one(output_node, "Bone names", resolve, BoneRenameRecipe)
               if output_node.inputs.get("Bone names") else None)
    sfm_rig = (_one(output_node, "SFM rig", resolve, IKRigRecipe)
               if output_node.inputs.get("SFM rig") else None)
    idle = _one(output_node, "Idle", resolve, IdleRecipe)
    if idle is None and output_node.auto_idle:
        idle = IdleRecipe()
    material_dirs = _text_or_string(output_node.material_dirs_text,
                                    output_node.material_dirs)
    result = ModelRecipe(
        model_path=output_node.model_path.strip(), models=models,
        mostlyopaque=output_node.mostlyopaque,
        ambientboost=output_node.ambientboost,
        material_dirs=tuple(line.strip() for line in
                            material_dirs.splitlines() if line.strip()),
        bodygroups=bodygroups, flex_entries=flex_entries,
        scale=output_node.model_scale,
        bone_names=renames.names if renames else (),
        skinfamilies=_many(output_node, "Skin families", resolve,
                           SkinFamilyRecipe),
        lods=_many(output_node, "LODs", resolve, LODRecipe),
        collision=_one(output_node, "Collision", resolve, CollisionRecipe),
        attachments=_many(output_node, "Attachments", resolve,
                          AttachmentRecipe),
        hitboxes=_many(output_node, "Hitboxes", resolve, HitboxRecipe),
        idle=idle,
        raw_snippets=_many(output_node, "Raw QC", resolve, RawQcRecipe),
        sfm_rig=sfm_rig,
    )
    result.geometries()
    return result


# Register sockets before nodes; parent owns insertion into package
# registration.
classes = (
    SourceIO_PG_FlexController,
    SourceIOModelEntrySocket, SourceIOPropLodSocket,
    SourceIOPropCollisionSocket, SourceIOPropAttachmentSocket,
    SourceIOPropHitboxSocket, SourceIOPropIdleSocket, SourceIOPropRawQcSocket,
    SourceIOPropGeometryNode, SourceIOModelEntryNode, SourceIOPropBlankNode,
    SourceIOPropBodygroupNode, SourceIOPropSkinFamilyNode,
    SourceIOPropLodNode, SourceIOPropCollisionNode,
    SourceIOPropAttachmentNode, SourceIO_OP_SpawnAttachmentEmpty,
    SourceIOPropHitboxNode, SourceIOPropIdleNode,
    SourceIOPropRawQcNode, SourceIOBoneNamesSocket, SourceIOBoneRenameNode,
    SourceIOPropModelNode,
)
