"""Resolve one model output, preview its manifest and build its artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from uuid import uuid4

import bpy

from SourceIO.blender_bindings.source1.model_export.compiler import (
    CompilationError, compile_direct, compile_with_source_tools,
    direct_command, validate_source_tools_compiler)
from SourceIO.blender_bindings.source1.model_export.dmx_axis import (
    orient_capture)
from SourceIO.blender_bindings.source1.model_export.dmx_builder import (
    _validate_versions, write_dmx, write_vca_sequences)
from SourceIO.blender_bindings.source1.model_export.dmx_capture import (
    capture_geometry, capture_vertex_animations)
from SourceIO.blender_bindings.source1.model_export.dmx_merge import (
    merge_collection_capture)
from SourceIO.blender_bindings.source1.model_export.dmx_stereo import (
    derive_rule_shapes)
from SourceIO.blender_bindings.source1.model_export.dmx_transform import (
    transform_capture)
from SourceIO.blender_bindings.source1.model_export.geometry_rig import (
    resolve_geometry_rig, validate_geometry_weights)
from SourceIO.blender_bindings.source1.model_export.material_assets import (
    BinaryArtifact, ExternalTextureSourceRecipe, ImageSourceRecipe,
    MaterialRecipe, VTFRecipe, material_asset_path, texture_pixels)
from SourceIO.blender_bindings.source1.model_export.prop_qc import (
    FlexControllerRecipe, GeometryRecipe, ModelRecipe, parse_bone_table,
    render_qc)
from SourceIO.blender_bindings.source1.model_export.sfm_rig import (
    SCRIPT_DIRECTORY, render_sfm_rig, script_name)
from SourceIO.blender_bindings.source1.model_export.smd_builder import (
    write_smd)
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_5
from .export_cache import (ExportCache, cache_path, geometry_fingerprint,
                           node_fingerprint)
from .nodes.material_nodes import (SourceIOExternalTextureNode,
                                   SourceIOImageSourceNode,
                                   SourceIOVTFOutputNode, VMT_NODE_TYPES)
from .nodes.prop_nodes import (SourceIOModelEntryNode,
                               SourceIOPropGeometryNode,
                               SourceIOPropModelNode, collect_model)


class GraphBuildError(RuntimeError):
    pass


class GraphResolver:
    """Compute each upstream output once, preserving multi-input link order."""

    def __init__(self):
        self.cache = {}
        self.active = set()

    def __call__(self, socket):
        if not socket.is_output:
            if not socket.links:
                return getattr(socket, 'default_value', None)
            if len(socket.links) != 1:
                raise GraphBuildError(
                    f'{socket.node.name}: {socket.name} needs one input')
            return self(socket.links[0].from_socket)
        node = socket.node
        key = (node, socket)
        if key in self.cache:
            return self.cache[key]
        if node in self.active:
            raise GraphBuildError(f'Cycle detected at {node.name}')
        self.active.add(node)
        try:
            if node.type == 'REROUTE':
                result = self(node.inputs[0])
            elif node.mute:
                link = next((item for item in node.internal_links
                             if item.to_socket == socket), None)
                if link is None:
                    raise GraphBuildError(
                        f'{node.name}: muted output has no bypass')
                result = self(link.from_socket)
            elif callable(getattr(node, 'recipe', None)):
                result = node.recipe(self)
            elif callable(getattr(node, 'process', None)):
                inputs = {}
                for item in node.inputs:
                    value = self(item)
                    if isinstance(value, (ImageSourceRecipe,
                                          ExternalTextureSourceRecipe)):
                        value = texture_pixels(value)
                    inputs[item.name] = value
                values = node.process(inputs) or {}
                for output in node.outputs:
                    if output.name in values:
                        self.cache[(node, output)] = values[output.name]
                result = self.cache.get(key)
            else:
                raise GraphBuildError(
                    f'{node.name} is not an export graph node')
            if result is None:
                raise GraphBuildError(
                    f'{node.name}: output {socket.name} is unavailable')
            self.cache[key] = result
            return result
        finally:
            self.active.remove(node)


@dataclass(frozen=True)
class FlexSource:
    key: str
    label: str
    detail: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildPlan:
    model: ModelRecipe
    content_root: Path
    game_root: Path
    qc_path: Path
    geometry_format: str
    encoding_version: int
    model_version: int
    up_axis: str
    compiler_backend: str
    compiler_command: str
    bst_engine_root: str
    geometry: tuple[tuple[object, Path], ...]
    assets: tuple[tuple[Path, bytes], ...]
    qc_text: str
    material_names: tuple[tuple[str, str], ...]
    flex_sources: tuple[FlexSource, ...]
    # Incremental export: VTFs reused from disk (their data is empty) and
    # fingerprints of the others.
    cached_assets: frozenset[Path] = frozenset()
    texture_keys: tuple[tuple[Path, str], ...] = ()
    trim_weights: bool = True

    @property
    def model_file(self) -> Path:
        return self.game_root / self.model.model_path

    def manifest(self) -> str:
        lines = [f'QC: {self.qc_path}', f'Compiled model: {self.model_file}',
                 f'Geometry format: {self.geometry_format}']
        if self.model.scale != 1.0:
            lines.append(f'Model scale: {self.model.scale:g}')
        lines.extend(f'Bone rename: {source} -> {target}'
                     for source, target in self.model.bone_names
                     if source != target)
        if self.geometry_format == 'DMX':
            lines.append(f'DMX encoding/model: {self.encoding_version}/'
                         f'{self.model_version}; up axis: {self.up_axis}')
        else:
            lines.append(f'SMD up axis: {self.up_axis}')
        for geometry, path in self.geometry:
            lines.append(f'Geometry: {path}')
            clip_directory = path.with_suffix('')
            lines.extend(f'Vertex animation: '
                         f'{clip_directory / f"vcaanim_{clip.name}.dmx"}'
                         for clip in geometry.vertex_animations
                         if clip.export_sequence)
        lines.extend(f'Asset: {path}' for path, _ in self.assets)
        for item in self.flex_sources:
            lines.append(f'Flex controllers: {item.label} -> {item.key}')
            lines.extend(f'    {line}' for line in item.detail)
        return '\n'.join(lines) + '\n\n' + self.qc_text


def _root(value: str, label: str) -> Path:
    if not value.strip():
        raise GraphBuildError(f'Configure {label} on the model output')
    if value.startswith('//') and not bpy.data.filepath:
        raise GraphBuildError(
            f'{label} uses a blend-relative path; save the .blend first')
    path = Path(bpy.path.abspath(value)).resolve()
    if path.exists() and not path.is_dir():
        raise GraphBuildError(f'{label} is not a directory: {path}')
    return path


_progress = None


class export_progress:
    """Drive Blender's progress indicator for one export action.

    Uses ``WindowManager.progress_*``. Export steps call ``_tick()``; the
    step total is an estimate, values past it are clamped.
    """

    def __init__(self, window_manager, total):
        self.window_manager = window_manager
        self.total = max(int(total), 1)
        self.done = 0

    def __enter__(self):
        global _progress
        self.window_manager.progress_begin(0, self.total)
        _progress = self
        return self

    def __exit__(self, *exc):
        global _progress
        _progress = None
        self.window_manager.progress_end()
        return False

    def step(self):
        self.done += 1
        self.window_manager.progress_update(min(self.done, self.total))


def _tick():
    if _progress is not None:
        _progress.step()


def _upstream_nodes(output):
    seen = set()
    queue = [output]
    while queue:
        node = queue.pop()
        if node in seen:
            continue
        seen.add(node)
        yield node
        for socket in node.inputs:
            queue.extend(link.from_node for link in socket.links)


def _artifacts(materials: tuple[MaterialRecipe, ...], game_root: Path):
    result = {}
    for material in materials:
        sources = [(material.vmt_path, material.vmt_text.encode('utf-8'))]
        sources.extend((texture.path, texture.data)
                       for texture in material.vtf_artifacts)
        for name, data in sources:
            relative = PurePosixPath(name)
            if (not name.startswith('materials/') or relative.is_absolute()
                    or any(part in ('', '.', '..')
                           for part in name.split('/'))):
                raise GraphBuildError(
                    f'Unsafe material asset path: {name!r}')
            destination = game_root.joinpath(*relative.parts)
            old = result.get(destination)
            if old is not None and old != data:
                raise GraphBuildError(
                    f'Conflicting material recipes for {destination}')
            result[destination] = data
    return tuple(result.items())


def _geometry_objects(geometry):
    if isinstance(geometry.source, bpy.types.Collection):
        return geometry.source.all_objects
    return (geometry.source,)


def _flex_shape_names(geometry) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        shape.name for obj in _geometry_objects(geometry)
        if obj.type == 'MESH' and obj.data.shape_keys
        for shape in obj.data.shape_keys.key_blocks[1:]))


def _rule_set_lines(rule_set) -> tuple[str, ...]:
    """Preview text of a flex rule set.

    Controllers (range, directly driven shape) and one
    ``mesh: shape = expression`` line per rule.
    """
    lines = []
    for control in rule_set.controls:
        raw = getattr(control, 'raw_names', ())
        lines.append(f'controller {control.name} '
                     f'[{control.minimum:g}..{control.maximum:g}]'
                     + (f' -> {", ".join(raw)}' if raw else ''))
    lines.extend(f'{rule.object_name}: {rule.shape_name} = {rule.expression}'
                 + (f' (wrinkle {rule.wrinkle_scale:g})'
                    if rule.wrinkle_scale else '')
                 for rule in rule_set.rules)
    return tuple(lines)


def _validate_flex_rules(entry) -> None:
    rule_set = entry.flex_rule_set
    geometry = entry.geometry
    if (rule_set.geometry.key != geometry.key
            or rule_set.geometry.source != geometry.source):
        raise GraphBuildError(
            f'Flex rules for {entry.name!r} must use its linked geometry')
    shapes = {}
    owners = set()
    for obj in _geometry_objects(geometry):
        if obj.type != 'MESH' or obj.data.shape_keys is None:
            continue
        for shape in obj.data.shape_keys.key_blocks[1:]:
            owners.add(shape.name)
            shapes[(obj.name, shape.name)] = None
    if not shapes:
        raise GraphBuildError(
            f'Flex geometry {geometry.key!r} has no shape keys')
    # Generated stereo halves cover their source shape key.
    targets = {(rule.object_name, rule.source_shape or rule.shape_name)
               for rule in rule_set.rules}
    owners.update(rule.shape_name for rule in rule_set.rules
                  if rule.source_shape)
    missing = shapes.keys() - targets
    extra = targets - shapes.keys()
    if missing or extra:
        details = []
        if missing:
            details.append('missing ' + ', '.join(
                f'{obj}:{shape}' for obj, shape in sorted(missing)))
        if extra:
            details.append('unknown ' + ', '.join(
                f'{obj}:{shape}' for obj, shape in sorted(extra)))
        raise GraphBuildError(f'Flex rules for {geometry.key!r} have invalid '
                              'shape coverage: ' + '; '.join(details))
    raw_names = set()
    for control in rule_set.controls:
        if isinstance(control, FlexControllerRecipe):
            raw_names.update(control.raw_names)

    unknown = raw_names - owners
    if unknown:
        raise GraphBuildError(f'Flex controller raw names not found on '
                              f'geometry {geometry.key!r}: '
                              f'{", ".join(sorted(unknown))}')


def _flex_sources(model: ModelRecipe,
                  geometry_format: str) -> tuple[FlexSource, ...]:
    signatures = {}
    result = []
    for entry in model.models + model.flex_entries:
        key = entry.geometry.key
        rule_signature = None
        if entry.flex_rule_set is not None:
            rule_signature = (entry.flex_rule_set.geometry.key,
                              entry.flex_rule_set.geometry.source,
                              entry.flex_rule_set.controls,
                              entry.flex_rule_set.rules)
        signature = (entry.flex_controller_source, entry.flex_controllers,
                     rule_signature)
        if key in signatures:
            if signatures[key] != signature:
                raise GraphBuildError(f'Geometry {key!r} has conflicting '
                                      'flex controller/rule settings')
            continue
        signatures[key] = signature
        if (not entry.flex_controller_source and not entry.flex_controllers
                and entry.flex_rule_set is None):
            continue
        if geometry_format != 'DMX' or entry.directive != 'MODEL':
            raise GraphBuildError(
                'Flex controllers and rules require a DMX $model entry')
        shapes = set(_flex_shape_names(entry.geometry))
        if not shapes:
            raise GraphBuildError(f'Flex geometry {key!r} has no shape keys')
        if entry.flex_rule_set is not None:
            _validate_flex_rules(entry)
            rule_count = len(entry.flex_rule_set.rules)
            result.append(FlexSource(
                key, f'{entry.name} ({rule_count} graph rules)',
                _rule_set_lines(entry.flex_rule_set)))
            continue
        if entry.flex_controllers:
            raw_names = {raw for control in entry.flex_controllers
                         for raw in control.raw_names}
            missing = raw_names - shapes
            if missing:
                raise GraphBuildError(f'Flex shape keys not found on '
                                      f'geometry {key!r}: '
                                      f'{", ".join(sorted(missing))}')
            control_count = len(entry.flex_controllers)
            result.append(FlexSource(
                key, f'{entry.name} ({control_count} node controls)'))
            continue
        if entry.geometry.auto_merge:
            raise GraphBuildError('Advanced flex Text cannot target '
                                  'auto-merged collection meshes; use node '
                                  'controls or graph rules instead')
        name = entry.flex_controller_source
        text = bpy.data.texts.get(name)
        if text is None:
            raise GraphBuildError(
                f'Flex controller Text {name!r} no longer exists')
        from SourceIO.library.utils import datamodel_v2
        content = text.as_string()
        if (not content.startswith('<!-- dmx encoding keyvalues2 ')
                or '-->' not in content.split('\n', 1)[0]):
            raise GraphBuildError(f'Invalid flex controller Text {name!r}: '
                                  'expected a KeyValues2 DMX header')
        try:
            dm = datamodel_v2.parse(content)
            operator = dm.root['combinationOperator']
            controls = operator['controls']
            operator['targets']  # KeyError when the operator has no targets
            raw_names = {raw for control in controls
                         for raw in control['rawControlNames']}
        except Exception as exc:
            raise GraphBuildError(
                f'Invalid flex controller Text {name!r}: {exc}') from exc
        if not shapes.intersection(raw_names):
            raise GraphBuildError(f'Flex controller Text {name!r} has no '
                                  f'controls for shape keys on geometry '
                                  f'{key!r}')
        result.append(FlexSource(key, name))
    return tuple(result)


def _sfm_rig_script(model: ModelRecipe, rigs,
                    game_root: Path) -> tuple[Path, bytes]:
    """The SFM IK rig script of a model.

    It is validated against the armature of the model's $model/$body
    entries.
    """
    armatures = tuple(dict.fromkeys(
        rigs[item.geometry.key] for item in model.models
        if rigs[item.geometry.key] is not None))
    if len(armatures) != 1:
        raise GraphBuildError('SFM IK rig needs the model entries to use '
                              'exactly one armature')
    armature = armatures[0]
    parents = {bone.name: bone.parent.name if bone.parent else None
               for bone in armature.data.bones}
    positions = {bone.name: tuple(armature.matrix_world @ bone.head_local)
                 for bone in armature.data.bones}
    text = render_sfm_rig(model.sfm_rig, parents, dict(model.bone_names),
                          model.model_path, positions)
    name = script_name(model.sfm_rig, model.model_path)
    path = game_root.joinpath(*PurePosixPath(SCRIPT_DIRECTORY).parts,
                              f'{name}.py')
    return path, text.encode('utf-8')


def _stub_textures(resolver, root_node):
    """Resolve VTF Output nodes to path-only recipes.

    VMTs then render without encoding any texture.
    """
    for node in _upstream_nodes(root_node):
        if isinstance(node, SourceIOVTFOutputNode):
            try:
                path = material_asset_path(node.vtf_name, '.vtf')
            except ValueError as exc:
                raise GraphBuildError(f'{node.name}: {exc}') from exc
            resolver.cache[(node, node.outputs['VTF'])] = VTFRecipe(
                BinaryArtifact(path, b''))


def _cached_textures(resolver, root_node, game_root: Path,
                     cache: ExportCache):
    """Stub VTF Outputs whose fingerprint matches the file on disk.

    Returns (fingerprints of the other outputs, stubbed paths).
    """
    keys, cached = {}, set()
    for node in _upstream_nodes(root_node):
        if not isinstance(node, SourceIOVTFOutputNode):
            continue
        try:
            relative = material_asset_path(node.vtf_name, '.vtf')
        except ValueError as exc:
            raise GraphBuildError(f'{node.name}: {exc}') from exc
        destination = game_root.joinpath(*PurePosixPath(relative).parts)
        key = node_fingerprint(node)
        if cache.fresh(destination, key):
            resolver.cache[(node, node.outputs['VTF'])] = VTFRecipe(
                BinaryArtifact(relative, b''))
            cached.add(destination)
        else:
            keys[destination] = key
    return keys, cached


def model_cache(output: SourceIOPropModelNode) -> ExportCache | None:
    """The output's export cache, if enabled.

    It is stored as ``<qc stem>.sourceio-cache.json`` next to the QC.
    """
    if not output.use_export_cache:
        return None
    try:
        qc_path = _root(output.content_root, 'content root') / PurePosixPath(
            output.model_path.strip()).with_suffix('.qc')
    except (GraphBuildError, ValueError):
        return None
    return ExportCache(cache_path(qc_path))


def plan_model(output: SourceIOPropModelNode, *, check_shared: bool = True,
               encode_textures: bool = True,
               cache: ExportCache | None = None) -> BuildPlan:
    """Build the export plan.

    ``encode_textures=False`` leaves VTF artifacts empty (mesh-only export);
    with a ``cache``, textures unchanged since the last export are not
    encoded either.
    """
    resolver = GraphResolver()
    texture_keys, cached_assets = {}, set()
    if not encode_textures:
        _stub_textures(resolver, output)
        # Asset comparison is meaningless with empty textures.
        check_shared = False
    elif cache is not None:
        texture_keys, cached_assets = _cached_textures(
            resolver, output, _root(output.game_root, 'game root'), cache)
    try:
        model = collect_model(output, resolver)
        if not output.native_settings_confirmed:
            raise GraphBuildError('Review geometry up axis, collection merge '
                                  'and triangulation (and DMX version for '
                                  'DMX graphs) for this saved graph, then '
                                  'enable Native settings reviewed')
        content_root = _root(output.content_root, 'content root')
        game_root = _root(output.game_root, 'game root')
        qc_path = content_root / PurePosixPath(
            model.model_path).with_suffix('.qc')
        nodes = tuple(_upstream_nodes(output))
        materials = []
        for node in nodes:
            if isinstance(node, VMT_NODE_TYPES):
                # Encodes the linked VTFs.
                materials.append(resolver(node.outputs['Material']))
                _tick()
        materials = tuple(materials)
        if any(not isinstance(item, MaterialRecipe) for item in materials):
            raise GraphBuildError(
                'A linked Material node returned an invalid recipe')
        materials = tuple(dict.fromkeys(materials))
        # Resolve mesh slots through their bound Blender Materials, not VMT
        # filenames. Prefer the old same-named choice when multiple recipes
        # provide skins.
        material_names = {}
        for item in materials:
            name = item.material.name
            stem = PurePosixPath(item.vmt_path).stem
            if (name not in material_names
                    or stem.casefold() == name.casefold()):
                material_names[name] = stem
        geometries = model.geometries()
        for geometry in geometries:
            if geometry.auto_merge and not isinstance(geometry.source,
                                                      bpy.types.Collection):
                raise GraphBuildError(
                    'Auto-merge requires a Collection geometry source')
        vca_models = {item.geometry.key for item in model.models
                      if item.directive == 'MODEL'}
        for geometry in geometries:
            if geometry.vertex_animations and geometry.key not in vca_models:
                raise GraphBuildError('Vertex animations require a $model '
                                      'DMX geometry entry')
        rigs = {}
        for geometry in geometries:
            rig = resolve_geometry_rig(geometry.source, geometry.rig)
            joints = bool(model.collision
                          and model.collision.mode == 'JOINTS'
                          and geometry.key == model.collision.geometry.key)
            validate_geometry_weights(geometry.source, rig, joints=joints)
            rigs[geometry.key] = rig
        for lod in model.lods:
            if rigs[lod.source.key] != rigs[lod.geometry.key]:
                raise GraphBuildError('LOD replacement rig must match its '
                                      'source geometry rig')
        if model.collision and model.collision.mode == 'JOINTS':
            collision_rig = rigs[model.collision.geometry.key]
            model_rigs = {rigs[item.geometry.key] for item in model.models}
            if collision_rig not in model_rigs:
                raise GraphBuildError('$collisionjoints rig must be the rig '
                                      'used by a $model/$body entry')
        for geometry in geometries:
            for obj in _geometry_objects(geometry):
                if obj.type != 'MESH' and not (
                        obj.type == 'CURVE'
                        and (obj.data.bevel_depth or obj.data.extrude)):
                    continue
                for slot in obj.material_slots:
                    if (slot.material is not None
                            and slot.material.name not in material_names):
                        raise GraphBuildError(
                            f'No linked VMT bound to {obj.type.lower()} slot '
                            f'{slot.material.name!r} on {obj.name!r}')
        vmt_stems = {PurePosixPath(item.vmt_path).stem.casefold()
                     for item in materials}
        for family in model.skinfamilies:
            for name in family.materials:
                if name.casefold() not in vmt_stems:
                    raise GraphBuildError(
                        f'Skin material {name!r} has no linked VMT recipe')
        explicit = list(model.material_dirs)
        for item in materials:
            directory = str(PurePosixPath(item.vmt_path).parent.relative_to(
                'materials'))
            if directory != '.' and directory not in explicit:
                explicit.append(directory)
        extension = output.geometry_format.lower()
        geometry = tuple(
            (item, qc_path.parent.joinpath(
                *PurePosixPath(f'{item.key}.{extension}').parts))
            for item in geometries)
        paths = {item.key: path.relative_to(qc_path.parent).as_posix()
                 for item, path in geometry}
        if output.geometry_format == 'DMX':
            encoding_version = int(output.dmx_encoding_version)
            model_version = int(output.dmx_model_version)
            _validate_versions(encoding_version, model_version)
        else:
            encoding_version = model_version = 0
            if any(item.vertex_animations or item.stereo_balance_groups
                   or item.auto_merge for item in geometries):
                raise GraphBuildError('Vertex animations, stereo balance and '
                                      'auto-merge require DMX geometry')
        render = bpy.context.scene.render
        qc_text = render_qc(model, tuple(explicit), paths,
                            vertex_animation_fps=render.fps / render.fps_base)
        assets = _artifacts(materials, game_root)
        if model.sfm_rig is not None:
            assets += (_sfm_rig_script(model, rigs, game_root),)
        source_files = set()
        for node in nodes:
            if isinstance(node, SourceIOExternalTextureNode):
                source_path = node.source_path
            elif (isinstance(node, SourceIOImageSourceNode)
                  and node.image is not None):
                source_path = node.image.filepath
            else:
                continue
            if source_path:
                source_files.add(
                    Path(bpy.path.abspath(source_path)).resolve())
        for destination, _ in assets:
            if (destination.suffix.lower() == '.vtf'
                    and destination.resolve() in source_files):
                raise GraphBuildError(f'VTF output would overwrite source '
                                      f'image {destination}; choose a '
                                      'different VTF path')
        flex_sources = _flex_sources(model, output.geometry_format)
        plan = BuildPlan(model, content_root, game_root, qc_path,
                         output.geometry_format, encoding_version,
                         model_version, output.dmx_up_axis,
                         output.compiler_backend, output.compiler_command,
                         output.bst_engine_root, geometry, assets, qc_text,
                         tuple(material_names.items()), flex_sources,
                         frozenset(cached_assets),
                         tuple(texture_keys.items()),
                         output.dmx_trim_weights)
        if check_shared:
            own_assets = dict(plan.assets)
            for sibling in output.id_data.nodes:
                if sibling == output or not isinstance(sibling,
                                                       SourceIOPropModelNode):
                    continue
                if (not sibling.model_path.strip()
                        or not sibling.game_root.strip()
                        or not sibling.inputs['Models'].is_linked):
                    continue
                other = plan_model(sibling, check_shared=False)
                if other.game_root != game_root:
                    continue
                if (other.model_file == plan.model_file
                        or other.qc_path == plan.qc_path):
                    raise GraphBuildError(
                        f'Model outputs {output.name!r} and '
                        f'{sibling.name!r} share an output path')
                for destination, data in other.assets:
                    if destination in plan.cached_assets:
                        # Reused from disk: nothing new is written there.
                        continue
                    if (destination in own_assets
                            and own_assets[destination] != data):
                        raise GraphBuildError(
                            f'Conflicting material recipes at {destination} '
                            f'between {output.name!r} and {sibling.name!r}')
        return plan
    except (ValueError, KeyError, TypeError) as exc:
        raise GraphBuildError(str(exc)) from exc


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _merged_entry(entry, mesh_name):
    """Retarget original per-object graph rules to the one merged mesh."""
    if entry is None or entry.flex_rule_set is None:
        return entry
    rules = {}
    for rule in entry.flex_rule_set.rules:
        merged = replace(rule, object_name=mesh_name)
        previous = rules.get(rule.shape_name)
        if previous is not None and previous != merged:
            raise GraphBuildError(f'Auto-merged shape {rule.shape_name!r} '
                                  'has conflicting flex rules')
        rules[rule.shape_name] = merged
    return replace(entry, flex_rule_set=replace(
        entry.flex_rule_set, rules=tuple(rules.values())))


def preview_text(name: str) -> bpy.types.Text:
    """The preview Text called ``name``.

    Blender 4.5 cuts datablock names to 63 bytes, so there long names would
    not be found again and could collide; the full name is kept on the Text
    and looked up instead.
    """
    if is_blender_5():
        return bpy.data.texts.get(name) or bpy.data.texts.new(name)
    for text in bpy.data.texts:
        if text.get('sourceio_preview') == name:
            return text
    text = bpy.data.texts.new(name)
    text['sourceio_preview'] = name
    return text


def _write_if_changed(path: Path, data: bytes) -> bool:
    try:
        if path.read_bytes() == data:
            return False
    except OSError:
        pass
    _atomic_write(path, data)
    return True


def _write_assets(assets, cached=frozenset(), keys=(),
                  cache: ExportCache | None = None):
    """Write material assets.

    VMTs are written only when their text changed, VTFs unless reused from
    the cache.
    """
    keys = dict(keys)
    for path, data in assets:
        if path in cached:
            continue
        if path.suffix.lower() == '.vtf' and cache is None:
            _atomic_write(path, data)
        elif not _write_if_changed(path, data):
            if cache is not None and path in keys:
                # Identical bytes are already on disk.
                cache.record(path, keys[path], written=False)
            continue
        if cache is not None and path in keys:
            cache.record(path, keys[path])


def export_plan(plan: BuildPlan, cache: ExportCache | None = None):
    export_geometry(plan, cache)
    _write_assets(plan.assets, plan.cached_assets, plan.texture_keys, cache)
    _write_if_changed(plan.qc_path, plan.qc_text.encode('utf-8'))
    _tick()


def export_geometry(plan: BuildPlan, cache: ExportCache | None = None):
    """Write the DMX/SMD meshes (with flex data and VCA sequences) only.

    With a ``cache``, meshes whose inputs and settings are unchanged since
    the last export are skipped.
    """
    entries = {entry.geometry.key: entry
               for entry in plan.model.models + plan.model.flex_entries
               if entry.directive == 'MODEL'}
    for entry in entries.values():
        if (entry.flex_controller_source and bpy.data.texts.get(
                entry.flex_controller_source) is None):
            raise GraphBuildError(f'Flex controller Text '
                                  f'{entry.flex_controller_source!r} was '
                                  'removed during export')
    collision = plan.model.collision
    for geometry, path in plan.geometry:
        _tick()
        source = geometry.source
        rig = geometry.rig
        joints = bool(collision and collision.mode == 'JOINTS'
                      and geometry.key == collision.geometry.key)
        entry = entries.get(geometry.key)
        advanced_text = (
            bpy.data.texts[entry.flex_controller_source].as_string()
            if entry and entry.flex_controller_source else None)
        key = None
        if cache is not None:
            settings = (plan.geometry_format, plan.encoding_version,
                        plan.model_version, plan.up_axis, plan.model.scale,
                        plan.model.bone_names, plan.material_names, joints,
                        repr(entry), plan.trim_weights, advanced_text)
            try:
                key = geometry_fingerprint(
                    geometry, resolve_geometry_rig(source, rig), settings)
            except (ValueError, TypeError, RuntimeError,
                    ReferenceError) as exc:
                # The scene changed since planning, e.g. a deleted object.
                raise GraphBuildError(
                    f'Geometry {geometry.key!r} changed during export: '
                    f'{exc}') from exc
            if cache.fresh(path, key):
                continue
        if plan.geometry_format == 'SMD':
            try:
                capture = capture_geometry(
                    source, rig, material_names=dict(plan.material_names),
                    include_shapes=False)
                capture = transform_capture(capture, plan.model.scale,
                                            dict(plan.model.bone_names))
                capture = orient_capture(capture, plan.up_axis)
                write_smd(capture, path, joints=joints)
            except (ValueError, TypeError) as exc:
                raise GraphBuildError(
                    f'Native SMD {geometry.key!r}: {exc}') from exc
            if cache is not None:
                cache.record(path, key)
            continue
        try:
            options = dict(
                material_names=dict(plan.material_names),
                stereo_balance_groups=dict(geometry.stereo_balance_groups))
            if geometry.vertex_animations:
                capture = capture_vertex_animations(
                    source, geometry.vertex_animations, rig, **options)
            else:
                capture = capture_geometry(source, rig, **options)
            if geometry.auto_merge:
                capture = merge_collection_capture(capture)
                entry = _merged_entry(entry, capture.source_name)
            if entry is not None and entry.flex_rule_set is not None:
                capture = derive_rule_shapes(capture,
                                             entry.flex_rule_set.rules)
            capture = transform_capture(capture, plan.model.scale,
                                        dict(plan.model.bone_names))
            capture = orient_capture(capture, plan.up_axis)
            write_dmx(capture, path, encoding_version=plan.encoding_version,
                      model_version=plan.model_version, entry=entry,
                      advanced_text=advanced_text,
                      trim_weights=plan.trim_weights)
            if any(clip.export_sequence
                   for clip in geometry.vertex_animations):
                write_vca_sequences(capture, path.with_suffix(''),
                                    encoding_version=plan.encoding_version,
                                    model_version=plan.model_version)
        except (ValueError, TypeError) as exc:
            raise GraphBuildError(
                f'Native DMX {geometry.key!r}: {exc}') from exc
        if cache is not None:
            cache.record(path, key)


def export_materials(output: SourceIOPropModelNode,
                     cache: ExportCache | None = None) -> tuple[Path, ...]:
    """Write every linked VMT and VTF of a model output.

    Meshes, flexes and the QC are untouched.
    """
    game_root = _root(output.game_root, 'game root')
    resolver = GraphResolver()
    keys, cached = ({}, set()) if cache is None else _cached_textures(
        resolver, output, game_root, cache)
    try:
        materials = []
        for node in _upstream_nodes(output):
            if isinstance(node, VMT_NODE_TYPES):
                materials.append(resolver(node.outputs['Material']))
                _tick()
        materials = tuple(dict.fromkeys(materials))
    except (ValueError, KeyError, TypeError) as exc:
        raise GraphBuildError(str(exc)) from exc
    if not materials:
        raise GraphBuildError(f'{output.name}: no VMT nodes are linked')
    if any(not isinstance(item, MaterialRecipe) for item in materials):
        raise GraphBuildError(
            'A linked Material node returned an invalid recipe')
    assets = _artifacts(materials, game_root)
    _write_assets(assets, frozenset(cached), tuple(keys.items()), cache)
    return tuple(path for path, _ in assets)


def sync_vmt(material_node) -> tuple[Path, ...]:
    """Write only this node's VMT text (no texture encoding).

    It goes to the game root of every model output using the node.
    """
    tree = material_node.id_data
    outputs = [node for node in tree.nodes
               if isinstance(node, SourceIOPropModelNode)
               and material_node in set(_upstream_nodes(node))]
    if not outputs:
        raise GraphBuildError(f'{material_node.name}: connect it to a '
                              'Source 1 model output first')
    resolver = GraphResolver()
    _stub_textures(resolver, material_node)
    try:
        material = resolver(material_node.outputs['Material'])
    except (ValueError, KeyError, TypeError) as exc:
        raise GraphBuildError(str(exc)) from exc
    written = []
    for root in dict.fromkeys(_root(output.game_root, 'game root')
                              for output in outputs):
        path = root.joinpath(*PurePosixPath(material.vmt_path).parts)
        _atomic_write(path, material.vmt_text.encode('utf-8'))
        written.append(path)
    return tuple(written)


def compile_plan(plan: BuildPlan, cache: ExportCache | None = None) -> str:
    if plan.compiler_backend == 'COMMAND':
        argv = direct_command(plan.qc_path, plan.game_root,
                              plan.compiler_command)
    else:
        engine = _root(plan.bst_engine_root, 'BST engine root')
        state = validate_source_tools_compiler(plan.game_root, engine)
    export_plan(plan, cache)
    model_file = plan.model_file
    before = model_file.stat().st_mtime_ns if model_file.is_file() else None
    if plan.compiler_backend == 'COMMAND':
        output = compile_direct(plan.qc_path, argv)
    else:
        output = compile_with_source_tools(plan.qc_path, plan.game_root,
                                           engine, state)
    if not model_file.is_file() or model_file.stat().st_mtime_ns == before:
        detail = f'\n{output}' if output else ''
        raise CompilationError(f'Compiler returned without updating '
                               f'{model_file}{detail}')
    return output


class _GraphAction(bpy.types.Operator):
    output_node: bpy.props.StringProperty()
    action = ''

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return bool(space and space.type == 'NODE_EDITOR' and space.edit_tree
                    and space.edit_tree.bl_idname
                    == 'sourceio.model_definition')

    def execute(self, context):
        tree = context.space_data.edit_tree
        output = (tree.nodes.get(self.output_node) if self.output_node
                  else context.active_node)
        if not isinstance(output, SourceIOPropModelNode):
            self.report({'ERROR'}, 'Select a Source 1 model output node')
            return {'CANCELLED'}
        upstream = tuple(_upstream_nodes(output))
        materials = sum(isinstance(node, VMT_NODE_TYPES) for node in upstream)
        meshes = sum(isinstance(node, SourceIOPropGeometryNode)
                     for node in upstream)
        total = {'MATERIALS': materials, 'MESHES': meshes,
                 'PREVIEW': materials}.get(self.action,
                                           materials + meshes + 1)
        with export_progress(context.window_manager, total):
            return self._run(output)

    def _run(self, output):
        cache = model_cache(output) if self.action != 'PREVIEW' else None
        try:
            return self._action(output, cache)
        finally:
            if cache is not None:
                try:
                    # Also after a failure: files written so far stay
                    # recorded.
                    cache.save()
                except OSError as exc:
                    self.report({'WARNING'},
                                f'Could not save export cache: {exc}')

    def _cache_note(self, cache):
        if cache is None:
            return ''
        return f' ({cache.written} written, {cache.skipped} unchanged)'

    def _action(self, output, cache):
        try:
            if self.action == 'MATERIALS':
                written = export_materials(output, cache)
                self.report({'INFO'}, f'Material files under '
                                      f'{output.game_root}: {len(written)}'
                                      f'{self._cache_note(cache)}')
                return {'FINISHED'}
            if self.action == 'MESHES':
                plan = plan_model(output, encode_textures=False)
                export_geometry(plan, cache)
                self.report({'INFO'}, f'Mesh files: {len(plan.geometry)}'
                                      f'{self._cache_note(cache)}')
                return {'FINISHED'}
            # The preview only needs texture paths, never the encoded VTF
            # data.
            plan = plan_model(output,
                              encode_textures=self.action != 'PREVIEW',
                              cache=cache)
            if self.action == 'PREVIEW':
                text = preview_text(f'SourceIO export preview - {output.name}')
                text.clear()
                text.write(plan.manifest())
                for path, data in plan.assets:
                    suffix = path.suffix.lower()
                    if suffix not in ('.vmt', '.py'):
                        continue
                    relative = path.relative_to(plan.game_root).as_posix()
                    kind = 'VMT' if suffix == '.vmt' else 'SFM rig'
                    asset_text = preview_text(
                        f'SourceIO {kind} preview - {output.name} '
                        f'- {relative}')
                    asset_text.clear()
                    asset_text.write(data.decode('utf-8'))
                self.report({'INFO'},
                            f'Preview written to Blender text {text.name!r}')
            elif self.action == 'EXPORT':
                export_plan(plan, cache)
                self.report({'INFO'}, f'Exported {plan.qc_path}'
                                      f'{self._cache_note(cache)}')
            else:
                output_text = compile_plan(plan, cache)
                self.report({'INFO'}, f'Compiled {plan.model.model_path}; '
                                      f'{output_text[:180]}')
        except (GraphBuildError, CompilationError, OSError,
                RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class SourceIO_OP_PreviewExportGraph(_GraphAction):
    bl_idname = 'sourceio.preview_export_graph'
    bl_label = 'Preview Source 1 export'
    action = 'PREVIEW'


class SourceIO_OP_ExportModelGraph(_GraphAction):
    bl_idname = 'sourceio.export_model_graph'
    bl_label = 'Export Source 1 model assets'
    action = 'EXPORT'


class SourceIO_OP_ExportModelMaterials(_GraphAction):
    bl_idname = 'sourceio.export_model_materials'
    bl_label = 'Export materials and textures only'
    bl_description = ('Write the linked VMTs and VTFs; meshes, flexes and '
                      'the QC are not exported')
    action = 'MATERIALS'


class SourceIO_OP_ExportModelMeshes(_GraphAction):
    bl_idname = 'sourceio.export_model_meshes'
    bl_label = 'Export meshes only'
    bl_description = ('Write the DMX/SMD meshes with flexes; materials, '
                      'textures and the QC are not exported')
    action = 'MESHES'


class SourceIO_OP_GenerateBoneTable(bpy.types.Operator):
    bl_idname = 'sourceio.generate_bone_table'
    bl_label = 'Generate bone table'
    bl_description = ('Fill the rename table with every bone of the rigs '
                      'used by the connected model outputs; existing renames '
                      'are kept')
    rename_node: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return _GraphAction.poll(context)

    def execute(self, context):
        tree = context.space_data.edit_tree
        node = tree.nodes.get(self.rename_node) or context.active_node
        if getattr(node, 'bl_idname', '') != 'SourceIOBoneRenameNode':
            self.report({'ERROR'}, 'Select a bone rename table node')
            return {'CANCELLED'}
        outputs = [item for item in tree.nodes
                   if isinstance(item, SourceIOPropModelNode)
                   and node in set(_upstream_nodes(item))]
        if not outputs:
            self.report({'ERROR'},
                        'Connect the table to a Source 1 model output first')
            return {'CANCELLED'}
        rigs = []
        resolver = GraphResolver()
        for output in outputs:
            for geometry_node in _upstream_nodes(output):
                if not isinstance(geometry_node, SourceIOPropGeometryNode):
                    continue
                try:
                    geometry = resolver(geometry_node.outputs[0])
                    rig = (resolve_geometry_rig(geometry.source, geometry.rig)
                           if isinstance(geometry, GeometryRecipe) else None)
                except (ValueError, RuntimeError):
                    continue
                if rig is not None and rig not in rigs:
                    rigs.append(rig)
        if not rigs:
            self.report({'ERROR'},
                        'No armature found on the connected geometry')
            return {'CANCELLED'}
        try:
            existing = (dict(parse_bone_table(node.table.as_string()))
                        if node.table else {})
        except ValueError as exc:
            self.report({'ERROR'}, f'Fix the current table first: {exc}')
            return {'CANCELLED'}
        lines = ['# Blender bone name = exported bone name',
                 '# Edit the right side; lines may be removed.']
        listed = set()
        for rig in rigs:
            lines.append(f'# {rig.name}')
            for bone in rig.data.bones:
                if bone.name not in listed:
                    listed.add(bone.name)
                    target = existing.get(bone.name, bone.name)
                    lines.append(f'{bone.name} = {target}')
        stale = [(source, target) for source, target in existing.items()
                 if source not in listed]
        if stale:
            lines.append('# Not found on the connected rigs')
            lines.extend(f'{source} = {target}' for source, target in stale)
        text = node.table or bpy.data.texts.new(f'{node.name} bones')
        text.clear()
        text.write('\n'.join(lines) + '\n')
        node.table = text
        self.report({'INFO'},
                    f'Wrote {len(listed)} bones to {text.name!r}')
        return {'FINISHED'}


class SourceIO_OP_SyncVMT(bpy.types.Operator):
    bl_idname = 'sourceio.sync_vmt'
    bl_label = 'Sync VMT to disk'
    bl_description = 'Write this VMT file only (textures are not re-encoded)'
    material_node: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return _GraphAction.poll(context)

    def execute(self, context):
        node = (context.space_data.edit_tree.nodes.get(self.material_node)
                or context.active_node)
        if not isinstance(node, VMT_NODE_TYPES):
            self.report({'ERROR'}, 'Select a VMT node')
            return {'CANCELLED'}
        try:
            written = sync_vmt(node)
        except (GraphBuildError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'},
                    'Wrote ' + ', '.join(str(path) for path in written))
        return {'FINISHED'}


class SourceIO_OP_CompileModelGraph(_GraphAction):
    bl_idname = 'sourceio.compile_model_graph'
    bl_label = 'Export and compile Source 1 model'
    action = 'COMPILE'


class SOURCEIO_UL_FlexControllers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.prop(item, 'name', text='', emboss=False)
        row.prop(item, 'stereo', text='Stereo', toggle=True)


class SourceIO_OP_EditFlexControllers(bpy.types.Operator):
    bl_idname = 'sourceio.edit_flex_controllers'
    bl_label = 'Edit DMX flex controllers'
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(items=(
        ('ADD', 'Add', 'Add a controller for an unused shape key'),
        ('REMOVE', 'Remove', 'Remove the selected controller'),
        ('POPULATE', 'Populate',
         'Add controllers for all unassigned shape keys'),
    ))

    @classmethod
    def poll(cls, context):
        return (_GraphAction.poll(context)
                and isinstance(context.active_node, SourceIOModelEntryNode))

    def execute(self, context):
        entry = context.active_node
        if self.action == 'REMOVE':
            if not entry.flex_controllers:
                return {'CANCELLED'}
            last = len(entry.flex_controllers) - 1
            entry.flex_controllers.remove(
                min(entry.flex_controller_index, last))
            entry.flex_controller_index = max(
                0, min(entry.flex_controller_index, last - 1))
            return {'FINISHED'}
        try:
            geometry = GraphResolver()(entry.inputs['Geometry'])
            if geometry is None:
                raise GraphBuildError(
                    'Connect Geometry to the Model Entry first')
            shapes = _flex_shape_names(geometry)
            if not shapes:
                raise GraphBuildError('Linked geometry has no shape keys')
        except (GraphBuildError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        used_shapes = {raw for control in entry.flex_controllers
                       for raw in (control.raw_name, control.second_raw_name)
                       if raw}
        names = [name for name in shapes if name not in used_shapes]
        if self.action == 'ADD':
            names = names[:1]
        if not names:
            self.report({'INFO'}, 'Every shape key already has a controller')
            return {'CANCELLED'}
        used_names = {control.name for control in entry.flex_controllers}
        for shape_name in names:
            name = shape_name
            suffix = 1
            while name in used_names:
                name = f'{shape_name}.{suffix:03d}'
                suffix += 1
            control = entry.flex_controllers.add()
            control.name = name
            control.raw_name = shape_name
            used_names.add(name)
        entry.flex_controller_text = None
        entry.flex_controller_index = len(entry.flex_controllers) - 1
        return {'FINISHED'}


class NODE_PT_sourceio_export_graph(bpy.types.Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Group'
    bl_label = 'Source 1 Export'

    @classmethod
    def poll(cls, context):
        return _GraphAction.poll(context)

    def draw(self, context):
        layout = self.layout
        output = context.active_node
        if not isinstance(output, SourceIOPropModelNode):
            layout.label(text='Select a model output node')
            return
        layout.label(text=output.model_path
                     or 'Set a game-relative model path')
        for op in (SourceIO_OP_PreviewExportGraph,
                   SourceIO_OP_ExportModelGraph,
                   SourceIO_OP_ExportModelMeshes,
                   SourceIO_OP_ExportModelMaterials,
                   SourceIO_OP_CompileModelGraph):
            layout.operator(op.bl_idname).output_node = output.name


class NODE_PT_sourceio_flex_rules(bpy.types.Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Group'
    bl_label = 'DMX Flex Controllers'

    @classmethod
    def poll(cls, context):
        return (_GraphAction.poll(context)
                and isinstance(context.active_node, SourceIOModelEntryNode))

    def draw(self, context):
        entry = context.active_node
        layout = self.layout
        if ('Flex Rules' in entry.inputs
                and entry.inputs['Flex Rules'].is_linked):
            layout.label(text='Flex rule group connected', icon='NODETREE')
            layout.label(text='Use Edit group for controllers and expressions')
            return
        row = layout.row()
        row.template_list('SOURCEIO_UL_FlexControllers', '', entry,
                          'flex_controllers', entry, 'flex_controller_index',
                          rows=4)
        buttons = row.column(align=True)
        edit = SourceIO_OP_EditFlexControllers.bl_idname
        buttons.operator(edit, text='', icon='ADD').action = 'ADD'
        buttons.operator(edit, text='', icon='REMOVE').action = 'REMOVE'
        layout.operator(edit,
                        text='Add shape-key controls').action = 'POPULATE'
        if entry.flex_controllers:
            control = entry.flex_controllers[
                min(entry.flex_controller_index,
                    len(entry.flex_controllers) - 1)]
            box = layout.box()
            box.prop(control, 'name')
            box.prop(control, 'raw_name', text='Shape')
            box.prop(control, 'second_raw_name', text='Second')
            row = box.row(align=True)
            row.prop(control, 'stereo')
            row.prop(control, 'eyelid')
            box.prop(control, 'minimum', text='Min')
            box.prop(control, 'maximum', text='Max')
            box.prop(control, 'wrinkle_scale')
            if control.second_raw_name:
                box.prop(control, 'second_wrinkle_scale')
        else:
            layout.separator()
            layout.label(text='Advanced DMX controller Text')
            layout.prop(entry, 'flex_controller_text')
            if entry.flex_controller_text is None:
                layout.label(text='Automatic controls use shape keys')


classes = (SourceIO_OP_PreviewExportGraph, SourceIO_OP_ExportModelGraph,
           SourceIO_OP_ExportModelMeshes, SourceIO_OP_ExportModelMaterials,
           SourceIO_OP_SyncVMT, SourceIO_OP_GenerateBoneTable,
           SourceIO_OP_CompileModelGraph, SOURCEIO_UL_FlexControllers,
           SourceIO_OP_EditFlexControllers, NODE_PT_sourceio_export_graph,
           NODE_PT_sourceio_flex_rules)
