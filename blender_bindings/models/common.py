import bpy
import numpy as np

from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import get_new_unique_collection
from SourceIO.library.models.mdl.structs.model import Model
from SourceIO.library.models.vtx.v7.structs.lod import ModelLod as VtxModel
from SourceIO.library.models.vtx.v7.structs.mesh import Mesh as VtxMesh
from SourceIO.library.models.mdl import Mdl

from SourceIO.library.models.mdl.load_animations import load_all_animations, load_mdl_animations
from SourceIO.blender_bindings.models.import_animations import import_animations_to_armature
from SourceIO.library.utils import Buffer
from SourceIO.library.shared.content_manager import ContentManager

def import_animations_common(mdl, buffer: Buffer, content_manager: ContentManager, model_path: str, scale: float, compact_animations: bool, include_all: bool, armature: bpy.types.Object):
    #if options.import_animations and armature:
    if include_all:
        animations = load_all_animations(mdl, buffer, content_manager, model_path)
    else:
        animations = load_mdl_animations(mdl, buffer, content_manager, model_path)
        
    import_animations_to_armature(armature, model_path, animations, scale, compact_animations)

def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indices, vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1).astype(np.uint32))
        vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
    return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset


def merge_meshes(model: Model, vtx_model: VtxModel):
    vtx_vertices = []
    acc = 0
    mat_arrays = []
    indices_array = []
    for n, (vtx_mesh, mesh) in enumerate(zip(vtx_model.meshes, model.meshes)):

        if not vtx_mesh.strip_groups:
            continue

        vertex_start = mesh.vertex_index_start
        indices, vertices, offset = merge_strip_groups(vtx_mesh)
        indices = np.add(indices, acc)
        mat_array = np.full(indices.shape[0] // 3, mesh.material_index)
        mat_arrays.append(mat_array)
        vtx_vertices.extend(np.add(vertices, vertex_start))
        indices_array.append(indices)
        acc += offset

    return vtx_vertices, np.hstack(indices_array), np.hstack(mat_arrays)


def put_into_collections(model_container: ModelContainer, model_name,
                         parent_collection=None, bodygroup_grouping=False):
    master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)
    if model_container.bodygroups:
        for bodygroup_name, meshes in model_container.bodygroups.items():
            if bodygroup_grouping:
                body_part_collection = get_new_unique_collection(bodygroup_name, master_collection)
            else:
                body_part_collection = master_collection

            for mesh in meshes:
                if mesh == None:
                    continue
                body_collection = get_new_unique_collection(mesh.name, body_part_collection)
                body_collection.objects.link(mesh)
    else:
        for obj in model_container.objects:
            master_collection.objects.link(obj)
    if model_container.armature:
        master_collection.objects.link(model_container.armature)

    if model_container.attachments:
        attachments_collection = get_new_unique_collection(model_name + '_ATTACHMENTS', master_collection)
        for attachment in model_container.attachments:
            attachments_collection.objects.link(attachment)
    if model_container.physics_objects:
        physics_collection = get_new_unique_collection(model_name + '_PHYSICS', master_collection)
        for physics in model_container.physics_objects:
            physics_collection.objects.link(physics)
    model_container.master_collection = master_collection
    return master_collection

def create_eyeballs(mdl: Mdl, armature: bpy.types.Object, mesh_obj: bpy.types.Object, model: Model, scale: float, extra_stuff: list):
    from math import atan
    from mathutils import Matrix, Vector

    eyeballs = model.eyeballs
    for mesh in model.meshes:
        if mesh.material_type != 1:
            continue
        eyeball = eyeballs[mesh.material_param]
        eyeball_name = eyeball.name or f'eye_{mesh.material_param}'
        forward = Vector(eyeball.forward)
        up = Vector(eyeball.up)
        eyeball_obj = bpy.data.objects.new(eyeball_name, None)
        #eyeball_obj['forward_debug'] = forward
        #eyeball_obj['up_debug'] = up
        eyeball_obj.show_in_front = True
        extra_stuff.append(eyeball_obj)

        eyeball_pos = Vector(eyeball.org) * scale
        eyeball_matrix_rotation = Matrix(
            [
                forward.cross(up),
                forward,
                up
            ]
        ).transposed()
        
        eyeball_obj.location = eyeball_pos
        eyeball_obj.rotation_mode = 'QUATERNION'
        eyeball_obj.scale = [scale]*3
        eyeball_obj.empty_display_type = 'SPHERE'

        con = eyeball_obj.constraints.new('CHILD_OF')
        con.target = armature
        con.subtarget = mdl.bones[eyeball.bone_index].name
        con.inverse_matrix.identity()
        eye_material = mdl.materials[mesh.material_index].bpy_material
        eye_material['eye_source'] = eyeball_obj
        eyeball_obj['eye_material'] = eye_material

        eyeball_obj.rotation_quaternion = eyeball_matrix_rotation.to_quaternion()

        locs, rots, scales = ['LOC_X', 'LOC_Y', 'LOC_Z'], ['ROT_W', 'ROT_X', 'ROT_Y', 'ROT_Z'], ['SCALE_X', 'SCALE_Y', 'SCALE_Z']

        prop_loc = eyeball_name + '_loc'
        prop_rot = eyeball_name + '_rot'
        prop_scale = eyeball_name + '_scale'
        mesh_obj[prop_loc] = [0.0]*3
        mesh_obj[prop_rot] = [0.0]*4 # quaternion
        mesh_obj[prop_scale] = [0.0]*3
        drivers_loc = mesh_obj.driver_add(f'["{prop_loc}"]')
        drivers_rot = mesh_obj.driver_add(f'["{prop_rot}"]')
        drivers_scale = mesh_obj.driver_add(f'["{prop_scale}"]')

        def get_obj_transforms_driver(drivers, transform_type, do_quaternion=False):
            for driver, transform_type in zip(drivers, transform_type):
                driver = driver.driver
                driver.type = 'AVERAGE'
                var = driver.variables.new()
                var.type = 'TRANSFORMS'
                var.targets[0].id = eyeball_obj
                if do_quaternion:
                    var.targets[0].rotation_mode = 'QUATERNION'
                var.targets[0].transform_type = transform_type
                
        get_obj_transforms_driver(drivers_loc, locs)
        get_obj_transforms_driver(drivers_rot, rots, True)
        get_obj_transforms_driver(drivers_scale, scales)

        mesh_obj[eyeball_name+'_iris_scale'] = 1/eyeball.iris_scale
        eyeball_obj.empty_display_size = 1/eyeball.iris_scale
        mesh_obj[eyeball_name+'_z_offset'] = atan(eyeball.z_offset)

        if (nodes := getattr(eye_material.node_tree, 'nodes', None)):
            if nodes.get('!EYE_LOC'):
                nodes['!EYE_LOC'].attribute_name = prop_loc
            if nodes.get('!EYE_ROT'):
                nodes['!EYE_ROT'].attribute_name = prop_rot
            if nodes.get('!EYE_LOC'):
                nodes['!EYE_SCALE'].attribute_name = prop_scale
            if nodes.get('!EYE_Z'):
                nodes['!EYE_Z'].attribute_name = eyeball_name + '_z_offset'
            if nodes.get('!EYE_IRIS_SCALE'):
                nodes['!EYE_IRIS_SCALE'].attribute_name = eyeball_name + '_iris_scale'

def make_bodygroup_selectors(mdl: Mdl, armature: bpy.types.Object, bodygroups: dict[str, list[bpy.types.Object]]):
    from string import ascii_lowercase

    def add_vis_drivers(
        controller: bpy.types.Object,
        subject: bpy.types.Object,
        data_path: str,
        index: int
    ):
        controller.update_tag()
        for path in ['hide_viewport', 'hide_render']:
            subject.driver_remove(path)
            curve = subject.driver_add(path)
            driver = curve.driver
            driver.type = 'SCRIPTED'
            var = driver.variables.new()
            targs = var.targets[0]
            targs.id_type = 'OBJECT'
            targs.id = controller
            targs.data_path = f'["{data_path}"]'
            driver.expression = f'var != {index}'

    bg_name_map = dict()

    def tally():
        for i in range(999):
            yield ''.join(map(lambda a: ascii_lowercase[int(a)], f'{i}'))
    tally = tally()

    for n, body_part in enumerate(mdl.body_parts):
        if len(body_part.models) < 2:
            continue

        enum_items = []
        bg_name = body_part.name
        bg_name_suffix = 'BG' + ' ' + next(tally) + ' ' + bg_name
        bg_name_map[bg_name] = bg_name_suffix

        armature[bg_name_suffix] = 0

        for index, (bpy_model, model) in enumerate(zip(bodygroups[bg_name], body_part.models)):
            enum_items.append((
                f'{n}',
                model.name,
                ''
            ))
            if bpy_model == None:
                continue
            add_vis_drivers(
                armature,
                bpy_model,
                bg_name_suffix,
                index
            )
        
        ui_settings = armature.id_properties_ui(bg_name_suffix)
        ui_settings.update(
            min=0,
            max=len(enum_items),
            items=enum_items
        )
    
    armature['bodygroup_name_map'] = bg_name_map


def generate_wrinkle_map_node_group(obj: bpy.types.Object):
    data: bpy.types.Mesh = obj.data
    shape_keys = data.shape_keys

    compress = list(
        filter(
            lambda a: a.name.startswith('WR.') and a.name.endswith('.C'),
            data.attributes
        )
    )
    stretch = list(
        filter(
            lambda a: a.name.startswith('WR.') and a.name.endswith('.S'),
            data.attributes
        )
    )

    if not len(compress) + len(stretch):
        return

    node_group: bpy.types.GeometryNodeTree = bpy.data.node_groups.new(f'wrinkles_{obj.name}'[:63], 'GeometryNodeTree')
    nodes = node_group.nodes
    links = node_group.links
    mod: bpy.types.NodesModifier = obj.modifiers.new('Wrinkle Map Data', 'NODES')
    mod.node_group = node_group
    if bpy.app.version >= (4, 0, 0):
        node_group.interface.new_socket(name='Output', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name='Input', in_out='INPUT', socket_type='NodeSocketGeometry')
    else:
        node_group.inputs.new('NodeSocketGeometry', 'Input')
        node_group.outputs.new('NodeSocketGeometry', 'Output')
    
    input = nodes.new('NodeGroupInput')
    input.location = [400, 100]
    output = nodes.new('NodeGroupOutput')
    output.location = [800, 0]
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = [400, 0]
    store = nodes.new('GeometryNodeStoreNamedAttribute')
    store.data_type = 'FLOAT2'
    store.domain = 'POINT'
    store.inputs[2].default_value = 'tension'
    store.location = [600, 0]

    links.new(combine.outputs[0], store.inputs[3])
    links.new(input.outputs[0], store.inputs[0])
    links.new(store.outputs[0], output.inputs[0])


    loc_compress = [0, 0]
    loc_stretch = [200, 0]
    
    for n, attr in [*enumerate(compress), *enumerate(stretch)]:
        loc, index = (loc_compress, 0) if attr.name.endswith('.C') else (loc_stretch, 1)
        shape_name = attr.name.split('.')[1]
        
        wrinkle = nodes.new('GeometryNodeInputNamedAttribute')
        wrinkle.inputs[0].default_value = attr.name
        wrinkle.location = loc
        wrinkle.name = 'WRINKLE MAP'
        wrinkle.label = shape_name

        mult = nodes.new('ShaderNodeMath')
        mult.operation = 'MULTIPLY'
        mult.location = loc
        mult.name = 'SHAPEKEY VALUE'
        mult.label = shape_name
        driver = mult.inputs[1].driver_add('default_value')
        var = driver.driver.variables.new()
        targ = var.targets[0]
        targ.id_type = 'KEY'
        targ.id = shape_keys
        targ.data_path = shape_keys.key_blocks[shape_name].path_from_id('value')
        driver.driver.type = 'AVERAGE'

        links.new(wrinkle.outputs[0], mult.inputs[0])

        if n == 0:
            last = mult.outputs[0]
            links.new(last, combine.inputs[index])
            continue
        maximum = nodes.new('ShaderNodeMath')
        maximum.name = 'MAXIMUM'
        maximum.operation = 'MAXIMUM'
        maximum.location = loc
        links.new(last, maximum.inputs[0])
        links.new(mult.outputs[0], maximum.inputs[1])
        links.new(maximum.outputs[0], combine.inputs[index])
        last = maximum.outputs[0]

def create_flex_drivers(obj, mdl):
    from string import ascii_lowercase
    from SourceIO.library.models.mdl.structs.flex import FlexController, FlexControllerUI, FlexOpType, FlexRule
    if not obj.data.shape_keys:
        return
    
    #nway_expr = 'max(min(({0}-{1})/({2}-{1}),({4}-{0})/({4}-{3})),0)'
    #two_way_0_expr = 'clamp({}*-1)'
    #two_way_1_expr = 'clamp({})'
    #upper_eye_expr = '(1-abs(min({}, 0)))*{}*{}'
    #lower_eye_expr = '(1-abs(max({}, 0)))*(1-{})*{}'

    all_exprs: list[tuple[str, tuple]] = mdl.rebuild_flex_rules()
    bpy.types.Scene.t = all_exprs # debug point
    data: bpy.types.Mesh = obj.data
    shape_keys = data.shape_keys
    kb = shape_keys.key_blocks

    def make_custom_property(name, min, max, default=0.0):
        obj.data[name] = default
        prop = obj.data.id_properties_ui(name)
        prop.update(min=min, max=max)

    def tally():
        for i in range(999):
            yield ''.join(
                map(
                    lambda a: ascii_lowercase[int(a)],
                    f'{i:03d}'
                )
            )
    tally = tally()

    flexcontrollers = dict()
    flexmap = dict()
    
    flexmap['flex_scale'] = flex_sort = f'{next(tally)}_fs'
    flex = dict()
    flex['controller'] = 'flex_scale'
    flex['type'] = 0b00
    make_custom_property(flex_sort, -10, 10, 1.0)

    flexcontrollers['Flex Scale'] = flex

    for flex_controller_ui in mdl.flex_ui_controllers:
        flex = dict()
        
        if flex_controller_ui.stereo:
            flex['type'] = 0b01
            left_controller = next(
                filter(lambda a: a.name == flex_controller_ui.left_controller,
                       mdl.flex_controllers
                )
            )
            right_controller = next(
                filter(lambda a: a.name == flex_controller_ui.right_controller,
                       mdl.flex_controllers
                )
            )
            flexmap[left_controller.name] = left_sort = f'{next(tally)}_{left_controller.name}'
            flexmap[right_controller.name] = right_sort = f'{next(tally)}_{right_controller.name}'
            flex['left'] = left_controller.name
            flex['right'] = right_controller.name
            make_custom_property(left_sort, left_controller.min, left_controller.max)
            make_custom_property(right_sort, right_controller.min, right_controller.max)
        else:
            flex['type'] = 0b00
            controller = next(filter(lambda a: a.name == flex_controller_ui.controller, mdl.flex_controllers))
            flexmap[controller.name] = controller_sort = f'{next(tally)}_{controller.name}'
            flex['controller'] = controller.name
            make_custom_property(controller_sort, controller.min, controller.max)
        
        if flex_controller_ui.nway_controller:
            flex['type'] |= 0b10
            nway = next(
                filter(
                    lambda a: a.name == flex_controller_ui.nway_controller,
                    mdl.flex_controllers
                )
            )
            flexmap[nway.name] = nway_sort = f'{next(tally)}_{nway.name}'
            flex['nway'] = nway.name
            make_custom_property(nway_sort, nway.min, nway.max)

        flexcontrollers[flex_controller_ui.name] = flex
    
    obj.data['flexmap'] = flexmap
    obj.data['flexcontrollers'] = flexcontrollers

    def var_tally():
        for i in range(999):
            yield 'V'+str(i)

    shape_keys = obj.data.shape_keys

    visited_flexes = set()
    de_duplicated_exprs = list()
    de_duped_vars = dict()

    for name, (expr, inputs) in reversed(all_exprs):
        de_duped_name = name
        dupe_count = 0
        de_duped_self_input = name + '_' + f'{dupe_count+1:03d}'

        while de_duped_name in visited_flexes:
            dupe_count += 1
            de_duped_name = name + '_' + f'{dupe_count:03d}'
            de_duped_self_input = name + '_' + f'{dupe_count+1:03d}'

        expr:str = expr.as_simple()
        de_duped_inputs = []

        de_duped_vars[de_duped_name] = de_duped_self_input
        for input, type in sorted(set(inputs), key=lambda a: len(a[0]), reverse=True):
            if type == 'fetch2':
                de_duped_var = de_duped_vars.get(input, input)
                de_duped_inputs.append(
                    (
                        de_duped_var,
                        type
                    )
                )
                expr = expr.replace(input, de_duped_var)
            else:
                de_duped_inputs.append(
                    (
                        input,
                        type
                    )
                )

        visited_flexes.add(de_duped_name)
        de_duplicated_exprs.append((de_duped_name, (expr, de_duped_inputs)))

    all_exprs = dict(reversed(de_duplicated_exprs))

    for name, (expr, inputs) in all_exprs.items():
        #expr = expr.as_simple()
        vtally = var_tally()
        is_kb = False
        if kb.get(name):
            kb[name].driver_remove('value')
            driv = kb[name].driver_add('value')
            is_kb = True
        else:
            shape_keys[name] = 0.0
            driv = shape_keys.driver_add(f'["{name}"]')

        [driv.modifiers.remove(mod) for mod in driv.modifiers]
        driv = driv.driver

        if '_' in name: # likely a combo expression
            match name[-1]:
                case 'R':
                    side = 'R'
                case 'L':
                    side = 'L'
                case _:
                    side = ''
                
            combo_inputs = set()
            combo_flexes = name.rstrip(side).split('_')
            expr_inputs = set(inputs)

            for combo in combo_flexes:
                combo += side
                combo = all_exprs.get(combo, False)
                if not combo: break
                combo_inputs.update(set(combo[1]))

            # all inputs match, most likely a product of the results of all mentioned flexes. take a shortcut instead
            if combo_inputs == expr_inputs:
                all_vars = []
                for combo in combo_flexes:
                    combo += side
                    var = driv.variables.new()
                    var.name = next(vtally); all_vars.append(var.name)
                    var.type = 'SINGLE_PROP'
                    targ = var.targets[0]

                    if kb.get(combo):
                        data_path = kb[combo].path_from_id('value')
                    elif shape_keys.get(combo):
                        data_path = f'["{combo}"]'
                    else:
                        shape_keys[combo] = 0.0
                        data_path = f'["{combo}"]'

                    targ.id_type = 'KEY'
                    targ.id = shape_keys
                    targ.data_path = data_path

                expr = f'clamp({"*".join(all_vars)})/(pow(FS, {len(all_vars)-1})+1e-16)'

                var = driv.variables.new()
                var.name = 'FS'
                var.type = 'SINGLE_PROP'
                targ = var.targets[0]
                targ.id_type = 'MESH'
                targ.id = data
                targ.data_path = '["{}"]'.format(flexmap['flex_scale'])

                driv.expression = expr

                continue

        # normal expressions
        for input, type in sorted(set(inputs), key=lambda a: len(a[0]), reverse=True):
            #if input == name and type == 'fetch2':
            #    expr = expr.replace(input, '1')
            #    continue

            var = driv.variables.new()
            var.name = next(vtally)
            var.type = 'SINGLE_PROP'
            targ = var.targets[0]
            if type == 'fetch2':
                if kb.get(input):
                    data_path = kb[input].path_from_id('value')
                elif shape_keys.get(input):
                    data_path = f'["{input}"]'
                else:
                    shape_keys[input] = 0.0
                    data_path = f'["{input}"]'

                targ.id_type = 'KEY'
                targ.id = shape_keys
                targ.data_path = data_path
            else:
                targ.id_type = 'MESH'
                targ.id = data
                targ.data_path = f'["{flexmap[input]}"]'
            
            expr = expr.replace(input, var.name)
            
            
        var = driv.variables.new()
        var.name = 'FS'
        var.type = 'SINGLE_PROP'
        targ = var.targets[0]
        targ.id_type = 'MESH'
        targ.id = data
        targ.data_path = '["{}"]'.format(flexmap['flex_scale'])
        #print(expr)
        try:
            assert len(expr) < 256
        except AssertionError:
            print(expr)
            print(name)
        #driv.expression = ('clamp' if is_kb else '') + '(' + expr.replace('--', '+') + ')' + '*FS'
        driv.expression = '(' + expr.replace('--', '+') + ')' + '*FS'

    return