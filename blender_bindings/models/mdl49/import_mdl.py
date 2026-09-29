from SourceIO.blender_bindings.models.materials import get_model_material_names
import itertools
import warnings
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector
from math import atan

from SourceIO.blender_bindings.models.common import merge_meshes, create_eyeballs, generate_wrinkle_map_node_group, make_bodygroup_selectors, create_flex_drivers
from SourceIO.blender_bindings.models.mdl44.import_mdl import create_armature
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.utils.bpy_utils import add_material, is_blender_4_1, get_or_create_material, ActionCurveFactory
from SourceIO.blender_bindings.utils.fast_mesh import FastMesh
from SourceIO.library.models.mdl.structs.header import StudioHDRFlags
from SourceIO.library.models.mdl.v44.vertex_animation_cache import preprocess_vertex_animation
from SourceIO.library.models.mdl.v49.flex_expressions import *
from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
from SourceIO.library.models.vtx.v7.vtx import Vtx
from SourceIO.library.models.vvd import Vvd
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils.common import get_slice
from SourceIO.library.utils.path_utilities import path_stem
from SourceIO.logger import SourceLogMan
#from string import ascii_lowercase

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::ModelLoader')



def import_model(content_manager: ContentManager, mdl: MdlV49, vtx: Vtx, vvd: Vvd,
                 options: ModelOptions):
    #full_material_names = collect_full_material_names([mat.name for mat in mdl.materials], mdl.materials_paths, content_manager)
                 #scale=1.0, create_drivers=False, load_refpose=False):
    full_material_names = get_model_material_names(content_manager, mdl)
    [setattr(mat, 'bpy_material', get_or_create_material(mat.name, full_material_names[mat.name])) for mat in mdl.materials if mat.bpy_material is None]
    skin_groups = {str(n): list(map(lambda a: a.bpy_material, group)) for (n, group) in enumerate(mdl.skin_groups)}
    # ensure all MaterialV49 has its bpy_material counterpart
    objects = []
    bodygroups = defaultdict(list)
    attachments = []
    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]
    extra_stuff = []
    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    vert_anim_fixed_point_scale = mdl.header.vert_anim_fixed_point_scale if (mdl.header.flags & StudioHDRFlags.VERT_ANIM_FIXED_POINT_SCALE !=0 ) else 1/4096
    armature = None
    vertex_anim_cache = preprocess_vertex_animation(mdl, vvd)
    
    scale = options.scale
    create_drivers = options.create_flex_drivers
    debug_stereo_balance = options.debug_stereo_balance
    load_refpose = options.load_refpose
    
    if not static_prop:
        armature = create_armature(mdl, scale, load_refpose)

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):

            if model.vertex_count == 0:
                bodygroups[body_part.name].append(None)
                continue
            object_name = model.name
            mesh_name = f'{mdl.header.name}_{body_part.name}_{object_name}_MESH'
            mesh_data = FastMesh.new(mesh_name)
            mesh_obj = bpy.data.objects.new(object_name, mesh_data)

            mesh_obj['active_skin'] = '0'
            mesh_obj['model_type'] = 's1'
            default_skin_groups = {str(n): list(map(lambda a: a.bpy_material.name, group)) for (n, group) in enumerate(mdl.skin_groups)}

            objects.append(mesh_obj)
            bodygroups[body_part.name].append(mesh_obj)
            mesh_obj['prop_path'] = path_stem(mdl.header.name)

            model_vertices = get_slice(all_vertices, model.vertex_offset, model.vertex_count)
            vtx_vertices, indices_array, material_indices_array = merge_meshes(model, vtx_model.model_lods[desired_lod])

            indices_array = np.array(indices_array, dtype=np.uint32)
            vertices = model_vertices[vtx_vertices]
            vertices_vertex = vertices['vertex']

            mesh_data.from_pydata(vertices_vertex * scale, [], np.flip(indices_array).reshape((-1, 3)))
            mesh_data.update()

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
            if not is_blender_4_1():
                mesh_data.use_auto_smooth = True

            material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                material = get_or_create_material(mat_name, full_material_names[mat_name])
                mdl.materials[mat_id].bpy_material = material
                material_remapper[mat_id] = add_material(material, mesh_obj)

            try:
                mesh_obj['skin_groups'] = skin_groups
            except:
                mesh_obj['skin_groups'] = default_skin_groups

            mesh_data.polygons.foreach_set('material_index', material_remapper[material_indices_array[::-1]])

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)

            uv_data = mesh_data.uv_layers.new()
            uvs = vertices['uv']
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())

            if vvd.extra_data:
                for extra_type, extra_data in vvd.extra_data.items():
                    extra_data = extra_data.reshape((-1, 2))
                    extra_uv = get_slice(extra_data, model.vertex_offset, model.vertex_count)
                    extra_uv = extra_uv[vtx_vertices]
                    uv_data = mesh_data.uv_layers.new(name=extra_type.name)
                    extra_uv[:, 1] = 1 - extra_uv[:, 1]
                    uv_data.data.foreach_set('uv', extra_uv[vertex_indices].flatten())

            if not static_prop:
                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = armature
                mesh_obj.parent = armature

                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in mdl.bones}

                for n, (bone_indices, bone_weights) in enumerate(zip(vertices['bone_id'], vertices['weight'])):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = mdl.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

                flexes = []
                for mesh in model.meshes:
                    if mesh.flexes:
                        flexes.extend([(mdl.flex_names[flex.flex_desc_index], flex) for flex in mesh.flexes])

                if flexes:
                    mesh_obj.shape_key_add(name='base')

                    if debug_stereo_balance:
                        # debug tool to get stereo flex balances. i'll leave it here just in case
                        side_right = mesh_obj.vertex_groups.new(name='blendright')
                        side_left = mesh_obj.vertex_groups.new(name='blendleft')
                        side_all = np.zeros(model.vertex_count, dtype=np.float32)

                        for flex_name, flex_desc in flexes:
                            vertex_animation = vertex_anim_cache[flex_name]
                            side = get_slice(vertex_animation['side'], model.vertex_offset, model.vertex_count).ravel()
                            side_all = np.maximum(side_all, side)
                        side_all = side_all[vtx_vertices] + 0.0

                        for n, vert in enumerate(vtx_vertices):
                            side_right.add([n], side_all[n], 'REPLACE')
                            side_left.add([n], 1-side_all[n], 'REPLACE')

                    for flex_name, flex_desc in flexes:
                        vertex_animation = vertex_anim_cache[flex_name]
                        flex_delta = get_slice(vertex_animation["pos"], model.vertex_offset, model.vertex_count)
                        flex_delta = flex_delta[vtx_vertices] * scale
                        model_vertices = get_slice(all_vertices['vertex'], model.vertex_offset, model.vertex_count)
                        model_vertices = model_vertices[vtx_vertices] * scale

                        side = get_slice(vertex_animation["side"], model.vertex_offset, model.vertex_count)
                        side = side[vtx_vertices] + 0.0
                        wrinkle = get_slice(vertex_animation["wrinkle"], model.vertex_offset, model.vertex_count)
                        wrinkle = wrinkle[vtx_vertices] + 0.0 # this will have to be explained to me :P
                        # model.vertex_count and vtx_vertices can differ in size, so doing something like this just makes it work?
                        # apparently vtx_vertices has duplicate indicies, which can be observed by turning it into a set.
                        # i'm just following here
                        # -hisanimations

                        if flex_desc.partner_index:
                            partner_name = mdl.flex_names[flex_desc.partner_index]
                            flexes, sides = [flex_name, partner_name], [1-side, side] if not debug_stereo_balance else [1.0, 1.0]
                        else:
                            flexes, sides = [flex_name], [1.0]

                        for n, (flex_name, side) in enumerate(zip(flexes, sides)):
                            shape_key = mesh_data.shape_keys.key_blocks.get(flex_name, None) or mesh_obj.shape_key_add(
                                name=flex_name)
                            shape_key.data.foreach_set("co", (flex_delta*side + model_vertices).ravel())
                            shape_key.value = 0.0
                            shape_key.slider_min = -10.0
                            shape_key.slider_max = 10.0
                            if flex_desc.partner_index and debug_stereo_balance:
                                if n == 0: shape_key.vertex_group = side_left.name
                                else: shape_key.vertex_group = side_right.name

                            if flex_desc.vertex_anim_type == 1:
                                # establish it as an attribute for wrinkle maps and designate it as a stretch or compress map
                                if wrinkle.max() > 0:
                                    wrinkle_name = f'WR.{flex_name}.S'
                                elif wrinkle.min() < 0:
                                    wrinkle_name = f'WR.{flex_name}.C'
                                else:
                                    continue
                                attr = mesh_data.attributes.get(wrinkle_name, None) or mesh_data.attributes.new(wrinkle_name, 'FLOAT', 'POINT')
                                wrinkle_data = (abs(wrinkle) * vert_anim_fixed_point_scale) * side
                                attr.data.foreach_set('value', wrinkle_data.ravel())


                    if create_drivers:
                        create_flex_drivers(mesh_obj, mdl)
                    if options.generate_wrinkle_map_node_group:
                        generate_wrinkle_map_node_group(mesh_obj)
            
            mesh_data.validate()

            if model.has_eyeballs:
                create_eyeballs(mdl, armature, mesh_obj, model, scale, extra_stuff)

    if mdl.attachments:
        attachments = create_attachments(mdl, armature if not static_prop else objects[0], scale)
    if not static_prop:
        if options.bodygroup_vis_switches:
            make_bodygroup_selectors(mdl, armature, bodygroups)
    attachments.extend(extra_stuff)

    return ModelContainer(objects, bodygroups, [], attachments, armature, None)


def create_flex_drivers_old(obj, mdl: MdlV49):
    from ...operators.flex_operators import SourceIO_PG_FlexController
    from SourceIO.library.models.mdl.structs.flex import FlexController, FlexControllerUI, FlexOpType, FlexRule
    if not obj.data.shape_keys:
        return
    
    #nway_expr = 'max(min(({0}-{1})/({2}-{1}),({4}-{0})/({4}-{3})),0)'
    #two_way_0_expr = 'clamp({}*-1)'
    #two_way_1_expr = 'clamp({})'
    #upper_eye_expr = '(1-abs(min({}, 0)))*{}*{}'
    #lower_eye_expr = '(1-abs(max({}, 0)))*(1-{})*{}'

    all_exprs = mdl.rebuild_flex_rules()
    bpy.types.Scene.t = all_exprs
    data: bpy.types.Mesh = obj.data
    shape_keys = data.shape_keys
    kb = shape_keys.key_blocks

    def make_custom_property(name, min, max, default=0.0):
        obj.data[name] = default
        prop = obj.data.id_properties_ui(name)
        prop.update(min=min, max=max)

    #tally = iter(range(999))
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
    def dom_tally():
        for i in range(999):
            yield 'D'+str(i)

    shape_keys = obj.data.shape_keys

    for name, (expr, inputs) in all_exprs.items():
        expr = expr.as_simple()
        vtally = var_tally()
        #dtally = dom_tally()
        if kb.get(name):
            kb[name].driver_remove('value')
            driv = kb[name].driver_add('value')
        else:
            shape_keys[name] = 0.0
            driv = shape_keys.driver_add(f'["{name}"]')

        [driv.modifiers.remove(mod) for mod in driv.modifiers]
        
        driv = driv.driver
        for input, type in set(inputs):
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
        assert len(expr) < 256
        driv.expression = expr.replace('--', '+') + '*FS'

    return
    int_to_flex = {n: flexmap[mdl.flex_controllers[n].name] for n, flex in enumerate(mdl.flex_controllers)}


    for rule in mdl.flex_rules:
        vtally = var_tally()
        dtally = dom_tally()
        expr = ''
        stack = []

        op_count = len(rule.flex_ops)

        for n, op in enumerate(rule.flex_ops):
            last = (n + 1) == op_count
            flex_op = op.op
            if flex_op == FlexOpType.CONST:
                stack.append(Value(op.value))
            elif flex_op == FlexOpType.FETCH1:
                inputs.append((self.flex_controllers[op.value].name, 'fetch1'))
                fc_ui = flex_controllers[self.flex_controllers[op.value].name]
                stack.append(FetchController(fc_ui.name, fc_ui.stereo))
            elif flex_op == FlexOpType.FETCH2:
                inputs.append((self.flex_names[op.value], 'fetch2'))
                stack.append(FetchFlex(self.flex_names[op.value]))
            elif flex_op == FlexOpType.ADD:
                stack.append(Add(stack.pop(-1), stack.pop(-1)))
            elif flex_op == FlexOpType.SUB:
                stack.append(Sub(stack.pop(-1), stack.pop(-1)))
            elif flex_op == FlexOpType.MUL:
                stack.append(Mul(stack.pop(-1), stack.pop(-1)))
            elif flex_op == FlexOpType.DIV:
                stack.append(Div(stack.pop(-1), stack.pop(-1)))
            elif flex_op == FlexOpType.NEG:
                stack.append(Neg(stack.pop(-1)))
            elif flex_op == FlexOpType.MAX:
                stack.append(Max(stack.pop(-1), stack.pop(-1)))
            elif flex_op == FlexOpType.MIN:
                stack.append(Min(stack.pop(-1), stack.pop(-1)))
            elif flex_op == FlexOpType.COMBO:
                stack.append(Combo(*[stack.pop(-1) for _ in range(op.value)]))
            elif flex_op == FlexOpType.DOMINATE:
                stack.append(Dominator(*[stack.pop(-1) for _ in range(op.value + 1)]))
            elif flex_op == FlexOpType.TWO_WAY_0:
                inputs.append((self.flex_controllers[op.value].name, '2WAY0'))
                fc_ui = flex_controllers[self.flex_controllers[op.value].name]
                stack.append(RClamp(FetchController(fc_ui.name, fc_ui.stereo),
                                    -1, 0, 1, 0))
            elif flex_op == FlexOpType.TWO_WAY_1:
                inputs.append((self.flex_controllers[op.value].name, '2WAY1'))
                fc_ui = flex_controllers[self.flex_controllers[op.value].name]
                stack.append(Clamp(FetchController(fc_ui.name, fc_ui.stereo), 0, 1), )
            elif flex_op == FlexOpType.NWAY:

                inputs.append((self.flex_controllers[op.value].name, 'NWAY'))
                fc_ui = flex_controllers[self.flex_controllers[op.value].name]
                flex_cnt = FetchController(fc_ui.name, fc_ui.stereo)

                flex_cnt_value = int(stack.pop(-1).value)
                inputs.append((self.flex_controllers[flex_cnt_value].name, 'NWAY'))
                fc_ui = flex_controllers[self.flex_controllers[flex_cnt_value].name]
                multi_cnt = FetchController(fc_ui.nway_controller, fc_ui.stereo)

                # Reversed the order, revert back if it wont help
                f_w = stack.pop(-1)
                f_z = stack.pop(-1)
                f_y = stack.pop(-1)
                f_x = stack.pop(-1)
                final_expr = NWay(multi_cnt, flex_cnt, f_x, f_y, f_z, f_w)
                stack.append(final_expr)
            elif flex_op == FlexOpType.DME_UPPER_EYELID:
                close_lid_v_controller = self.flex_controllers[op.value]
                inputs.append((close_lid_v_controller.name, 'DUE'))
                close_lid_v = RClamp(FetchController(close_lid_v_controller.name),
                                        close_lid_v_controller.min, close_lid_v_controller.max,
                                        0, 1)

                flex_cnt_value = int(stack.pop(-1).value)
                close_lid_controller = self.flex_controllers[flex_cnt_value]
                inputs.append((close_lid_controller.name, 'DUE'))
                close_lid = RClamp(FetchController(close_lid_controller.name),
                                    close_lid_controller.min, close_lid_controller.max,
                                    0, 1)

                blink_index = int(stack.pop(-1).value)
                # blink = Value(0.0)
                # if blink_index >= 0:
                #     blink_controller = self.flex_controllers[blink_index]
                #     inputs.append((blink_controller.name, 'DUE'))
                #     blink_fetch = FetchController(blink_controller.name)
                #     blink = CustomFunction('rclamped', blink_fetch,
                #                            blink_controller.min, blink_controller.max,
                #                            0, 1)

                eye_up_down_index = int(stack.pop(-1).value)
                eye_up_down = Value(0.0)
                if eye_up_down_index >= 0:
                    eye_up_down_controller = self.flex_controllers[eye_up_down_index]
                    inputs.append((eye_up_down_controller.name, 'DUE'))
                    eye_up_down_fetch = FetchController(eye_up_down_controller.name)
                    eye_up_down = RClamp(eye_up_down_fetch,
                                            eye_up_down_controller.min, eye_up_down_controller.max,
                                            -1, 1)

                stack.append(CustomFunction('upper_eyelid_case', eye_up_down, close_lid_v, close_lid))
            elif flex_op == FlexOpType.DME_LOWER_EYELID:
                close_lid_v_controller = self.flex_controllers[op.value]
                inputs.append((close_lid_v_controller.name, 'DUE'))
                close_lid_v = RClamp(FetchController(close_lid_v_controller.name),
                                        close_lid_v_controller.min, close_lid_v_controller.max,
                                        0, 1)

                flex_cnt_value = int(stack.pop(-1).value)
                close_lid_controller = self.flex_controllers[flex_cnt_value]
                inputs.append((close_lid_controller.name, 'DUE'))
                close_lid = RClamp(FetchController(close_lid_controller.name),
                                    close_lid_controller.min, close_lid_controller.max,
                                    0, 1)

                blink_index = int(stack.pop(-1).value)
                # blink = Value(0.0)
                # if blink_index >= 0:
                #     blink_controller = self.flex_controllers[blink_index]
                #     inputs.append((blink_controller.name, 'DUE'))
                #     blink_fetch = FetchController(blink_controller.name)
                #     blink = CustomFunction('rclamped', blink_fetch,
                #                            blink_controller.min, blink_controller.max,
                #                            0, 1)

                eye_up_down_index = int(stack.pop(-1).value)
                eye_up_down = Value(0.0)
                if eye_up_down_index >= 0:
                    eye_up_down_controller = self.flex_controllers[eye_up_down_index]
                    inputs.append((eye_up_down_controller.name, 'DUE'))
                    eye_up_down_fetch = FetchController(eye_up_down_controller.name)
                    eye_up_down = RClamp(eye_up_down_fetch,
                                            eye_up_down_controller.min, eye_up_down_controller.max,
                                            -1, 1)

                stack.append(CustomFunction('lower_eyelid_case', eye_up_down, close_lid_v, close_lid))
            elif flex_op == FlexOpType.OPEN:
                continue
            else:
                print("Unknown OP", op)

    return

    def _parse_simple_flex(missing_flex_name: str):
        flexes = missing_flex_name.split('_')
        if not all(flex in data.flex_controllers for flex in flexes):
            return None
        return Combo([FetchController(flex) for flex in flexes]), [(flex, 'fetch2') for flex in flexes]

    st = '\n    '

    for flex_controller_ui in mdl.flex_ui_controllers:
        cont: SourceIO_PG_FlexController = data.flex_controllers.add()

        if flex_controller_ui.nway_controller:
            nway_cont: SourceIO_PG_FlexController = data.flex_controllers.add()
            nway_cont.stereo = False
            multi_controller = next(filter(lambda a: a.name == flex_controller_ui.nway_controller, mdl.flex_controllers)
                                    )
            nway_cont.name = flex_controller_ui.nway_controller
            nway_cont.set_from_controller(multi_controller)

        if flex_controller_ui.stereo:
            left_controller = next(
                filter(lambda a: a.name == flex_controller_ui.left_controller, mdl.flex_controllers)
            )
            right_controller = next(
                filter(lambda a: a.name == flex_controller_ui.right_controller, mdl.flex_controllers)
            )
            cont.stereo = True
            cont.name = flex_controller_ui.name
            assert left_controller.max == right_controller.max
            assert left_controller.min == right_controller.min
            cont.set_from_controller(left_controller)
        else:
            controller = next(filter(lambda a: a.name == flex_controller_ui.controller, mdl.flex_controllers))
            cont.stereo = False
            cont.name = flex_controller_ui.name
            cont.set_from_controller(controller)
    blender_py_file = """
import bpy

def rclamped(val, a, b, c, d):
    if ( a == b ):
        return d if val >= b else c;
    return c + (d - c) * min(max((val - a) / (b - a), 0.0), 1.0)
    
def clamp(val, a, b):
    return min(max(val, a), b)

def nway(multi_value, flex_value, x, y, z, w):
    if multi_value <= x or multi_value >= w:  # outside of boundaries
        multi_value = 0.0
    elif multi_value <= y:
        multi_value = rclamped(multi_value, x, y, 0.0, 1.0)
    elif multi_value >= z:
        multi_value = rclamped(multi_value, z, w, 1.0, 0.0)
    else:
        multi_value = 1.0
    return multi_value * flex_value

def nway(multi_value, flex_value, x, y, zw):
    z=zw[0]
    w=list(zw[1])[0]
    if multi_value <= x or multi_value >= w:  # outside of boundaries
        multi_value = 0.0
    elif multi_value <= y:
        multi_value = rclamped(multi_value, x, y, 0.0, 1.0)
    elif multi_value >= z:
        multi_value = rclamped(multi_value, z, w, 1.0, 0.0)
    else:
        multi_value = 1.0
    return multi_value * flex_value


def combo(*values):
    val = values[0]
    for v in values[1:]:
        val*=v
    return val
    
def dom(dm, *values):
    val = 1
    for v in values:
        val *= v
    return val * (1 - dm)

def lower_eyelid_case(eyes_up_down,close_lid_v,close_lid):
    if eyes_up_down > 0.0:
        return (1.0 - eyes_up_down) * (1.0 - close_lid_v) * close_lid
    else:
        return  (1.0 - close_lid_v) * close_lid

def upper_eyelid_case(eyes_up_down,close_lid_v,close_lid):
    if eyes_up_down < 0.0:
        return (1.0 - abs(eyes_up_down)) * close_lid_v * close_lid
    else:
        return  close_lid_v * close_lid


bpy.app.driver_namespace["combo"] = combo
bpy.app.driver_namespace["dom"] = dom
bpy.app.driver_namespace["nway"] = nway
bpy.app.driver_namespace["rclamped"] = rclamped

    """
    
    run_function="""
def create_drivers():
"""
        
    def normalize_name(name):
        return name.replace("-", "_").replace(" ", "_")

    for flex_name, (expr, inputs) in all_exprs.items():
        normalized_flex_name = normalize_name(flex_name)
        driver_name = f"{normalized_flex_name}_driver"
        if driver_name in globals():
            continue

        input_definitions = []
        for inp in inputs:
            input_name = inp[0]
            normalized_input_name = normalize_name(inp[0])
            if inp[1] in ('fetch1', '2WAY1', '2WAY0', 'NWAY', 'DUE'):
                if 'left_' in input_name:
                    if 'Lid' in input_name:
                        input_definitions.append(
                        f'{normalized_input_name} = obj_data.flex_controllers["{input_name.replace("left_", "")}"].value_left')
                    else:
                        input_definitions.append(
                        f'{normalized_input_name.replace("left_", "")} = obj_data.flex_controllers["{input_name.replace("left_", "")}"].value_left')                        
                elif 'right_' in input_name:
                    if 'Lid' in input_name:
                        input_definitions.append(
                        f'{normalized_input_name} = obj_data.flex_controllers["{input_name.replace("right_", "")}"].value_right')
                    else:
                        input_definitions.append(
                        f'{normalized_input_name.replace("right_", "")} = obj_data.flex_controllers["{input_name.replace("right_", "")}"].value_right')
                else:
                    input_definitions.append(
                        f'{normalized_input_name} = obj_data.flex_controllers["{inp[0]}"].value')
            elif inp[1] == 'fetch2':
                input_definitions.append(
                    f'{normalized_input_name} = obj_data.shape_keys.key_blocks["{input_name}"].value')
            else:
                raise NotImplementedError(f'"{inp[1]}" is not supported')
        print(f"{flex_name} = {expr}")
        template_function = f"""
def {driver_name}(obj_data):
    {st.join(input_definitions)}
    return {expr}
"""
        run_function += f"""
    bpy.app.driver_namespace["{driver_name}"] = {driver_name}
"""
        blender_py_file += template_function

    for shape_key in shape_key_block.key_blocks:

        flex_name = shape_key.name
        normalized_flex_name = normalize_name(flex_name)

        if flex_name == 'base':
            continue        
        if flex_name not in all_exprs:
            warnings.warn(f'Rule for {flex_name} not found! Generating basic rule.')
            expr, inputs = _parse_simple_flex(flex_name) or (None, None)
            if not expr or not inputs:
                warnings.warn(f'Failed to generate basic rule for {flex_name}!')
                cont: SourceIO_PG_FlexController = data.flex_controllers.add()
                cont.name = flex_name
                cont.mode = 1
                cont.value_min = 0
                cont.value_max = 1
                template_function = f"""
def {normalized_flex_name}_driver(obj_data):
    return obj_data.flex_controllers["{flex_name}"].value
"""
                blender_py_file += template_function
            else:
                template_function = f"""
def {normalized_flex_name}_driver(obj_data):
    {st.join([i[0] for i in inputs])}
    return {expr}
"""
                blender_py_file += template_function
            run_function+=f"""
    bpy.app.driver_namespace["{normalized_flex_name}_driver"] = {normalized_flex_name}_driver
"""
    blender_py_file+=run_function
    blender_py_file+="""
create_drivers()
"""
    
    driver_file = bpy.data.texts.new(f'{mdl.header.name}.py')
    driver_file.write(blender_py_file)
    driver_file.use_module = True
    driver_file.as_module().create_drivers()
    
    for shape_key in shape_key_block.key_blocks:        
        flex_name = shape_key.name
        normalized_flex_name = normalize_name(flex_name)
        
        if flex_name == 'base':
            continue  
        
        shape_key.driver_remove("value")
        fcurve = shape_key.driver_add("value")
        [fcurve.modifiers.remove(mod) for mod in fcurve.modifiers]

        driver = fcurve.driver
        driver.type = 'SCRIPTED'
        driver.expression = f"{normalized_flex_name}_driver(obj_data)"
        var = driver.variables.new()
        var.name = 'obj_data'
        var.targets[0].id_type = 'OBJECT'
        var.targets[0].id = obj
        var.targets[0].data_path = f"data"


def create_attachments(mdl: MdlV49, armature: bpy.types.Object, scale):
    attachments = []
    for attachment in mdl.attachments:
        empty = bpy.data.objects.new(attachment.name, None)
        pos = Vector(attachment.pos) * scale
        rot = Euler(attachment.rot)

        empty.matrix_basis.identity()
        empty.scale *= scale
        empty.location = pos
        empty.rotation_euler = rot

        if armature.type == 'ARMATURE':
            modifier = empty.constraints.new(type="CHILD_OF")
            modifier.target = armature
            modifier.subtarget = mdl.bones[attachment.parent_bone].name
            modifier.inverse_matrix.identity()

        attachments.append(empty)

    return attachments


def import_animations(cm: ContentProvider, mdl: MdlV49, armature: bpy.types.Object,
                      scale: float):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.scene.collection.objects.link(armature)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if not armature.animation_data:
        armature.animation_data_create()

    for n, anim in enumerate(mdl.anim_descs):
        animation_data = mdl.animations[n]
        action = bpy.data.actions.new(anim.name)
        action.use_fake_user = True
        factory = ActionCurveFactory(action, armature)
        curve_per_bone = {}

        for bone in mdl.bones:
            bone_name = bone.name
            bl_bone = armature.pose.bones.get(bone_name)
            bl_bone.rotation_mode = 'QUATERNION'
            bone_string = f'pose.bones["{bone_name}"].'
            group = factory.new_group(bone_name)
            pos_curves = []
            rot_curves = []
            for i in range(3):
                pos_curve = factory.new_fcurve(data_path=bone_string + "location", index=i, group=group)
                pos_curve.keyframe_points.add(count=anim.frame_count)
                pos_curve.auto_smoothing = "CONT_ACCEL"
                pos_curves.append(pos_curve)
            for i in range(4):
                rot_curve = factory.new_fcurve(data_path=bone_string + "rotation_quaternion", index=i, group=group)
                rot_curve.keyframe_points.add(count=anim.frame_count)
                rot_curve.auto_smoothing = "CONT_ACCEL"
                rot_curves.append(rot_curve)
            curve_per_bone[bone_name] = pos_curves, rot_curves
        for bone_id, bone in enumerate(mdl.bones):
            pos_curves, rot_curves = curve_per_bone[bone.name]
            for curve in itertools.chain(pos_curves, rot_curves):
                curve.keyframe_points.add(count=anim.frame_count)
            bl_bone = armature.pose.bones.get(bone.name)
            for frame_id in range(anim.frame_count):
                anim_data = animation_data[frame_id, bone_id]
                bl_bone.matrix_basis.identity()
                pos = Vector(anim_data["pos"]) * scale
                x, y, z, w = anim_data["rot"]
                rot = Quaternion((w, x, y, z))
                mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()

                if bl_bone.parent:
                    mat = bl_bone.parent.matrix @ mat if bl_bone.parent else mat
                    bl_bone.matrix = mat
                    pos, rot = bl_bone.location, bl_bone.rotation_quaternion
                    for i in range(3):
                        pos_curves[i].keyframe_points[frame_id].co = (frame_id, (pos[i]))

                    for i in range(4):
                        rot_curves[i].keyframe_points[frame_id].co = (frame_id, (rot[i]))
                else:
                    mat = bl_bone.matrix.inverted() @ mat
                    pos, rot, scl = mat.decompose()
                    for i in range(3):
                        pos_curves[i].keyframe_points[frame_id].co = (frame_id, (pos[i]))

                    for i in range(4):
                        rot_curves[i].keyframe_points[frame_id].co = (frame_id, (rot[i]))
                bl_bone.matrix = Matrix.Identity(4)
        for pos_curves, rot_curves in curve_per_bone.values():
            for curve in rot_curves + pos_curves:
                curve.update()
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.collection.objects.unlink(armature)
