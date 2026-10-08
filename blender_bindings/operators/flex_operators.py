import bpy
from bpy.props import (BoolProperty, FloatProperty,
                       IntProperty, StringProperty,
                       EnumProperty)
from bpy.types import Operator
from bpy.app.handlers import persistent

import numpy as np

_dragging = False
_updating_values = False

#@persistent
#def ensure_var_reset(a=None, b=None):
#    global _dragging
#    global _updating_values
#
#    _dragging = False
#    _updating_values = False

def get_frame(context):
    return context.scene.frame_float if context.scene.show_subframe else context.scene.frame_current

if bpy.app.version >= (4, 4, 0):
    def has_key(context, obj, slider) -> bool:
        from bpy_extras import anim_utils
        data = obj.data
        if not (anim_data := getattr(data, 'animation_data', None)):
            return False
        action = getattr(anim_data, 'action', None)
        action_slot = getattr(anim_data, 'action_slot', None)
        channelbag = anim_utils.action_get_channelbag_for_slot(action, action_slot)
        
        if not channelbag:
            return False

        curv = channelbag.fcurves.find(f'["{slider}"]')
        if curv == None: return False

        points: bpy.types.FCurveKeyframePoints = curv.keyframe_points
        p_array = np.zeros(len(points)*2, dtype=np.float32)
        points.foreach_get('co', p_array)
        p_array = p_array.reshape((-1, 2))[:, 0]

        return get_frame(context) in p_array
else:
    def has_key(context, obj, slider) -> bool:
        data = obj.data
        if not (anim_data := getattr(data, 'animation_data', None)):
            return False
        if not (action := getattr(anim_data, 'action', None)):
            return False
        
        curv = action.fcurves.find(f'["{slider}"]')
        if curv == None: return False

        points: bpy.types.FCurveKeyframePoints = curv.keyframe_points
        p_array = np.zeros(len(points)*2, dtype=np.float32)
        points.foreach_get('co', p_array)
        p_array = p_array.reshape((-1, 2))[:, 0]

        return get_frame(context) in p_array

def map_range(x,a,b,c,d, clamp=None):
    y=(x-a)/(b-a)*(d-c)+c
    
    if clamp:
        return min(max(y, c), d)
    else:
        return y


def slideupdate(self, context):
    global _dragging
    global _updating_values

    if _updating_values: return None # allow slider values to be reset back to 0 without undergoing any updates
    
    if not _dragging:
        _dragging = True
        bpy.ops.sourceio.sliderhandler('INVOKE_DEFAULT')

    return None

class SOURCEIO_OT_adjust_balance(Operator):
    bl_idname = 'sourceio.adjust_balance'
    bl_label = 'Adjust'
    bl_description = 'Adjust the Left/Right by 0.1'
    amount: FloatProperty()

    def execute(self, context):
        from math import ceil, floor
        props = context.scene.sourceio_props
        val = round(props.lr_balance * 10, 1)
        if val % 1 == 0:
            balance = val/10 + self.amount
            props.lr_balance = balance
            return {'FINISHED'}
        if self.amount > 0.0:
            props.lr_balance = ceil(val)/10
        else:
            props.lr_balance = floor(val)/10
        return {'FINISHED'}

# noinspection PyPep8Naming
class SourceIO_PG_FlexController(bpy.types.PropertyGroup):
    display_name: StringProperty(name='Display Name', default='')
    name: StringProperty()
    value: FloatProperty(name='Additive Value', default=0.0, update=slideupdate, min=-1, max=1, options=set(), description='Sliders are additive, and will be reset!')
    split: BoolProperty(name='Stereo', default=False, options=set())

    minimum: FloatProperty(options=set())
    maximum: FloatProperty(options=set())

    originalval: FloatProperty(options=set())
    originalvalR: FloatProperty(options=set())
    originalvalL: FloatProperty(options=set())

    realvalue: BoolProperty(default=True, options=set())

    R: StringProperty()
    L: StringProperty()

def handle_slider(context: bpy.types.Context, obj: bpy.types.Object, slider: SourceIO_PG_FlexController):
    props = context.scene.sourceio_props
    data = obj.data
    magnitude = max(abs(slider.maximum), abs(slider.minimum))

    if not slider.split:
        data[slider.name] = min(max(slider.originalval + slider.value*magnitude, slider.minimum), slider.maximum)
    else:
        Rmult = map_range(props.lr_balance, -1, 0.0, 0.0, 1.0, True)
        Lmult = map_range(props.lr_balance, 1.0, 0.0, 0.0, 1.0, True)
        data[slider.R] = min(max(slider.originalvalR + (slider.value * Rmult * magnitude), slider.minimum), slider.maximum)
        data[slider.L] = min(max(slider.originalvalL + (slider.value * Lmult * magnitude), slider.minimum), slider.maximum)

class SOURCEIO_OT_SLIDERHANDLER(Operator):
    bl_idname = 'sourceio.sliderhandler'
    bl_label = ''
    slider = StringProperty()
    stop: BoolProperty(default = False)

    active_sliders = set()

    '''This operator decides what will happen to the sliders once
    the user stops manipulating them. It only deals with keyframing and resetting.
    If the sliders were changed and auto keyframing is enabled, keyframe.
    If one slider was cancelled, then all were cancelled, and reset
    the face to its original position.'''

    def resetsliders(self, sliders, context):
        face = context.object
        for i in sliders:
            if i.split:
                face.data[i.R] = i.originalvalR
                face.data[i.L] = i.originalvalL
            else:
                face.data[i.name] = i.originalval
        face.data.update()

    def modal(self, context, event):
        global _dragging
        global _updating_values
        obj = context.object
        face = obj

        if self.stop:
            _dragging = False
            return {'FINISHED'}

        if not self.active_sliders:
            for slider in obj.flex_controllers:
                if slider.split:
                    slider.originalvalR = face.data[slider.R]
                    slider.originalvalL = face.data[slider.L]
                else:
                    slider.originalval = face.data[slider.name]

                if slider.value != 0.0:
                    self.active_sliders.add(slider)
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            for slider in self.active_sliders:
                handle_slider(context, obj, slider)

            obj.data.update()
            return {'PASS_THROUGH'}

        if event.value == 'RELEASE':
            self.stop = True
            _dragging = False # allow sliders to initialize original values again
            _updating_values = True

            for slider in self.active_sliders:
                handle_slider(context, obj, slider)

            for slider in self.active_sliders:
                if slider.value == 0.0:
                    self.resetsliders(self.active_sliders, context)
                    break

            else: # if the previous for loop did not break, then nothing was cancelled/reset. keyframe if auto-keyframing is enabled.
                if context.scene.tool_settings.use_keyframe_insert_auto:
                    for slider in self.active_sliders:
                        if slider.split:
                            face.data.keyframe_insert(data_path=f'["{slider.R}"]', frame=get_frame(context))
                            face.data.keyframe_insert(data_path=f'["{slider.L}"]', frame=get_frame(context))
                        else:
                            face.data.keyframe_insert(data_path=f'["{slider.name}"]', frame=get_frame(context))

            for slider in self.active_sliders:
                slider.value = 0.0

            obj.data.update()
            self.active_sliders.clear()

            _updating_values = False
            return {'PASS_THROUGH'}
        
        return {'PASS_THROUGH'}
    
    def invoke(self, context, event):
        self.stop = False
        self.active_sliders.clear()

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class SOURCEIO_OT_KEY_FLEX_CONTROLLER(Operator):
    bl_idname = 'sourceio.key_flex_controller'
    bl_label = 'Keyframe Flex Controller'
    bl_description = 'Keyframe this flex controller'
    delete: BoolProperty(default=False)
    flex_controller: StringProperty()

    bl_options = {'UNDO'}

    def execute(self, context):
        obj = context.object
        slider = obj.flex_controllers[self.flex_controller]
        if slider.split:
            is_keyed_on_frame = has_key(context, obj, slider.R) and has_key(context, obj, slider.L)
        else:
            is_keyed_on_frame = has_key(context, obj, slider.name)

        data = obj.data
        if slider.split:
            if is_keyed_on_frame:
                data.keyframe_delete(data_path=f'["{slider.R}"]', frame=get_frame(context))
                data.keyframe_delete(data_path=f'["{slider.L}"]', frame=get_frame(context))
            else:
                data.keyframe_insert(data_path=f'["{slider.R}"]', frame=get_frame(context))
                data.keyframe_insert(data_path=f'["{slider.L}"]', frame=get_frame(context))
        else:
            if is_keyed_on_frame:
                data.keyframe_delete(data_path=f'["{slider.name}"]', frame=get_frame(context))
            else:
                data.keyframe_insert(data_path=f'["{slider.name}"]', frame=get_frame(context))

        return {'FINISHED'}

class SOURCEIO_OT_resetface(Operator):
    bl_idname = 'sourceio.resetface'
    bl_label = 'Reset Face'
    bl_description = 'Reset all sliders to 0.0'
    bl_options = {'UNDO'}

    def execute(self, context):
        data = context.object.data
        
        if data.get('flexmap'):
            flexes = list(data['flexmap'].values())
        elif data.get('flexcontrollers'):
            flexes = list(data['flexcontrollers'].values())

        for flex in flexes:
            if flex == 'aaa_fs':
                if data[flex] != 1.0:
                    data[flex] = 1.0
                    if context.scene.tool_settings.use_keyframe_insert_auto:
                        data.keyframe_insert(data_path=f'["{flex}"]', frame=get_frame(context))
                continue
            if data[flex] != 0.0:
                data[flex] = 0.0
                if context.scene.tool_settings.use_keyframe_insert_auto:
                    data.keyframe_insert(data_path=f'["{flex}"]', frame=get_frame(context))
        
        data.update()
        return {'FINISHED'}

class SOURCEIO_OT_KEY_ALL_FLEXCONTROLLERS(Operator):
    bl_idname = 'sourceio.key_all_flexcontrollers'
    bl_label = 'Key Every Flex Controller'
    bl_description = 'Add a keyframe to every flex controller on this frame'
    bl_options = {'UNDO'}

    def execute(self, context):
        obj = context.object
        data = obj.data
        for i in obj.flex_controllers:
            if i.split:
                data.keyframe_insert(data_path=f'["{i.L}"]', frame=get_frame(context))
                data.keyframe_insert(data_path=f'["{i.R}"]', frame=get_frame(context))
                continue
            data.keyframe_insert(data_path=f'["{i.name}"]', frame=get_frame(context))
        
        return {'FINISHED'}

class SOURCEIO_PT_FlexControlPanel(bpy.types.Panel):
    #bl_label = ''
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    #bl_parent_id = "TRIFECTA_PT_PANEL"
    #bl_category = 'TF2-Trifecta'
    bl_options = {'HEADER_LAYOUT_EXPAND'}

    bl_label = ''
    bl_idname = 'SOURCEIO_PT_FlexControlPanel'
    bl_parent_id = "SOURCEIO_PT_Utils"

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)  # type:bpy.types.Object
        if not obj:
            return False

        return obj and obj.type == 'MESH' and len(obj.flex_controllers)
    
    def draw_header(self, context):
        l = self.layout
        row = l.row()
        r = row.row()
        r.alignment = 'EXPAND'
        r.separator()
        r.label(icon='RESTRICT_SELECT_OFF', text='Flex Controllers')

    def draw(self, context):
        props = context.scene.sourceio_props
        layout = self.layout
        obj = context.object

        if obj.data.library:
            layout.label('Object data is linked.')
            layout.label(text='Flexes are frozen until localized!')
            return {'FINISHED'}

        box = layout.box()
        row = box.row(align=True)

        col = row.column()
        col.template_list('SOURCEIO_UL_FlexControllerList', 'Flex Controllers', obj, 'flex_controllers', props, 'flex_controller_index')
        col = row.column()

        col.scale_y = 1.5
        col.scale_x = 1.2
        col.row().prop(context.scene.tool_settings, 'use_keyframe_insert_auto', text='')
        col.separator()
        col.row().operator('sourceio.key_all_flexcontrollers', icon='DECORATE_KEYFRAME', text='')
        col.separator()
        col.row().operator('sourceio.resetface', icon='LOOP_BACK', text='')

        col = box.column(align=False)
        row = col.row(align=True)
        op = row.operator('sourceio.adjust_balance', text='', icon='TRIA_LEFT')
        op.amount = -0.1
        row.prop(props, 'lr_balance', slider=True, text='')
        op = row.operator('sourceio.adjust_balance', text='', icon='TRIA_RIGHT')
        op.amount = 0.1
        
        r = col.row()
        r.alignment = 'CENTER'
        r.label(text='L <--> R Balance')
        r.enabled = False


class SOURCEIO_UL_FlexControllerList(bpy.types.UIList):

    def draw_item(self, context,
            layout, data,
            item, icon,
            active_data, active_propname,
            index):

        is_keyed_on_frame = False

        #if context.scene.sourceio_props.noKeyStatus:
        #    is_keyed_on_frame = False
        #else:
        obj = context.object
        if item.split:
            is_keyed_on_frame = has_key(context, obj, item.R) or has_key(context, obj, item.L)
        else:
            is_keyed_on_frame = has_key(context, obj, item.name)

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if not item.realvalue:
                row.alert = is_keyed_on_frame
                row.prop(item, 'value', slider=True, text=item.display_name)
                row.alert = False
            else:
                if item.split:
                    row.prop(context.object.data, f'["{item.R}"]', text='R')
                    row.prop(context.object.data, f'["{item.L}"]', text='L')
                else:
                    row.prop(context.object.data, f'["{item.name}"]', text=item.display_name)

            op = row.operator('sourceio.key_flex_controller', icon='DECORATE_KEYFRAME' if is_keyed_on_frame else 'DECORATE_ANIMATE', text='', depress=is_keyed_on_frame, emboss=False)
            op.flex_controller = item.name
            row.prop(item, 'realvalue', icon='RESTRICT_VIEW_OFF' if item.realvalue else 'RESTRICT_VIEW_ON', text='', emboss = False)

        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text='')


classes = (
    SourceIO_PG_FlexController,
    SOURCEIO_UL_FlexControllerList,
    SOURCEIO_PT_FlexControlPanel,
    SOURCEIO_OT_adjust_balance,
    SOURCEIO_OT_SLIDERHANDLER,
    SOURCEIO_OT_KEY_FLEX_CONTROLLER,
    SOURCEIO_OT_resetface,
    SOURCEIO_OT_KEY_ALL_FLEXCONTROLLERS
)
