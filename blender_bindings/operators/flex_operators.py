import bpy
from bpy.props import (BoolProperty, CollectionProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)

from ...library.source1.mdl.structs.flex import FlexController
from .shared_operators import UITools


def update_max_min(self: 'SourceIO_PG_FlexController', _):
    if self.stereo:
        if self['valuen_left'] >= self.value_max:
            self['valuen_left'] = self.value_max
        if self['valuen_left'] <= self.value_min:
            self['valuen_left'] = self.value_min

        if self['valuen_right'] >= self.value_max:
            self['valuen_right'] = self.value_max
        if self['valuen_right'] <= self.value_min:
            self['valuen_right'] = self.value_min
    else:
        if self['valuen'] >= self.value_max:
            self['valuen'] = self.value_max
        if self['valuen'] <= self.value_min:
            self['valuen'] = self.value_min


# noinspection PyPep8Naming
class SourceIO_PG_FlexController(bpy.types.PropertyGroup):
    name: StringProperty()
    stereo: BoolProperty(name="stereo")

    value_max: FloatProperty(name='max')
    value_min: FloatProperty(name='min')

    mode: IntProperty(name='mode')

    valuen: FloatProperty(name="value", min=-100.0, max=100.0, update=update_max_min)
    valuezo: FloatProperty(name="value", min=0.0, max=1.0)
    valuenoo: FloatProperty(name="value", min=-1.0, max=1.0)
    valuenoz: FloatProperty(name="value", min=-1.0, max=0.0)

    valuen_left: FloatProperty(name="value_left", min=-100.0, max=100.0, update=update_max_min)
    valuezo_left: FloatProperty(name="value_left", min=0.0, max=1.0)
    valuenoo_left: FloatProperty(name="value_left", min=-1.0, max=1.0)
    valuenoz_left: FloatProperty(name="value_left", min=-1.0, max=0.0)

    valuen_right: FloatProperty(name="value_right", min=-100.0, max=100.0, update=update_max_min)
    valuezo_right: FloatProperty(name="value_right", min=0.0, max=1.0)
    valuenoo_right: FloatProperty(name="value_right", min=-1.0, max=1.0)
    valuenoz_right: FloatProperty(name="value_right", min=-1.0, max=0.0)

    def set_from_controller(self, controller: FlexController):
        self.value_min = controller.min
        self.value_max = controller.max
        if controller.min == 0.0 and controller.max == 1.0:
            self.mode = 1
        elif controller.min == -1.0 and controller.max == 1.0:
            self.mode = 2
        elif controller.min == -1.0 and controller.max == 0:
            self.mode = 3
        else:
            self.mode = 0

    def get_slot_name(self):
        if self.mode == 0:
            return 'valuen'
        elif self.mode == 1:
            return 'valuezo'
        elif self.mode == 2:
            return 'valuenoo'
        elif self.mode == 3:
            return 'valuenoz'

    @property
    def value(self):
        if self.mode == 0:
            return self.valuen
        elif self.mode == 1:
            return self.valuezo
        elif self.mode == 2:
            return self.valuenoo
        elif self.mode == 3:
            return self.valuenoz

    @value.setter
    def value(self, new_value):
        if self.mode == 0:
            self.valuen = new_value
        elif self.mode == 1:
            self.valuezo = new_value
        elif self.mode == 2:
            self.valuenoo = new_value
        elif self.mode == 3:
            self.valuenoz = new_value

    @property
    def value_right(self):
        if self.mode == 0:
            return self.valuen_right
        elif self.mode == 1:
            return self.valuezo_right
        elif self.mode == 2:
            return self.valuenoo_right
        elif self.mode == 3:
            return self.valuenoz_right

    @value_right.setter
    def value_right(self, new_value):
        if self.mode == 0:
            self.valuen_right = new_value
        elif self.mode == 1:
            self.valuezo_right = new_value
        elif self.mode == 2:
            self.valuenoo_right = new_value
        elif self.mode == 3:
            self.valuenoz_right = new_value

    @property
    def value_left(self):
        if self.mode == 0:
            return self.valuen_left
        elif self.mode == 1:
            return self.valuezo_left
        elif self.mode == 2:
            return self.valuenoo_left
        elif self.mode == 3:
            return self.valuenoz_left

    @value_left.setter
    def value_left(self, new_value):
        if self.mode == 0:
            self.valuen_left = new_value
        elif self.mode == 1:
            self.valuezo_left = new_value
        elif self.mode == 2:
            self.valuenoo_left = new_value
        elif self.mode == 3:
            self.valuenoz_left = new_value

    def draw_item(self, layout, icon):
        split = layout.split(factor=0.3, align=True)
        split.label(text=self.name, icon_value=icon)
        # layout.prop(controller_entry, "name", text="", emboss=False, icon_value=icon)
        if self.stereo:
            row = split.row()
            if self.mode == 0:
                row.prop(self, 'valuen_left', text='', slider=True)
                row.prop(self, 'valuen_right', text='', slider=True)
            elif self.mode == 1:
                row.prop(self, 'valuezo_left', text='', slider=True)
                row.prop(self, 'valuezo_right', text='', slider=True)
            elif self.mode == 2:
                row.prop(self, 'valuenoo_left', text='', slider=True)
                row.prop(self, 'valuenoo_right', text='', slider=True)
            elif self.mode == 3:
                row.prop(self, 'valuenoz_left', text='', slider=True)
                row.prop(self, 'valuenoz_right', text='', slider=True)
        else:
            if self.mode == 0:
                split.prop(self, 'valuen', text='', slider=True)
            elif self.mode == 1:
                split.prop(self, 'valuezo', text='', slider=True)
            elif self.mode == 2:
                split.prop(self, 'valuenoo', text='', slider=True)
            elif self.mode == 3:
                split.prop(self, 'valuenoz', text='', slider=True)


class SOURCEIO_PT_FlexControlPanel(UITools, bpy.types.Panel):
    bl_label = 'Flex controllers'
    bl_idname = 'SOURCEIO_PT_FlexControlPanel'
    bl_parent_id = "SOURCEIO_PT_Utils"

    @classmethod
    def poll(cls, context):
        obj = context.active_object  # type:bpy.types.Object

        return obj and obj.type == 'MESH' and obj.data.flex_controllers is not None

    def draw(self, context):
        obj = context.active_object  # type:bpy.types.Object
        self.layout.template_list("SOURCEIO_UL_FlexControllerList", "",
                                  obj.data, "flex_controllers",
                                  obj.data, "flex_selected_index")


class SOURCEIO_UL_FlexControllerList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        operator = data
        controller_entry: SourceIO_PG_FlexController = item
        layout.use_property_decorate = True
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            controller_entry.draw_item(layout, icon)
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)


classes = (
    SourceIO_PG_FlexController,
    SOURCEIO_UL_FlexControllerList,
    SOURCEIO_PT_FlexControlPanel,
)
