from typing import List

from ....shared.content_providers.content_manager import ContentManager
from ....utils import datamodel
from ....utils.datamodel import Element
from .base_element import BaseElement
from .film_clip import FilmClip


def find_by_name_n_type(array, name, elem_type):
    for elem in array:
        if elem.name == name and elem.type == elem_type:
            return elem
    return None


class DmeChannel:
    def __init__(self, transform: datamodel.Element):
        self.__trans = transform
        print(self.__trans)


def _quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    nx = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    ny = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    nz = math.degrees(math.atan2(t3, t4))

    return nx, ny, nz


class Entity:
    root = None  # type: Session

    @classmethod
    def set_root(cls, root):
        cls.root = root

    def __init__(self, animset: datamodel.Element, channel_set: datamodel.Element):
        self.animset: datamodel.Element = animset
        self.channel_set: datamodel.Element = channel_set

    @property
    def __transform(self):
        return find_by_name_n_type(self.animset.controls, 'transform', 'DmeTransformControl') or \
               find_by_name_n_type(self.animset.controls, 'rootTransform', 'DmeTransformControl')

    @property
    def position(self):
        if self.__transform:
            return self.__transform.positionChannel.fromElement.get(self.__transform.positionChannel.fromAttribute)
        return 0, 0, 0

    @property
    def orientation_q(self):
        if self.__transform:
            return self.__transform.orientationChannel.fromElement.get(
                self.__transform.orientationChannel.fromAttribute)
        return 0, 0, 0

    @property
    def orientation(self):
        if self.orientation_q[0] != float('nan'):
            return _quaternion_to_euler(*self.orientation_q)

    def __repr__(self):
        return '{}<name:{} at X:{:.2f} Y:{:.2f} Z:{:.2f} rot: X:{:.2f} Y:{:.2f} Z:{:.2f}>'.format(
            self.__class__.__name__, self.animset.name,
            *self.position,
            *self.orientation)


class Camera(Entity):
    pass


class Bone:
    @property
    def name(self):
        return self._element.name

    @property
    def position(self):
        return getattr(self._element.positionChannel.toElement, self._element.positionChannel.toAttribute)

    @property
    def rotation(self):
        return _quaternion_to_euler(
            *getattr(self._element.orientationChannel.toElement, self._element.orientationChannel.toAttribute))

    @property
    def rotation_q(self):
        return getattr(self._element.orientationChannel.toElement, self._element.orientationChannel.toAttribute)

    def __init__(self, bone_element: datamodel.Element):
        self._element = bone_element

    def __repr__(self):
        return 'Bone<name:{} at X:{:.2f} Y:{:.2f} Z:{:.2f} rot: X:{:.2f} Y:{:.2f} Z:{:.2f} W:{:.2f}>'.format(self.name,
                                                                                                             *self.position,
                                                                                                             *self.rotation_q)


class Model(Entity):

    def __init__(self, animset: datamodel.Element, channel_set: datamodel.Element):
        super().__init__(animset, channel_set)
        self.bones = []
        self.flexes = {}
        self.parse()

    @property
    def name(self):
        return self.animset.name

    def parse(self):
        for bone_elem in self.animset.controls:
            if bone_elem.type == 'DmeTransformControl':
                bone = Bone(bone_elem)
                # print(bone)
                self.bones.append(bone)
        self.flexes = {a: b for (a, b) in zip(self.animset.gameModel.flexnames, self.animset.gameModel.flexWeights)}

    @property
    def model_path(self):
        return self.animset.gameModel.modelName

    @property
    def model_file(self):
        return self.root.find_model(self.model_path)

    @property
    def __transform(self):
        return find_by_name_n_type(self.animset.controls, 'rootTransform', 'DmeTransformControl')

    @property
    def root_transform(self):
        return self.__transform


class Light(Entity):
    pass

    @property
    def color(self):
        r = find_by_name_n_type(self.animset.controls, 'color_red', 'DmElement').value
        g = find_by_name_n_type(self.animset.controls, 'color_green', 'DmElement').value
        b = find_by_name_n_type(self.animset.controls, 'color_blue', 'DmElement').value
        return r, g, b

    def __repr__(self):
        return '{}<name:{} at X:{:.2f} Y:{:.2f} Z:{:.2f} rot: X:{:.2f} Y:{:.2f} Z:{:.2f} ' \
               'color: R:{:.2f} G:{:.2f} B:{:.2f}>'.format(self.__class__.__name__, self.animset.name,
                                                           *self.position,
                                                           *self.orientation,
                                                           *self.color)


class Session(BaseElement):

    @staticmethod
    def find_map(map_name):
        return ContentManager().find_file(map_name, additional_dir='maps')

    @staticmethod
    def find_model(model: str):
        return ContentManager().find_file(model)

    def __init__(self, element: Element):
        super().__init__(element)

    @property
    def active_clip(self):
        return FilmClip(self._element['activeClip'])

    @property
    def misc_bin(self):
        return self._element['miscBin']

    @property
    def camera_bin(self):
        return self._element['cameraBin']

    @property
    def clip_bin(self):
        return [FilmClip(clip) for clip in self._element['clipBin']]

    @property
    def settings(self):
        return self._element['settings']

    @staticmethod
    def get_root_transform(controls: List[datamodel.Element]):
        for control in controls:
            if control.name == 'rootTransform' and control.type == 'DmeTransformControl':
                return control

    @staticmethod
    def get_element(controls: List[datamodel.Element], name, type):
        for control in controls:
            if control.name == name and control.type == type:
                return control

    @staticmethod
    def lerp(value, lo, hi):
        f = hi - lo
        return lo + (f * value)

    def load_scene(self):
        for entity in self.entities:
            if type(entity) is Model:
                print('Loading model', entity.name)
                self.load_model(entity)

    # def load_model(self, entity: Model):
    #     from SourceIO.source1.mdl.mdl2model import Source2Blender
    #     import bpy, mathutils
    #     model = entity.model_file
    #     # rot = mathutils.Quaternion(entity.orientation_q)
    #     # rot = rot.to_euler('XZY')
    #     # rot.x, rot.y, rot.z = rot.y, rot.x, rot.z
    #     # rot.x = math.pi / 2 - rot.x
    #     # rot.z = rot.z - math.pi / 2
    #     bl_model = Source2Blender(model, False, self.sfm_path, custom_name=entity.name)
    #     bl_model.load(False)
    #     rot = mathutils.Quaternion(entity.orientation_q).to_euler('XYZ')
    #     rot.y = rot.y - math.pi
    #     # rot.z = rot.z - math.pi
    #     rot = rot.to_quaternion()
    #     bl_model.armature_obj.location = entity.position
    #     bl_model.armature_obj.rotation_mode = "QUATERNION"
    #     bl_model.armature_obj.rotation_quaternion = rot
    #     ob = bl_model.armature_obj
    #     bpy.ops.object.select_all(action="DESELECT")
    #     ob.select_set(True)
    #     bpy.context.view_layer.objects.active = ob
    #     bpy.ops.object.mode_set(mode='POSE')
    #     for bone_ in entity.bones:  # type: Bone
    #         print(bone_)
    #         bone = ob.pose.bones.get(bone_.name)
    #         if bone:
    #             rot = mathutils.Quaternion()
    #             rot.x, rot.y, rot.z, rot.w = bone_.rotation_q
    #             # rot.x,rot.y,rot.z,rot.w = bone_.valueOrientation
    #             rot = rot.to_euler('YXZ')
    #             mat = mathutils.Matrix.Translation(bone_.position) @ rot.to_matrix().to_4x4()
    #             bone.matrix_basis.identity()
    #             bone.matrix = bone.parent.matrix @ mat if bone.parent else mat
    #         else:
    #             print("Missing", bone_.name, "bone")
    #     bpy.ops.object.mode_set(mode='OBJECT')
    # #
    # def load_lights(self):
    #     for aset, cset in self.lights:  # type: datamodel.Element,datamodel.Element
    #         transform = self.get_element(aset.controls, 'transform', 'DmeTransformControl')
    #         pos = transform.valuePosition
    #         rot = mathutils.Quaternion()
    #         rot.x, rot.y, rot.z, rot.w = transform.valueOrientation
    #         verticalFOV = self.get_element(aset.controls, 'verticalFOV', 'DmElement').channel.toElement
    #         intensity = self.get_element(aset.controls, 'intensity', 'DmElement').channel.toElement
    #         r = self.get_element(aset.controls, 'color_red', 'DmElement').value
    #         g = self.get_element(aset.controls, 'color_green', 'DmElement').value
    #         b = self.get_element(aset.controls, 'color_blue', 'DmElement').value
    #         print(aset.items())
    #         print(transform.items())
    #         fov = self.lerp(verticalFOV.value, verticalFOV.lo, verticalFOV.hi)
    #         print(fov)
    #         rot = rot.to_euler('XYZ')
    #         rot.x, rot.y, rot.z = rot.y, rot.x, rot.z
    #         rot.x = math.pi / 2 - rot.x
    #         rot.z = rot.z - math.pi / 2
    #         intensity = self.lerp(intensity.value, intensity.lo, intensity.hi)
    #
    #         light_data = bpy.data.lights.new(name="light_2.80", type='SPOT')
    #         light_data.energy = 30
    #
    #         # create new object with our light datablock
    #         light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)
    #
    #         # link light object
    #         bpy.context.collection.objects.link(light_object)
    #
    #         # make it active
    #         bpy.context.view_layer.objects.active = light_object
    #
    #         lamp = light_object
    #         lamp.rotation_euler = rot
    #         lamp.location = pos
    #
    #         lamp.name = aset.name
    #         lamp_data = lamp.data
    #         lamp.scale = [100, 100, 100]
    #         lamp_data.spot_size = fov * (math.pi / 180)
    #         lamp_data.use_nodes = True
    #         lamp_nodes = lamp_data.node_tree.nodes['Emission']
    #         lamp_nodes.inputs['Strength'].default_value = intensity * 10
    #         lamp_nodes.inputs['Color'].default_value = (r, g, b, 1)
    #
    # def create_cameras(self):
    #     for aset, cset in self.cameras:  # type: datamodel.Element,datamodel.Element
    #         name = aset.name
    #         cam = bpy.data.cameras.new(name)
    #         main_collection = bpy.data.collections.new("SFM OBJECTS")
    #         bpy.context.scene.collection.children.link(main_collection)
    #         cam_ob = bpy.data.objects.new(name, cam)
    #         main_collection.objects.link(cam_ob)
    #         cam_ob.data.lens_unit = 'MILLIMETERS'
    #         transform = self.get_element(aset.controls, 'transform', 'DmeTransformControl')
    #         print(transform.positionChannel.toElement.items())
    #         print(transform.orientationChannel.toElement.items())
    #         print(transform.items())
    #         pos = transform.valuePosition
    #         rot = mathutils.Quaternion()
    #         rot.x, rot.y, rot.z, rot.w = transform.valueOrientation
    #         rot = rot.to_euler('XYZ')
    #         # rot.y = rot.y - math.pi
    #         rot.x, rot.y, rot.z = rot.y, rot.x, rot.z
    #         rot.x = math.pi / 2 - rot.x
    #         rot.z = rot.z - math.pi / 2
    #         focalDistance = self.get_element(aset.controls, 'focalDistance', 'DmElement').channel.toElement
    #         focalDistance = self.lerp(focalDistance.value, focalDistance.lo, focalDistance.hi)
    #         cam_ob.data.lens = focalDistance
    #         cam_ob.location = pos
    #         cam_ob.rotation_euler = rot
    #         cam_ob.data.clip_end = 100 * 500
    #         cam_ob.scale = [100, 100, 100]
    #
    #         # print(focalDistance)
