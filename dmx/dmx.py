import sys
import os.path
from pathlib import Path
from typing import List

import bpy
import mathutils

import math

from ..utilities.valve_utils import GameInfoFile
from ..utilities import datamodel

from ..mdl.mdl2model import Source2Blender

import re


class DMX:

    def _get_proj_root(self, path: Path):
        if path.parts[-1] == 'game':
            return path.parent
        else:
            return self._get_proj_root(path.parent)

    def find_map(self, mapname):
        for game in self.gameinfo:
            maps_folder = os.path.join(self.sfm_path, 'game', game, 'maps')
            allmaps = os.listdir(maps_folder)
            for map in allmaps:
                if mapname == map:
                    return os.path.join(maps_folder, map)
        else:
            sys.stderr.write('Can\'t find map {}'.format(mapname))
            return False

    def find_model(self, model: str):
        return self.gameinfo.find_file(model, extention='.mdl',use_recursive=True)

    def __init__(self, filepath, game_dir=None):
        self.DMX = datamodel.load(filepath)
        self.filepath = Path(filepath)
        if not game_dir:
            game_dir = self._get_proj_root(self.filepath)
        os.environ['VProject'] = str(game_dir)
        self.gameinfo = None
        gameinfo_path = Path(game_dir) / 'gameinfo.txt'
        if gameinfo_path.is_file():
            self.gameinfo = GameInfoFile(gameinfo_path)
        self.lights = []
        self.models = []
        self.cameras = []
        self.session = self.DMX.find_elements('session')[0]
        active_clip = self.session.activeClip
        self.map = active_clip.mapname
        sub_clip_track_group = active_clip.subClipTrackGroup
        tracks = sub_clip_track_group.tracks[0]
        film_clip = tracks.children[0]
        channel_clips = film_clip.trackGroups[0].tracks[0].children
        animation_sets = film_clip.animationSets
        for cset, aset in zip(channel_clips, animation_sets):  # type: datamodel.Element,datamodel.Element
            # pass
            print(aset.name, cset.name)
            print(aset.id, cset.id)
            print(aset.type, cset.type)
            # print(aset.items())
            if 'camera' in dict(aset.items()):
                self.cameras.append((aset, cset))
            if 'light' in dict(aset.items()):
                self.lights.append((aset, cset))
                continue
            if 'gameModel' in dict(aset.items()):
                gamemodel = aset.gameModel
                modelName = gamemodel.modelName
                self.models.append((modelName, aset, cset))

    # def load_map(self):
    #     try:
    #         import BSP_import
    #     except:
    #         return
    #     map = self.find_map(self.map)
    #     if map:
    #         BSP_import.mesh(map, False, os.path.join(self.sfm_path, 'game', 'usermod'), False)

    def get_root_transform(self, controls: List[datamodel.Element]):
        for control in controls:
            if control.name == 'rootTransform' and control.type == 'DmeTransformControl':
                return control

    def get_element(self, controls: List[datamodel.Element], name, type):
        for control in controls:
            if control.name == name and control.type == type:
                return control

    @staticmethod
    def lerp(value, lo, hi):
        f = hi - lo
        return lo + (f * value)

    def load_models(self):

        for model_name, aset, cset in self.models:
            model = self.find_model(model_name)
            root = self.get_root_transform(aset.controls)
            # croot = self.get_element(cset.channels,)
            coords = root.valuePosition
            rot = mathutils.Quaternion(root.valueOrientation)
            rot = rot.to_euler('XYZ')
            rot.x, rot.y, rot.z = rot.y, rot.x, rot.z
            rot.x = math.pi / 2 - rot.x
            rot.z = rot.z - math.pi / 2
            # print(coords, rot)
            # print(aset.controls)
            # print(aset.controls[0].name)
            # print(dict(aset.controls[0].items()))
            bl_model = Source2Blender(model, True, None, co = coords, rot=rot, custom_name= aset.name)
            bl_model.load()
            ob = bl_model.armature_obj
            bpy.ops.object.select_all(action="DESELECT")
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
            bpy.ops.object.mode_set(mode='POSE')
            for bone_ in aset.controls:  # type: datamodel.Element
                if bone_.type == 'DmeTransformControl':
                    bone = ob.pose.bones.get(bone_.name)
                    if bone:
                        cbonep = self.get_element(cset.channels, bone_.name + '_p', 'DmeChannel').toElement
                        cboneo = self.get_element(cset.channels, bone_.name + '_o', 'DmeChannel').toElement
                        # coords = mathutils.Vector(bone_.valuePosition)
                        coords = mathutils.Vector(cbonep.position)
                        rot = mathutils.Quaternion()
                        rot.x, rot.y, rot.z, rot.w = cboneo.orientation
                        # rot.x,rot.y,rot.z,rot.w = bone_.valueOrientation
                        rot = rot.to_euler('YXZ')
                        mat = mathutils.Matrix.Translation(coords) @ rot.to_matrix().to_4x4()
                        bone.matrix_basis.identity()
                        if bone.parent:
                            bone.matrix = bone.parent.matrix @ mat
                        else:
                            bone.matrix = mat
            bpy.ops.object.mode_set(mode='OBJECT')

    def load_lights(self):
        for aset, cset in self.lights:  # type: datamodel.Element,datamodel.Element
            transform = self.get_element(aset.controls, 'transform', 'DmeTransformControl')
            pos = transform.valuePosition
            rot = mathutils.Quaternion()
            rot.x, rot.y, rot.z, rot.w = transform.valueOrientation
            verticalFOV = self.get_element(aset.controls, 'verticalFOV', 'DmElement').channel.toElement
            intensity = self.get_element(aset.controls, 'intensity', 'DmElement').channel.toElement
            r = self.get_element(aset.controls, 'color_red', 'DmElement').value
            g = self.get_element(aset.controls, 'color_green', 'DmElement').value
            b = self.get_element(aset.controls, 'color_blue', 'DmElement').value
            print(aset.items())
            print(transform.items())
            fov = self.lerp(verticalFOV.value, verticalFOV.lo, verticalFOV.hi)
            print(fov)
            rot = rot.to_euler('XYZ')
            rot.x, rot.y, rot.z = rot.y, rot.x, rot.z
            rot.x = math.pi / 2 - rot.x
            rot.z = rot.z - math.pi / 2
            intensity = self.lerp(intensity.value, intensity.lo, intensity.hi)

            light_data = bpy.data.lights.new(name="light_2.80", type='SPOT')
            light_data.energy = 30

            # create new object with our light datablock
            light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

            # link light object
            bpy.context.collection.objects.link(light_object)

            # make it active
            bpy.context.view_layer.objects.active = light_object

            lamp = light_object
            lamp.rotation_euler=rot
            lamp.location=pos

            lamp.name = aset.name
            lamp_data = lamp.data
            lamp.scale = [100, 100, 100]
            lamp_data.spot_size = fov * (math.pi / 180)
            lamp_data.use_nodes = True
            lamp_nodes = lamp_data.node_tree.nodes['Emission']
            lamp_nodes.inputs['Strength'].default_value = intensity * 10
            lamp_nodes.inputs['Color'].default_value = (r, g, b, 1)

    def create_cameras(self):
        for aset, cset in self.cameras:  # type: datamodel.Element,datamodel.Element
            name = aset.name
            cam = bpy.data.cameras.new(name)
            main_collection = bpy.data.collections.new("SFM OBJECTS")
            bpy.context.scene.collection.children.link(main_collection)
            cam_ob = bpy.data.objects.new(name, cam)
            main_collection.objects.link(cam_ob)
            cam_ob.data.lens_unit = 'MILLIMETERS'
            transform = self.get_element(aset.controls, 'transform', 'DmeTransformControl')
            print(transform.positionChannel.toElement.items())
            print(transform.orientationChannel.toElement.items())
            print(transform.items())
            pos = transform.valuePosition
            rot = mathutils.Quaternion()
            rot.x, rot.y, rot.z, rot.w = transform.valueOrientation
            rot = rot.to_euler('XYZ')
            # rot.y = rot.y - math.pi
            rot.x, rot.y, rot.z = rot.y, rot.x, rot.z
            rot.x = math.pi / 2 - rot.x
            rot.z = rot.z - math.pi / 2
            focalDistance = self.get_element(aset.controls, 'focalDistance', 'DmElement').channel.toElement
            focalDistance = self.lerp(focalDistance.value, focalDistance.lo, focalDistance.hi)
            cam_ob.data.lens = focalDistance
            cam_ob.location = pos
            cam_ob.rotation_euler = rot
            cam_ob.data.clip_end = 100 * 500
            cam_ob.scale = [100, 100, 100]

            # print(focalDistance)


if __name__ == '__main__':
    a = DMX(r'E:\PYTHON\DMX_reader\test_data\some_tests6.dmx', r'H:\SteamLibrary\SteamApps\common\SourceFilmmaker')
    a.load_models()
    a.load_lights()
    a.create_cameras()
