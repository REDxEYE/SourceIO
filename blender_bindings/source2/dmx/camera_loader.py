import math

import bpy
from mathutils import Euler, Quaternion, Vector

from SourceIO.library.utils.datamodel import load


def load_camera(dmx_camera_path):
    camera_data = load(dmx_camera_path)
    scene = camera_data.root
    camera_info = scene['camera']
    fov = camera_info['fieldOfView']
    aperture = camera_info['aperture']
    if 'channelsClip' in scene:
        camera_clip = scene['channelsClip']
    else:
        camera_clip = scene['animationList']['animations'][0]
    fps = camera_clip['frameRate']

    camera = bpy.data.cameras.new(name=camera_info.name)
    camera_obj = bpy.data.objects.new(camera_info.name, camera)

    bpy.context.scene.collection.objects.link(camera_obj)

    camera_obj.location = Vector(camera_info['transform']['position'])
    camera_obj.rotation_quaternion = Quaternion(camera_info['transform']['orientation'])
    camera.lens = 0.5 * 36 / math.tan(math.radians(fov) / 2)
    camera_obj.rotation_mode = 'QUATERNION'

    for channel in camera_clip['channels']:
        value_log = channel['log']
        value_layer = value_log['layers'][0]
        if channel.name.endswith('_p'):
            for time, value in zip(value_layer['times'], value_layer['values']):
                frame = math.ceil(time * fps)
                pos = Vector([value[1], -value[0], value[2]])
                camera_obj.location = pos
                camera_obj.keyframe_insert(data_path="location", frame=frame)
        elif channel.name.endswith('_o'):
            for time, value in zip(value_layer['times'], value_layer['values']):
                frame = math.ceil(time * fps)
                quat = Quaternion([value[1], value[3], value[0], value[2]])
                quat.rotate(Euler([-math.pi / 2, 0, -math.pi]))
                camera_obj.rotation_quaternion = quat
                camera_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        elif channel.name.endswith('_fieldOfView') or channel.name.endswith('fieldOfView'):
            for time, value in zip(value_layer['times'], value_layer['values']):
                frame = math.ceil(time * fps)
                camera.lens = 0.5 * 36 / math.tan(math.radians(value) / 2)
                camera.keyframe_insert(data_path="lens", frame=frame)

    pass
