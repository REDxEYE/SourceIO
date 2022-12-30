from typing import List

from mathutils import Euler, Matrix, Quaternion, Vector


def convert_source_rotation(rot: List[float]):
    qrot = Quaternion([rot[0], rot[1], -rot[3], rot[2]])
    # qrot.rotate(Euler([0, 0, 90]))
    return qrot


def convert_source_position(pos: List[float]):
    pos = Vector([pos[0], pos[2], -pos[1]])
    # pos.rotate(Euler([0, -90, 0]))
    return pos


def convert_source_animset_rotation(rot: List[float]):
    return convert_source_rotation(rot)


def convert_source_animset_position(pos: List[float]):
    pos = Vector([pos[0], -pos[2], pos[1]])
    return pos
