from collections import defaultdict
from typing import List, Optional, Dict
import bpy

from ...library.source1.mdl.v49.mdl_file import MdlV49 as S1Mdl
from ...library.goldsrc.mdl_v10.mdl_file import Mdl as GMdlV4
from ...library.goldsrc.mdl_v4.mdl_file import Mdl as GMdl
from ...library.source2.resource_types import ValveCompiledModel
from ...library.source1.vtx.v7.vtx import Vtx
from ...library.source1.vvd import Vvd


class ModelContainer:
    def __init__(self):
        self.armature: Optional[bpy.types.Object] = None
        self.objects: List[bpy.types.Object] = []
        self.bodygroups: Dict[str, List[bpy.types.Object]] = defaultdict(list)

    def clone(self):
        raise NotImplementedError()


class GoldSrcModelContainer(ModelContainer):
    def clone(self):
        new_container = GoldSrcModelContainer(self.mdl)
        all_objects = self.objects.copy()
        for body_group_name, objects in self.bodygroups.items():
            for obj in objects:
                all_objects.remove(obj)
                mesh_data = obj.data.copy()
                mesh_obj = obj.copy()

                # mesh_obj['skin_groups'] = obj['skin_groups']
                # mesh_obj['active_skin'] = obj['active_skin']
                # mesh_obj['model_type'] = obj['model_type']
                mesh_obj.data = mesh_data
                new_container.objects.append(mesh_obj)
                new_container.bodygroups[body_group_name].append(mesh_obj)
        if self.armature:
            arm_data = self.armature.data.copy()
            arm_obj = self.armature.copy()
            arm_obj.data = arm_data
            new_container.armature = arm_obj
            for obj in new_container.objects:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE':
                        mod.object = arm_obj
                        obj.parent = arm_obj
        return new_container

    def __init__(self, mdl: GMdl):
        super().__init__()
        self.mdl: GMdl = mdl


class GoldSrcV4ModelContainer(GoldSrcModelContainer):
    def __init__(self, mdl: GMdlV4):
        super().__init__(mdl)
        self.mdl: GMdlV4 = mdl


class Source1ModelContainer(ModelContainer):
    def __init__(self, mdl: S1Mdl, vvd: Optional[Vvd], vtx: Vtx):
        super().__init__()
        self.mdl: S1Mdl = mdl
        self.vvd: Vvd = vvd
        self.vtx: Vtx = vtx
        self.attachments = []

    def clone(self):
        new_container = Source1ModelContainer(self.mdl, self.vvd, self.vtx)
        all_objects = self.objects.copy()
        for body_group_name, objects in self.bodygroups.items():
            for obj in objects:
                all_objects.remove(obj)
                mesh_data = obj.data.copy()
                mesh_obj = obj.copy()

                mesh_obj['skin_groups'] = obj['skin_groups']
                mesh_obj['active_skin'] = obj['active_skin']
                mesh_obj['model_type'] = obj['model_type']
                mesh_obj.data = mesh_data
                new_container.objects.append(mesh_obj)
                new_container.bodygroups[body_group_name].append(mesh_obj)
        if self.armature:
            arm_data = self.armature.data.copy()
            arm_obj = self.armature.copy()
            arm_obj.data = arm_data
            new_container.armature = arm_obj
            for obj in new_container.objects:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE':
                        mod.object = arm_obj
                        obj.parent = arm_obj
        return new_container


class Source2ModelContainer(ModelContainer):
    def __init__(self, vmdl: ValveCompiledModel):
        super().__init__()
        self.vmdl = vmdl
        self.physics_objects: List[bpy.types.Object] = []
        self.attachments = []

    def clone(self):
        new_container = Source2ModelContainer(self.vmdl)
        all_objects = self.objects.copy()
        for body_group_name, objects in self.bodygroups.items():
            for obj in objects:
                all_objects.remove(obj)
                mesh_data = obj.data.copy()
                mesh_obj = obj.copy()

                mesh_obj['skin_groups'] = obj['skin_groups']
                mesh_obj['active_skin'] = obj['active_skin']
                mesh_obj['model_type'] = obj['model_type']
                mesh_obj.data = mesh_data
                new_container.objects.append(mesh_obj)
                new_container.bodygroups[body_group_name].append(mesh_obj)
        if self.armature:
            arm_data = self.armature.data.copy()
            arm_obj = self.armature.copy()
            arm_obj.data = arm_data
            new_container.armature = arm_obj
            for obj in new_container.objects:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE':
                        mod.object = arm_obj
                        obj.parent = arm_obj
            if self.attachments:
                for attachment in self.attachments:
                    new_attachment = attachment.copy()
                    new_attachment.parent = arm_obj
                    new_container.attachments.append(new_attachment)
        return new_container
