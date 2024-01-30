from collections import defaultdict
from typing import Optional, Union

import bpy

class ModelContainer:

    def __init__(self,
                 objects: list[bpy.types.Object],
                 armature: Optional[bpy.types.Armature] = None,
                 object_groups: Optional[dict[str, list[bpy.types.Object]]] = None,
                 attachments: Optional[list[bpy.types.Object]] = None,
                 physics: Optional[list[bpy.types.Object]] = None,
                 ):
        self.armature = armature
        self.objects = objects
        self.bodygroups: dict[str, list[bpy.types.Object]] = object_groups or defaultdict(list)
        self.attachments: list[bpy.types.Object] = attachments or []
        self.physics: list[bpy.types.Object] = physics or []
        self.collection = None

    def clone(self):
        all_objects = self.objects.copy()
        new_objects = []
        new_bodygroups = defaultdict(list)
        for body_group_name, objects in self.bodygroups.items():
            for obj in objects:
                all_objects.remove(obj)
                mesh_data = obj.data.copy()
                mesh_obj = obj.copy()

                # mesh_obj['skin_groups'] = obj['skin_groups']
                # mesh_obj['active_skin'] = obj['active_skin']
                # mesh_obj['model_type'] = obj['model_type']
                mesh_obj.data = mesh_data
                new_objects.append(mesh_obj)
                new_bodygroups[body_group_name].append(mesh_obj)
        new_armature = None
        if self.armature:
            arm_data = self.armature.data.copy()
            new_armature = self.armature.copy()
            new_armature.data = arm_data
            for obj in new_objects:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE':
                        mod.object = new_armature
                        obj.parent = new_armature
        return ModelContainer(new_objects, new_armature, new_bodygroups)


class GoldSrcModelContainer(ModelContainer):
    def __init__(self):
        super().__init__([], None, None, None)

    def clone(self):
        new_container = GoldSrcModelContainer()
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


class GoldSrcV4ModelContainer(GoldSrcModelContainer):
    def __init__(self):
        super().__init__()


class Source1ModelContainer(ModelContainer):
    def __init__(self):
        super().__init__([], None, None, None)
        self.physics = []
        self.attachments = []

    def clone(self):
        new_container = Source1ModelContainer()
        for body_group_name, objects in self.bodygroups.items():
            for obj in objects:
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
    def __init__(self):
        super().__init__([], None, None, None)
        self.physics_objects: list[bpy.types.Object] = []
        self.attachments = []

    def clone(self):
        new_container = Source2ModelContainer()
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
