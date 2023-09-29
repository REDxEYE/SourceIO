from dataclasses import dataclass
from typing import Optional

import bpy

from ....library.utils import Buffer
from ...shared.model_container import Source1ModelContainer
from ...utils.utils import get_new_unique_collection


@dataclass
class FileImport:
    mdl_file: Buffer
    vvd_file: Optional[Buffer]
    vtx_file: Buffer
    vvc_file: Optional[Buffer]
    phy_file: Optional[Buffer]

    def is_valid(self):
        if self.mdl_file is None:
            return False
        if self.mdl_file.size() == 0:
            return False
        return True


def put_into_collections(model_container: Source1ModelContainer, model_name,
                         parent_collection=None, bodygroup_grouping=False):
    master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)

    for bodygroup_name, meshes in model_container.bodygroups.items():
        if bodygroup_grouping:
            body_part_collection = get_new_unique_collection(bodygroup_name, master_collection)
        else:
            body_part_collection = master_collection

        for mesh in meshes:
            body_collection = get_new_unique_collection(mesh.name, body_part_collection)
            body_collection.objects.link(mesh)
    if model_container.armature:
        master_collection.objects.link(model_container.armature)

    if model_container.attachments:
        attachments_collection = get_new_unique_collection(model_name + '_ATTACHMENTS', master_collection)
        for attachment in model_container.attachments:
            attachments_collection.objects.link(attachment)
    if model_container.physics:
        physics_collection = get_new_unique_collection(model_name + '_PHYSICS', master_collection)
        for physics in model_container.physics:
            physics_collection.objects.link(physics)
    model_container.collection = master_collection
    return master_collection
