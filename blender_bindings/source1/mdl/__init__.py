from dataclasses import dataclass
from typing import Optional

import bpy

from ...shared.model_container import Source1ModelContainer
from ...utils.utils import get_new_unique_collection
from ....library.utils.byte_io_mdl import ByteIO


@dataclass
class FileImport:
    mdl_file: ByteIO
    vvd_file: ByteIO
    vtx_file: ByteIO
    vvc_file: Optional[ByteIO]
    phy_file: Optional[ByteIO]


def put_into_collections(model_container: Source1ModelContainer, model_name,
                         parent_collection=None, bodygroup_grouping=False):
    static_prop = model_container.armature is None
    if not static_prop:
        master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)
    else:
        master_collection = parent_collection or bpy.context.scene.collection
    model_container.collection = master_collection
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
    return master_collection
