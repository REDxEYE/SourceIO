from dataclasses import dataclass, field
from typing import Optional

import bpy


@dataclass
class ModelContainer:
    objects: list[bpy.types.Object]
    bodygroups: dict[str, list[bpy.types.Object]]
    physics_objects: list[bpy.types.Object] = field(default_factory=list)
    attachments: list[bpy.types.Object] = field(default_factory=list)
    armature: Optional[bpy.types.Object] = None
    master_collection: Optional[bpy.types.Collection] = None
