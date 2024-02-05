from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

import bpy

from ...library.models.mdl.v4.mdl_file import Mdl as GMdl
from ...library.models.mdl.v10.mdl_file import Mdl as GMdlV4
from ...library.models.mdl.v36.mdl_file import MdlV36 as S1MdlV36
from ...library.models.mdl.v44.mdl_file import MdlV44 as S1MdlV44
from ...library.models.mdl.v49.mdl_file import MdlV49 as S1MdlV49
from ...library.models.vtx.v7.vtx import Vtx
from ...library.models.vvd import Vvd
from ...library.source2.resource_types import CompiledModelResource


@dataclass
class ModelContainer:
    objects: list[bpy.types.Object]
    bodygroups: dict[str, list[bpy.types.Object]]
    physics_objects: list[bpy.types.Object] = field(default_factory=list)
    attachments: list[bpy.types.Object] = field(default_factory=list)
    armature: Optional[bpy.types.Object] = None
    master_collection: Optional[bpy.types.Collection] = None
