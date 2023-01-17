from typing import BinaryIO, Optional

from ...library.utils import Buffer
from ..goldsrc.mdl_v4.import_mdl import import_model as import_model_v4
from ..goldsrc.mdl_v6.import_mdl import import_model as import_model_v6
from ..goldsrc.mdl_v10.import_mdl import import_model as import_model_v10


def import_model(name: str, mdl_buffer: Buffer, mdl_texture_file: Optional[Buffer], scale=1.0,
                 parent_collection=None, disable_collection_sort=False, re_use_meshes=False):
    assert mdl_buffer.read(4) == b'IDST'
    version = mdl_buffer.read_int32()
    mdl_buffer.seek(0)
    if version == 4:
        return import_model_v4(name, mdl_buffer, scale, parent_collection, disable_collection_sort, re_use_meshes)
    elif version == 6:
        return import_model_v6(mdl_buffer, mdl_texture_file, scale, parent_collection, disable_collection_sort,
                               re_use_meshes)
    elif version == 10:
        return import_model_v10(mdl_buffer, mdl_texture_file, scale, parent_collection, disable_collection_sort,
                                re_use_meshes)
