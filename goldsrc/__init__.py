from typing import BinaryIO, Optional

from .bsp.import_bsp import BSP
from .mdl.import_mdl import import_model as import_model_v6
from .mdl_v4.import_mdl import import_model as import_model_v4
from ..utilities.byte_io_mdl import ByteIO


def import_model(mdl_file: BinaryIO, mdl_texture_file: Optional[BinaryIO], scale=1.0,
                 parent_collection=None, disable_collection_sort=False, re_use_meshes=False):
    reader = ByteIO(mdl_file)

    assert reader.read(4) == b'IDST'
    version = reader.read_int32()
    reader.seek(0)
    reader.file = None
    if version == 4:
        import_model_v4(mdl_file, scale, parent_collection, disable_collection_sort, re_use_meshes)
    else:
        import_model_v6(mdl_file, mdl_texture_file, scale, parent_collection, disable_collection_sort, re_use_meshes)
