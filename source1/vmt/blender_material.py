from typing import Dict, Type

from ..vmt.valve_material import VMT
from .shader_base import ShaderBase
from .shaders import vertexlit_generic, lightmap_generic, worldvertextransition, cable, unlit_generic, eyerefract
from ...bpy_utils import BPYLoggingManager

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('blender_material')


class BlenderMaterial:
    _handlers: Dict[str, Type[ShaderBase]] = dict()

    sub: Type[ShaderBase]
    for sub in ShaderBase.all_subclasses():
        print(f'Registered handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, file_object, material_name):
        self.material_name: str = material_name[-63:]
        self.vmt: VMT = VMT(file_object)

        self.vmt.parse()

    def create_material(self):
        handler: ShaderBase = self._handlers.get(self.vmt.shader, ShaderBase)(self.vmt)
        handler.create_nodes(self.material_name)
        handler.align_nodes()
        if self.vmt.shader not in self._handlers:
            logger.error(f'Shader "{self.vmt.shader}" not currently supported by SourceIO')
        pass

