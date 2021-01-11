from typing import Dict, Type

from .shader_base import ShaderBase
from ..logging import BPYLoggingManager
from ...goldsrc.mdl.structs.texture import StudioTexture
from ...source1.vmt.valve_material import VMT
from .shaders.source1_shader_base import Source1ShaderBase
from .shaders.goldsrc_shader_base import GoldSrcShaderBase

# from .shaders.source2_shader_base import Source2ShaderBase

from .shaders.source1_shaders import eyerefract, cable, unlit_generic, lightmap_generic, vertexlit_generic, \
    worldvertextransition
from .shaders.goldsrc_shaders import goldsrc_shader

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('blender_material')


class MaterialLoaderBase:
    def __init__(self, material_name):
        self.material_name: str = material_name[-63:]

    def create_material(self):
        handler: ShaderBase = self._handlers.get(self.vmt.shader, ShaderBase)(self.vmt)
        handler.create_nodes(self.material_name)
        handler.align_nodes()
        if self.vmt.shader not in self._handlers:
            logger.error(f'Shader "{self.vmt.shader}" not currently supported by SourceIO')
        pass


class Source1MaterialLoader(MaterialLoaderBase):
    _handlers: Dict[str, Type[Source1ShaderBase]] = dict()
    sub: Type[ShaderBase]
    for sub in Source1ShaderBase.all_subclasses():
        print(f'Registered Source1 material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, file_object, material_name):
        super().__init__(material_name)
        self.material_name: str = material_name[-63:]
        self.vmt: VMT = VMT(file_object)

        self.vmt.parse()

    def create_material(self):
        handler: Source1ShaderBase = self._handlers.get(self.vmt.shader, Source1ShaderBase)(self.vmt)
        handler.create_nodes(self.material_name)
        handler.align_nodes()
        if self.vmt.shader not in self._handlers:
            logger.error(f'Shader "{self.vmt.shader}" not currently supported by SourceIO')
        pass


class GoldSrcMaterialLoader(MaterialLoaderBase):
    _handlers: Dict[str, Type[GoldSrcShaderBase]] = dict()
    sub: Type[ShaderBase]
    for sub in GoldSrcShaderBase.all_subclasses():
        print(f'Registered GoldSrc material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, goldsrc_material: StudioTexture, material_name):
        super().__init__(material_name)
        self.material_name: str = material_name[-63:]
        self.texture_data = goldsrc_material

    def create_material(self):
        handler: GoldSrcShaderBase = self._handlers['goldsrc_shader'](self.texture_data)
        handler.create_nodes(self.material_name)
        handler.align_nodes()
