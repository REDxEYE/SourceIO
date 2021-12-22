import traceback
from pathlib import Path
from typing import Dict, Type, Union

from .shader_base import ShaderBase
from ...library.goldsrc.mdl_v10.structs.texture import StudioTexture
from ...logger import SLoggingManager
from ...library.source1.vmt import VMT

from .shaders.goldsrc_shader_base import GoldSrcShaderBase
from .shaders.source1_shader_base import Source1ShaderBase
from .shaders.source2_shader_base import Source2ShaderBase

from .shaders import source1_shaders
from .shaders import source2_shaders
from .shaders import goldsrc_shaders
from .shaders import debug_material

log_manager = SLoggingManager()
logger = log_manager.get_logger('MaterialLoader')


class MaterialLoaderBase:
    def __init__(self, material_name):
        self.material_name: str = material_name[-63:]

    def create_material(self):
        pass


class Source1MaterialLoader(MaterialLoaderBase):
    _handlers: Dict[str, Type[Source1ShaderBase]] = dict()
    sub: Type[ShaderBase]
    for sub in Source1ShaderBase.all_subclasses():
        logger.info(f'Registered Source1 material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, file_object, material_name):
        super().__init__(material_name)
        self.material_name: str = material_name[-63:]
        self.vmt: VMT = VMT(file_object, self.material_name)

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
        logger.info(f'Registered GoldSrc material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, goldsrc_material: StudioTexture, material_name):
        super().__init__(material_name)
        self.material_name: str = material_name[-63:]
        self.texture_data = goldsrc_material

    def create_material(self):
        handler: GoldSrcShaderBase = self._handlers['goldsrc_shader'](self.texture_data)
        try:
            handler.create_nodes(self.material_name)
        except Exception as ex:
            logger.error(f'Failed to load material, due to {ex} error')
            traceback.print_exc()
            logger.debug(f'Failed material: {self.material_name}:{self.texture_data.name}')
        handler.align_nodes()


class Source2MaterialLoader(MaterialLoaderBase):
    _handlers: Dict[str, Type[Source2ShaderBase]] = dict()
    sub: Type[ShaderBase]
    for sub in Source2ShaderBase.all_subclasses():
        logger.info(f'Registered Source2 material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, source2_material_data, material_name, resources: Dict[Union[str, int], Path]):
        super().__init__(material_name)
        self.material_name: str = material_name[-63:]
        self.resources = resources
        self.texture_data = source2_material_data

    def create_material(self):
        shader = self.texture_data['m_shaderName']
        handler: Source2ShaderBase = self._handlers.get(
            shader, Source2ShaderBase)(self.texture_data, self.resources)

        if shader not in self._handlers:
            logger.error(f'Shader "{shader}" not currently supported by SourceIO')
        try:
            handler.create_nodes(self.material_name)
        except Exception as ex:
            logger.error(f'Failed to load material, due to {ex} error')
            traceback.print_exc()
            logger.debug(f'Failed material: {self.material_name}')
        handler.align_nodes()
