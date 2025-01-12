import traceback
from typing import Type

import bpy

from SourceIO.library.models.mdl.v10.structs.texture import StudioTexture
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.vmt import VMT
from SourceIO.library.source2 import CompiledMaterialResource
from SourceIO.logger import SourceLogMan
from .shader_base import ShaderBase
from .shaders.goldsrc_shader_base import GoldSrcShaderBase
from .shaders.source1_shader_base import Source1ShaderBase
from .shaders.source2_shader_base import Source2ShaderBase
from .shaders.source2_shaders.dummy import DummyShader

# noinspection PyUnresolvedReferences
from .shaders import source1_shaders, source2_shaders, goldsrc_shaders
from SourceIO.library.source2.blocks.kv3_block import KVBlock
from ...library.source2.compiled_resource import DATA_BLOCK

log_manager = SourceLogMan()
logger = log_manager.get_logger('MaterialLoader')


class MaterialLoaderBase:
    def __init__(self, material_name):
        self.material_name: str = material_name

    def create_material(self, material: bpy.types.Material):
        pass


class Source1MaterialLoader(MaterialLoaderBase):
    _handlers: dict[str, Type[Source1ShaderBase]] = dict()
    sub: Type[ShaderBase]
    for sub in Source1ShaderBase.all_subclasses():
        logger.info(f'Registered Source1 material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, content_manager: ContentManager, file_object, material_name):
        super().__init__(material_name)
        self.vmt: VMT = VMT(file_object, self.material_name, content_manager)
        self.content_manager = content_manager

    def create_material(self, material: bpy.types.Material):
        handler: Source1ShaderBase = self._handlers.get(self.vmt.shader, Source1ShaderBase)(self.content_manager,
                                                                                            self.vmt)

        handler.create_nodes(material)
        material['shader_type'] = handler._vmt.shader
        try:
            params = handler._vmt.data.to_dict()
            # if (dx90 := (params.get('>=dx90') or params.get('>=DX90'))):
            #    for key, value in dx90.items():
            #        params[key] = value
            #    # unravel it a bit, because dx90 is how we usually see the materials
            material['vmt_parameters'] = handler._vmt.data.to_dict()
        except:
            pass
        handler.align_nodes()
        if self.vmt.shader not in self._handlers:
            logger.error(f'Shader "{self.vmt.shader}" not currently supported by SourceIO')
        return handler.bpy_material


class GoldSrcMaterialLoader(MaterialLoaderBase):
    _handlers: dict[str, Type[GoldSrcShaderBase]] = dict()
    sub: Type[ShaderBase]
    for sub in GoldSrcShaderBase.all_subclasses():
        logger.info(f'Registered GoldSrc material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, goldsrc_material: StudioTexture, material_name):
        super().__init__(material_name)
        self.material_name: str = material_name
        self.texture_data = goldsrc_material

    def create_material(self, material: bpy.types.Material):
        handler: GoldSrcShaderBase = self._handlers['goldsrc_shader'](self.texture_data)
        try:
            handler.create_nodes(material)
        except Exception as ex:
            logger.error(f'Failed to load material, due to {ex} error')
            traceback.print_exc()
            logger.debug(f'Failed material: {self.material_name}:{self.texture_data.name}')
        handler.align_nodes()


class Source2MaterialLoader(MaterialLoaderBase):
    _handlers: dict[str, Type[Source2ShaderBase]] = dict()
    sub: Type[ShaderBase]
    for sub in Source2ShaderBase.all_subclasses():
        logger.info(f'Registered Source2 material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, content_manager: ContentManager, material_resource: CompiledMaterialResource, material_name,
                 tinted: bool = False):
        super().__init__(material_name)
        self.content_manager = content_manager
        self.material_name: str = material_name[:63]
        self.material_resource = material_resource
        self.tinted = tinted

    def create_material(self, material: bpy.types.Material):
        if material.get('source1_loaded', False):
            logger.info(f'Skipping loading of {material} as it already loaded')
            return

        data = self.material_resource.get_block(KVBlock, block_name="DATA")
        if not data:
            return
        shader = data['m_shaderName']
        handler: Source2ShaderBase = self._handlers.get(shader, DummyShader)(self.content_manager,
                                                                             self.material_resource, self.tinted)

        if shader not in self._handlers:
            logger.error(f'Shader "{shader}" not currently supported by SourceIO')
        try:
            handler.create_nodes(material)
        except Exception as ex:
            logger.error(f'Failed to load material, due to {ex} error')
            traceback.print_exc()
            logger.debug(f'Failed material: {self.material_name}')
        handler.align_nodes()
