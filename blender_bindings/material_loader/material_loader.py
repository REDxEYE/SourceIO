import traceback
from enum import Enum
from typing import Type, Any

import bpy

from SourceIO.library.models.mdl.v10.structs.texture import StudioTexture
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.vmt import VMT
from SourceIO.library.source2 import CompiledMaterialResource
from SourceIO.logger import SourceLogMan
from .node_arranger import nodes_iterate
from .shader_base import ShaderBase, ExtraMaterialParameters
from .shaders.goldsrc_shader_base import GoldSrcShaderBase
from .shaders.source1_shader_base import Source1ShaderBase
from .shaders.source2_shader_base import Source2ShaderBase
from .shaders.source2_shaders.dummy import DummyShader

# noinspection PyUnresolvedReferences
from .shaders import source1_shaders, source2_shaders, goldsrc_shaders
from SourceIO.library.source2.blocks.kv3_block import KVBlock
from SourceIO.library.utils.perf_sampler import timed
from ..utils.bpy_utils import is_blender_4_3

log_manager = SourceLogMan()
logger = log_manager.get_logger('MaterialLoader')


class MaterialLoaderBase:
    def __init__(self, material_name):
        self.material_name: str = material_name

    def create_material(self, material: bpy.types.Material):
        pass


class ShaderRegistry:
    _handlers: dict[str, Type[Source1ShaderBase]] = dict()

    for sub in Source1ShaderBase.all_subclasses():
        logger.info(f'Registered Source1 material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub
    for sub in GoldSrcShaderBase.all_subclasses():
        logger.info(f'Registered goldsrc material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub
    for sub in Source2ShaderBase.all_subclasses():
        logger.info(f'Registered Source2 material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    @classmethod
    def source1_create_nodes(cls, content_manager: ContentManager,
                             material: bpy.types.Material,
                             vmt: VMT,
                             extra_parameters: dict[ExtraMaterialParameters, Any]):
        if not cls._initial_setup(material):
            return
        shader = vmt.shader
        if shader not in cls._handlers:
            logger.error(f'Shader "{shader}" not currently supported by SourceIO')

        handler: Source1ShaderBase = cls._handlers.get(shader, Source1ShaderBase)(content_manager, vmt)
        handler.bpy_material = material
        try:
            handler.create_nodes(material, extra_parameters)
        except Exception as e:
            logger.error(f'Failed to load material, due to {e} error')
            traceback.print_exc()
            logger.debug(f'Failed material: {material.name}')
        # params = handler._vmt.data.to_dict()
        material['vmt_parameters'] = vmt.data.to_dict()
        handler.align_nodes()

    @classmethod
    def source2_create_nodes(cls, content_manager: ContentManager,
                             material: bpy.types.Material,
                             material_resource: CompiledMaterialResource,
                             extra_parameters: dict[ExtraMaterialParameters, Any]):
        if not cls._initial_setup(material):
            return
        data = material_resource.get_block(KVBlock, block_name="DATA")
        if not data:
            return
        shader = data['m_shaderName']
        use_tint = extra_parameters.get(ExtraMaterialParameters.USE_OBJECT_TINT, False)
        handler: Source2ShaderBase = cls._handlers.get(shader, DummyShader)(content_manager,
                                                                            material_resource,
                                                                            use_tint
                                                                            )
        handler.bpy_material = material
        try:
            handler.create_nodes(material, extra_parameters)
        except Exception as e:
            logger.error(f'Failed to load material, due to {e} error')
            traceback.print_exc()
            logger.debug(f'Failed material: {material.name}')
        cls.align_nodes(material)
        for unused_texture in handler.unused_textures.copy():
            texture_path = material_resource.get_texture_property(unused_texture, None)
            if texture_path is not None:
                handler.logger.warn(f"Unused texture {unused_texture} {texture_path}")
                handler._get_texture(unused_texture, (0, 0, 0, 0), False)

    @classmethod
    def _initial_setup(cls, material: bpy.types.Material):
        if material.get('source_loaded', False):
            return False

        material.use_nodes = True
        material['source_loaded'] = True
        material.use_nodes = True
        cls._clean_nodes(material)
        if not is_blender_4_3():
            material.blend_method = 'OPAQUE'
            material.shadow_method = 'OPAQUE'
        material.use_screen_refraction = False
        material.refraction_depth = 0.2
        return True

    @staticmethod
    def _clean_nodes(material: bpy.types.Material):
        for node in material.node_tree.nodes:
            material.node_tree.nodes.remove(node)

    @staticmethod
    def align_nodes(material):
        nodes_iterate(material.node_tree)
        material.node_tree.nodes.update()


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
            handler.create_nodes(material,{})
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
            handler.create_nodes(material,{})
        except Exception as ex:
            logger.error(f'Failed to load material, due to {ex} error')
            traceback.print_exc()
            logger.debug(f'Failed material: {self.material_name}')
        handler.align_nodes()
