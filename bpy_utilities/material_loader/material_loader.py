from pathlib import Path
from typing import Dict, Type, Any, Union

from .shader_base import ShaderBase
from ..logging import BPYLoggingManager
from ...goldsrc.mdl.structs.texture import StudioTexture
from ...source1.vmt.valve_material import VMT

from .shaders.goldsrc_shader_base import GoldSrcShaderBase
from .shaders.source1_shader_base import Source1ShaderBase
from .shaders.source2_shader_base import Source2ShaderBase

from .shaders.source1_shaders import eyerefract, cable, unlit_generic, lightmap_generic, vertexlit_generic, \
    worldvertextransition
from .shaders.goldsrc_shaders import goldsrc_shader
from .shaders.source2_shaders import vr_complex, vr_skin, vr_eyeball

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('material_loader')


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


class Source2MaterialLoader(MaterialLoaderBase):
    _handlers: Dict[str, Type[Source2ShaderBase]] = dict()
    sub: Type[ShaderBase]
    for sub in Source2ShaderBase.all_subclasses():
        print(f'Registered GoldSrc material handler for {sub.__name__} shader')
        _handlers[sub.SHADER] = sub

    def __init__(self, source2_material_data, material_name, resources: Dict[Union[str, int], Path]):
        super().__init__(material_name)
        self.material_name: str = material_name[-63:]
        self.resources = resources
        self.texture_data = source2_material_data

    def create_material(self):
        handler: Source2ShaderBase = self._handlers.get(
            self.texture_data['m_shaderName'], Source2ShaderBase)(self.texture_data, self.resources)
        handler.create_nodes(self.material_name)
        handler.align_nodes()
