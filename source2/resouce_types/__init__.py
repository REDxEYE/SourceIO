from .resource import ValveCompiledResource
from .physics import ValveCompiledPhysics
from .material import ValveCompiledMaterial
from .texture import ValveCompiledTexture
from .model import ValveCompiledModel
from .world import ValveCompiledWorld
from .morph import ValveCompiledMorph
from .resource_manifest import ValveCompiledResourceManifest

def get_resource_loader_from_ext(ext: str):
    if ext == '.vmdl_c':
        return ValveCompiledModel
    elif ext == '.vwrld_c':
        return ValveCompiledWorld
    elif ext == '.vtex_c':
        return ValveCompiledTexture
    elif ext == '.vphys_c':
        return ValveCompiledPhysics
    elif ext == '.vrman_c':
        return ValveCompiledResourceManifest
    else:
        return ValveCompiledResource
