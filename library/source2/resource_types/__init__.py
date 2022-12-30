from .resource import ValveCompiledResource
from .vmat import ValveCompiledMaterial
from .vmdl import ValveCompiledModel
from .vmorf import ValveCompiledMorph
from .vphys import ValveCompiledPhysics
from .vrman import ValveCompiledResourceManifest
from .vtex import ValveCompiledTexture
from .vwrld import ValveCompiledWorld


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
    elif ext == '.vmat_c':
        return ValveCompiledMaterial
    else:
        return ValveCompiledResource
