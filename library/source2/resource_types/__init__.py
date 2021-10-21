from .resource import ValveCompiledResource
from .vphys import ValveCompiledPhysics
from .vmat import ValveCompiledMaterial
from .vtex import ValveCompiledTexture
from .vmdl import ValveCompiledModel
from .vwrld import ValveCompiledWorld
from .vmorf import ValveCompiledMorph
from .vrman import ValveCompiledResourceManifest


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
