"""WorldVertexTransition lives with LightmappedGeneric.

In the SDK, ``worldvertextransition.cpp`` fills a ``LightmappedGeneric_DX9_Vars_t``
and calls ``DrawLightmappedGeneric_DX9`` -- it is the same shader under a different
name -- so the implementation is shared. This module only re-exports the class to
keep the historical import path working.
"""
from .lightmap_generic import WorldVertexTransition

__all__ = ['WorldVertexTransition']
