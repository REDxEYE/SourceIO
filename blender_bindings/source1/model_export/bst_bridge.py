"""Access Blender Source Tools for the optional BST compiler backend."""

from __future__ import annotations

import importlib

import bpy


class SourceToolsError(RuntimeError):
    pass


def _source_tools():
    try:
        module = importlib.import_module('io_scene_valvesource')
        state = importlib.import_module('io_scene_valvesource.utils').State
        bpy.ops.export_scene.smd.get_rna_type()
    except (ImportError, AttributeError, RuntimeError, KeyError) as exc:
        raise SourceToolsError('Enable Blender Source Tools (io_scene_valvesource) for the BST compiler backend') from exc
    if not hasattr(module, 'register') or not hasattr(state, 'update_scene'):
        raise SourceToolsError('The installed Blender Source Tools API is not compatible with SourceIO')
    return state


