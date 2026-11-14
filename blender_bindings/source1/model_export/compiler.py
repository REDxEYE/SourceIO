"""Compile a generated QC through the direct or add-on compiler."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import bpy

from .bst_bridge import SourceToolsError, _source_tools


class CompilationError(RuntimeError):
    pass


def direct_command(qc_path: Path, game_root: Path, command: str) -> list[str]:
    """Expand a configured argv template before writing any export artifacts.

    Example: ``wine /path/to/studiomdl.exe -nop4 -game {game} {qc}``.
    Tokens are parsed before placeholder expansion, so spaces inside either
    resolved path cannot turn into additional command arguments.
    """
    if not command.strip():
        raise CompilationError('Configure a studiomdl command before compiling')
    try:
        tokens = shlex.split(command)
        if not tokens or not any('{qc}' in token for token in tokens):
            raise CompilationError('The compiler command must contain {qc}')
        if not any('{game}' in token for token in tokens):
            raise CompilationError('The compiler command must contain {game}')
        return [token.format(qc=str(qc_path), game=str(game_root)) for token in tokens]
    except (ValueError, KeyError, IndexError, AttributeError) as exc:
        raise CompilationError(f'Invalid studiomdl command: {exc}') from exc


def compile_direct(qc_path: Path, argv: list[str]) -> str:
    """Run a prevalidated command without a shell."""
    try:
        result = subprocess.run(argv, cwd=qc_path.parent, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise CompilationError(f'Cannot launch studiomdl: {exc}') from exc
    output = '\n'.join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if result.returncode:
        raise CompilationError(f'studiomdl exited with code {result.returncode}\n{output}')
    return output


def validate_source_tools_compiler(game_root: Path, engine_root: Path):
    """Check the add-on compiler prerequisites before export."""
    state = _source_tools()
    if not (engine_root / 'studiomdl.exe').is_file():
        raise CompilationError(f'BST requires {engine_root / "studiomdl.exe"}')
    if not (game_root / 'gameinfo.txt').is_file():
        raise CompilationError(f'BST requires {game_root / "gameinfo.txt"}')
    if bpy.context.window is None:
        raise SourceToolsError('Blender Source Tools needs an active Blender window to compile')
    return state


def compile_with_source_tools(qc_path: Path, game_root: Path, engine_root: Path, state) -> str:
    """Use the add-on compiler operator in a disposable scene.

    It constructs ``engine_root/studiomdl.exe`` and requires gameinfo.txt;
    use the direct backend for Wine/Proton wrappers.
    """
    window = bpy.context.window
    previous_scene_name = window.scene.name
    scene = bpy.data.scenes.new('SourceIO_compile')
    scene_name = scene.name
    try:
        window.scene = scene
        scene.vs.game_path = str(game_root)
        scene.vs.engine_path = str(engine_root)
        if not bpy.ops.smd.compile_qc.poll():
            raise CompilationError('BST compiler is unavailable for this game/toolchain')
        from io_scene_valvesource.export_smd import SMD_OT_Compile
        from io_scene_valvesource.utils import Logger
        logger = Logger()
        if SMD_OT_Compile.compileQCs(logger, str(qc_path)) != 1:
            raise CompilationError('; '.join(logger.log_errors) or f'BST failed to compile {qc_path}')
        return '\n'.join(logger.log_warnings)
    finally:
        previous = bpy.data.scenes.get(previous_scene_name)
        if previous is not None:
            window.scene = previous
            state._updateEngineBranch()
            state._validateGamePath()
        stored_scene = bpy.data.scenes.get(scene_name)
        if stored_scene is not None:
            bpy.data.scenes.remove(stored_scene)
