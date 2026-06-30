"""
Load all animations from an MDL file and its associated .ani file.

Handles both:
  - Inline animations (animblock_id == 0): data in the MDL itself
  - External animations (animblock_id > 0): data in a separate .ani file

Also resolves include_models to gather animations from animation sub-MDLs
(e.g. dog.mdl → dog_animations.mdl → dog_animations.ani).
"""
from dataclasses import dataclass
from typing import Optional

import numpy.typing as npt

from SourceIO.library.models.mdl.structs.ani_file import AniFile, AnimBlockEntry, read_anim_block_table
from SourceIO.library.models.mdl.structs.local_animation import StudioAnimDesc, ANIM_DTYPE
from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import Buffer, FileBuffer
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('AnimLoader')


@dataclass(slots=True)
class AnimationData:
    name: str
    fps: float
    frame_count: int
    bone_names: list[str]
    frames: dict[str, npt.NDArray] # dict[bone_name, ANIM_DTYPE]
    # frames: npt.NDArray  # shape: (frame_count, bone_count), dtype=ANIM_DTYPE
    is_looping: bool
    is_delta: bool


def load_animations_from_mdl(mdl, mdl_buffer: Buffer,
                             content_manager: ContentManager,
                             model_path: TinyPath | None = None) -> list[AnimationData]:
    """
    Load all animations from an MDL, including external .ani blocks.
    Returns a list of AnimationData for each successfully loaded animation.
    """
    bones = mdl.bones
    anim_descs: list[StudioAnimDesc] = mdl.anim_descs

    if not anim_descs:
        return []

    ani_buffer = _resolve_ani_file(mdl, content_manager, model_path)
    block_table = _get_block_table(mdl, mdl_buffer)

    raw_ani_buffer = ani_buffer.buffer if ani_buffer is not None else None

    results = []
    for anim_desc in anim_descs:
        try:
            frames = anim_desc.read_animations(mdl_buffer, bones, raw_ani_buffer, block_table)
            if frames is None:
                print(f"Animation {anim_desc.name} has no frames")
                continue
            from .structs.local_animation import AnimDescFlags
            results.append(AnimationData(
                name=anim_desc.name,
                fps=anim_desc.fps,
                frame_count=anim_desc.frame_count,
                bone_names=[b.name for b in bones],
                frames=dict(frames), # convert from defaultdict to normal dict
                is_looping=bool(anim_desc.flags & AnimDescFlags.LOOPING),
                is_delta=bool(anim_desc.flags & AnimDescFlags.DELTA),
            ))
        except Exception as ex:
            logger.error(f"Failed to load animation '{anim_desc.name}': {ex}")
            continue

    return results


def load_all_animations(mdl:MdlV49, mdl_buffer: Buffer,
                        content_manager: ContentManager,
                        model_path: TinyPath | None = None) -> list[AnimationData]:
    """
    Load animations from the main MDL and all its include_models.
    This gives the complete animation set for a character.
    """
    all_animations = load_animations_from_mdl(mdl, mdl_buffer, content_manager, model_path)

    if not hasattr(mdl, 'include_models') or not mdl.include_models:
        return all_animations

    for include_path in mdl.include_models:
        inc_buffer = content_manager.find_file(TinyPath(include_path))
        if inc_buffer is None:
            logger.info(f"Include model not found: {include_path}")
            continue

        try:
            inc_mdl = MdlV49.from_buffer(inc_buffer)
            inc_anims = load_animations_from_mdl(
                inc_mdl, inc_buffer, content_manager, TinyPath(include_path))
            all_animations.extend(inc_anims)
        except Exception as ex:
            logger.error(f"Failed to load include model '{include_path}': {ex}")
            continue

    return all_animations


def _resolve_ani_file(mdl, content_manager: ContentManager,
                      model_path: TinyPath | None) -> Optional[AniFile]:
    """Find and open the .ani file referenced by the MDL's anim_block_name."""
    if not hasattr(mdl, 'header'):
        return None

    anim_block_name = getattr(mdl.header, 'anim_block_name', '')
    if not anim_block_name:
        return None

    ani_path = TinyPath(anim_block_name)
    ani_buffer = content_manager.find_file(ani_path)

    if ani_buffer is None and model_path is not None:
        # Try relative to model path
        relative_ani = model_path.parent / TinyPath(ani_path).name
        ani_buffer = content_manager.find_file(relative_ani)

    if ani_buffer is None and model_path is not None:
        # Try same directory with matching name
        ani_local = model_path.with_suffix('.ani')
        if ani_local.exists():
            ani_buffer = FileBuffer(ani_local)

    if ani_buffer is None:
        logger.info(f"ANI file not found: {anim_block_name}")
        return None

    try:
        return AniFile.from_buffer(ani_buffer)
    except ValueError as ex:
        logger.error(f"Failed to open ANI file {anim_block_name}: {ex}")
        return None


def _get_block_table(mdl, mdl_buffer: Buffer) -> list[AnimBlockEntry]:
    """Read the anim_block table from the MDL header."""
    block_offset = getattr(mdl.header, 'anim_block_offset', 0)
    block_count = getattr(mdl.header, 'anim_block_count', 0)
    return read_anim_block_table(mdl_buffer, block_offset, block_count)


