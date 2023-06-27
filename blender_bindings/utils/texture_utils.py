import os
from pathlib import Path
from typing import Tuple, Optional

import bpy
import numpy as np

from ...logger import SLoggingManager

logger = SLoggingManager().get_logger("TextureUtils")


def check_texture_cache(texture_path: Path) -> Optional[bpy.types.Image]:
    if bpy.context.scene.get("texture_name_to_texture", None) is None:
        bpy.context.scene["texture_name_to_texture"] = {}
    short_name = bpy.context.scene["texture_name_to_texture"].get(texture_path.as_posix(), None)
    if short_name is not None:
        if short_name + '.png' in bpy.data.images:
            return bpy.data.images[f'{short_name}.png']
        elif short_name + '.hdr' in bpy.data.images:
            return bpy.data.images[f'{short_name}.hdr']
    if bpy.context.scene.TextureCachePath == "":
        return None
    full_path = Path(bpy.context.scene.TextureCachePath) / texture_path.with_suffix(".png")
    image = None
    if full_path.exists():
        image = bpy.data.images.load(full_path.as_posix(), check_existing=True)
    full_path = full_path.with_suffix(".hdr")
    if full_path.exists():
        image = bpy.data.images.load(full_path.as_posix(), check_existing=True)
    full_path = full_path.with_suffix(".tga")
    if full_path.exists():
        image = bpy.data.images.load(full_path.as_posix(), check_existing=True)
    if image is None:
        return
    logger.info(f"Loaded {texture_path!r} texture from disc")
    image.alpha_mode = "CHANNEL_PACKED"
    image.name = texture_path.stem
    return image


def create_and_cache_texture(texture_path: Path, dimensions: Tuple[int, int], data: np.ndarray, is_hdr: bool = False,
                             invert_y: bool = False):
    if bpy.context.scene.get("texture_name_to_texture", None) is None:
        bpy.context.scene["texture_name_to_texture"] = {}
    image = bpy.data.images.new(texture_path.stem, width=dimensions[0], height=dimensions[1], alpha=True)
    image.alpha_mode = "CHANNEL_PACKED"
    image.file_format = "HDR" if is_hdr else "PNG"
    bpy.context.scene["texture_name_to_texture"][texture_path.as_posix()] = image.name

    if invert_y and not is_hdr:
        data[:, :, 1] = 1 - data[:, :, 1]

    image.pixels.foreach_set(data.ravel())
    if bpy.context.scene.TextureCachePath != "":
        save_path = Path(bpy.context.scene.TextureCachePath) / texture_path
        save_path = save_path.with_suffix(".hdr" if is_hdr else ".png")
        os.makedirs(save_path.parent, exist_ok=True)
        image.save(filepath=save_path.as_posix())
        image.filepath = save_path.as_posix()
        logger.info(f"Save {texture_path.as_posix()!r} texture to disc: {save_path}")
    else:
        image.pack()
        logger.info(f"Save {texture_path.as_posix()!r} texture to memory")
    return image
