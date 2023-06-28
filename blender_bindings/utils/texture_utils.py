import os
from hashlib import md5
from pathlib import Path
from typing import Tuple, Optional

import bpy
import numpy as np

from ...logger import SLoggingManager

logger = SLoggingManager().get_logger("TextureUtils")


def _get_texture(texture_path: Path, *other_args):
    md_ = md5(texture_path.as_posix().encode("ascii"))
    for key in other_args:
        if key:
            md_.update(key.encode("ascii"))
    key = md_.hexdigest()
    cache = bpy.context.scene.get("texture_name_to_texture", {})
    if key in cache:
        return cache[key]


def _add_texture(texture_path: Path, real_name: str, *other_args):
    md_ = md5(texture_path.as_posix().encode("ascii"))
    for key in other_args:
        if key:
            md_.update(key.encode("ascii"))
    key = md_.hexdigest()
    cache = bpy.context.scene.get("texture_name_to_texture", {})
    cache[key] = real_name
    bpy.context.scene["texture_name_to_texture"] = cache


def check_texture_cache(texture_path: Path) -> Optional[bpy.types.Image]:
    short_name = _get_texture(texture_path)
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
    image = bpy.data.images.new(texture_path.stem, width=dimensions[0], height=dimensions[1], alpha=True)
    image.alpha_mode = "CHANNEL_PACKED"
    image.file_format = "HDR" if is_hdr else "PNG"
    _add_texture(texture_path, image.name)

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
