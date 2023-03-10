from pathlib import Path
from typing import Tuple, Optional

import bpy
import numpy as np

from ...logger import SLoggingManager

logger = SLoggingManager().get_logger("TextureUtils")


def check_texture_cache(name: str) -> Optional[bpy.types.Image]:
    if name + '.png' in bpy.data.images:
        return bpy.data.images[f'{name}.png']
    elif name + '.hdr' in bpy.data.images:
        return bpy.data.images[f'{name}.hdr']

    if bpy.context.scene.TextureCachePath == "":
        return None
    full_path = (Path(bpy.context.scene.TextureCachePath) / name).with_suffix(".png")
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
    logger.info(f"Loaded {name!r} texture from disc")
    image.alpha_mode = "CHANNEL_PACKED"
    image.name = name
    return image


def create_and_cache_texture(name: str, dimensions: Tuple[int, int], data: np.ndarray, is_hdr: bool = False):
    image = bpy.data.images.new(name, width=dimensions[0], height=dimensions[1], alpha=True)
    image.alpha_mode = "CHANNEL_PACKED"
    image.file_format = "HDR" if is_hdr else "PNG"
    image.pixels.foreach_set(data.ravel())
    if bpy.context.scene.TextureCachePath != "":
        save_path = (Path(bpy.context.scene.TextureCachePath) / name).with_suffix(".hdr" if is_hdr else ".png")
        image.save(filepath=save_path.as_posix())
        logger.info(f"Save {name!r} texture to disc: {save_path}")
        image.filepath = save_path.as_posix()
    else:
        image.pack()
        logger.info(f"Save {name!r} texture to memory")
    return image
