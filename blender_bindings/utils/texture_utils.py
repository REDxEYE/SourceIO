import os
from hashlib import md5
from typing import Optional

import bpy
import numpy as np

from SourceIO.library.utils.rustlib import save_exr, save_png, encode_exr, encode_png
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

logger = SourceLogMan().get_logger("TextureUtils")


def _get_texture(texture_path: TinyPath, *other_args):
    md_ = md5(texture_path.as_posix().encode("ascii"))
    for key in other_args:
        if key:
            md_.update(key.encode("ascii"))
    key = md_.hexdigest()
    cache = bpy.context.scene.get("texture_name_to_texture", {})
    if key in cache:
        return cache[key]


def _add_texture(texture_path: TinyPath, real_name: str, *other_args):
    md_ = md5(texture_path.as_posix().encode("ascii"))
    for key in other_args:
        if key:
            md_.update(key.encode("ascii"))
    key = md_.hexdigest()
    cache = bpy.context.scene.get("texture_name_to_texture", {})
    cache[key] = real_name
    bpy.context.scene["texture_name_to_texture"] = cache


def check_texture_cache(texture_path: TinyPath) -> Optional[bpy.types.Image]:
    for image_existing in bpy.data.images:
        if (fp := image_existing.get('full_path')) is None:
            continue
        if fp.lower() == texture_path.lower():
            return image_existing

    short_name = _get_texture(texture_path)
    if short_name is not None:
        if short_name + '.png' in bpy.data.images:
            return bpy.data.images[f'{short_name}.png']
        elif short_name + '.hdr' in bpy.data.images:
            return bpy.data.images[f'{short_name}.hdr']
    if bpy.context.scene.TextureCachePath == "":
        return None
    full_path = TinyPath(bpy.context.scene.TextureCachePath) / texture_path.with_suffix(".png")
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
    image['full_path'] = texture_path.lower()
    return image


def create_and_cache_texture(texture_path: TinyPath, dimensions: tuple[int, int], data: np.ndarray, is_hdr: bool = False,
                             invert_y: bool = False):
    _add_texture(texture_path, texture_path.stem)
    if invert_y and not is_hdr:
        data[:, :, 1] = 1 - data[:, :, 1]
    data = data.ravel()

    if bpy.context.scene.TextureCachePath != "":
        save_path = TinyPath(bpy.context.scene.TextureCachePath) / texture_path
        os.makedirs(save_path.parent, exist_ok=True)
        save_path = save_path.with_suffix(".exr" if is_hdr else ".png")
        if is_hdr:
            save_exr(data.ravel(), dimensions[0], dimensions[1], save_path)
        else:
            save_png((data * 255).astype(np.uint8), dimensions[0], dimensions[1], save_path)
        posix_path = save_path.as_posix()
        image = bpy.data.images.load(posix_path)
        image.alpha_mode = 'CHANNEL_PACKED'
        logger.info(f"Save {texture_path.as_posix()!r} texture to disc: {save_path}")
    else:
        if is_hdr:
            image_data = encode_exr(data, dimensions[0], dimensions[1])
        else:
            image_data = encode_png((data * 255).astype(np.uint8), dimensions[0], dimensions[1])
        image = bpy.data.images.new(texture_path.stem, width=1, height=1)
        image.pack(data=image_data, data_len=len(image_data))
        image.source = 'FILE'
        image.alpha_mode = 'CHANNEL_PACKED'
        logger.info(f"Save {texture_path.as_posix()!r} texture to memory")
    image['full_path'] = texture_path.as_posix().lower()

    return image
