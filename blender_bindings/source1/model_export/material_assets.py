"""Prop material/texture recipes and in-memory Source 1 asset rendering.

Paths in artifacts are game-root-relative (``materials/...``). A caller owns
writing and deduplicating them; none of these functions write to disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np

from SourceIO.library.utils.kv1 import KV1Block, Severity, dumps, loads
from SourceIO.library.utils.pylib.vtf import (
    ImageFormat, MipFilter, TextureFlags, VTFFile, load_vtf_texture)


@dataclass(frozen=True)
class BinaryArtifact:
    path: str
    data: bytes


@dataclass(frozen=True)
class ImageSourceRecipe:
    image: bpy.types.Image


@dataclass(frozen=True)
class ExternalTextureSourceRecipe:
    path: str


@dataclass(frozen=True)
class VTFRecipe:
    """Generated or copied texture with a game-root-relative artifact path."""
    artifact: BinaryArtifact

    @property
    def reference(self) -> str:
        return self.artifact.path[len('materials/'):-len('.vtf')]


@dataclass(frozen=True)
class MaterialRecipe:
    material: bpy.types.Material
    vmt_path: str
    vmt_text: str
    vtf_artifacts: tuple[BinaryArtifact, ...]


@dataclass(frozen=True)
class MaterialParameters:
    model: bool = True
    translucent: bool = False
    alphatest: bool = False
    alphatest_reference: float = 0.5
    nocull: bool = False
    selfillum: bool = False
    phong: bool = False
    phong_boost: float = 1.0
    phong_exponent: float = 5.0
    envmap: str = ''
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class KVPair:
    key: str
    value: str | tuple[KVPair, ...]
    condition: str | None = None


def material_asset_path(value: str, extension: str) -> str:
    """Return a safe, normalized game-root-relative path under materials/."""
    name = value.strip().replace('\\', '/')
    if name.lower().startswith('materials/'):
        name = name[len('materials/'):]
    if (not name or name.startswith('/') or ':' in name or
            any(part in ('', '.', '..') for part in name.split('/')) or
            any(ord(char) < 32 or char == '"' for char in name)):
        raise ValueError(f'Expected a relative materials path with '
                         f'{extension} extension: {value!r}')
    if name.lower().endswith(extension):
        name = name[:-len(extension)]
    if not name.rsplit('/', 1)[-1]:
        raise ValueError(
            f'Expected a filename before {extension}: {value!r}')
    return f'materials/{name}{extension}'


def _entries(block: KV1Block) -> tuple[KVPair, ...]:
    return tuple(KVPair(entry.key,
                        _entries(entry.value)
                        if isinstance(entry.value, KV1Block) else entry.value,
                        entry.condition) for entry in block.entries)


def parse_kv_entries(text: str, label: str) -> tuple[KVPair, ...]:
    """Parse an ordered KV1 fragment.

    Duplicate keys and nested blocks are retained.
    """
    if not text.strip():
        return ()
    parsed = loads(f'"SourceIOFragment"\n{{\n{text}\n}}', name=label,
                   keep_disabled=True)
    errors = [str(diag) for diag in parsed.diagnostics
              if diag.severity == Severity.ERROR]
    if errors or parsed.includes or len(parsed.entries) != 1:
        detail = '; '.join(errors) or 'unexpected root or include directive'
        raise ValueError(f'Invalid {label}: {detail}')
    body = parsed.entries[0].value
    if not isinstance(body, KV1Block):
        raise ValueError(f'Invalid {label}: expected KeyValues entries')
    return _entries(body)


def _append_pairs(block: KV1Block, pairs: tuple[KVPair, ...]) -> None:
    for pair in pairs:
        if '"' in pair.key or (isinstance(pair.value, str)
                               and '"' in pair.value):
            raise ValueError(
                f'Quotes are not supported in VMT key/value {pair.key!r}')
        value = pair.value
        if isinstance(value, tuple):
            nested = KV1Block()
            _append_pairs(nested, value)
            value = nested
        block.append(pair.key, value, pair.condition)


def _image_pixels(image: bpy.types.Image) -> np.ndarray:
    width, height = map(int, image.size)
    if width < 1 or height < 1:
        raise ValueError(f'Image {image.name!r} has no pixels')
    # Blender's Image.pixels exposes RGBA floats regardless of source file
    # channels.
    data = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(data)
    return data.reshape((height, width, 4))


def texture_pixels(
        source: ImageSourceRecipe | ExternalTextureSourceRecipe) -> np.ndarray:
    """Decode a graph image source for editable texture operators and
    preview.
    """
    if isinstance(source, ImageSourceRecipe):
        return _image_pixels(source.image)
    if not isinstance(source, ExternalTextureSourceRecipe):
        raise ValueError('Expected an image or external texture source')
    path = Path(bpy.path.abspath(source.path))
    if path.suffix.lower() == '.vtf':
        raw = path.read_bytes()
        payload, width, height, is_float = load_vtf_texture(raw)
        pixels = np.frombuffer(payload,
                               dtype=np.float32 if is_float else np.uint8)
        pixels = pixels.reshape((height, width, 4))[::-1].astype(np.float32)
        if not is_float:
            pixels /= 255.0
            vtf = VTFFile()
            vtf.load_bytes(raw)
            if (vtf.get_flag(TextureFlags.SRGB)
                    and not vtf.get_flag(TextureFlags.NORMAL)):
                from SourceIO.library.utils.math_utilities import (
                    srgb_to_linear)
                pixels[..., :3] = srgb_to_linear(pixels[..., :3].copy(), 1)
        return pixels
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        return _image_pixels(image)
    finally:
        bpy.data.images.remove(image)


def _encode_vtf(pixels: np.ndarray, fmt: ImageFormat, mip_filter: MipFilter,
                generate_mipmaps: bool, is_normal: bool,
                linear_space: bool = True) -> bytes:
    """Encode bottom-up RGBA floats as a VTF.

    Blender exposes 8-bit images as their stored, already gamma-encoded
    values, so ``linear_space`` writes the pixels unchanged without the SRGB
    flag, like vtex does. Without it colours are converted from linear to
    sRGB and flagged SRGB. Normal maps are never converted.
    """
    pixels = np.asarray(pixels)
    if (pixels.ndim != 3 or pixels.shape[2] not in (3, 4)
            or min(pixels.shape[:2]) < 1):
        raise ValueError(
            'Texture source must contain an H x W x RGB/RGBA pixel array')
    height, width, channels = pixels.shape
    # Blender and existing graph operations use bottom-up, normalized pixels.
    # Source VTF expects top-down bytes. Never alter the shared input array.
    if not np.issubdtype(pixels.dtype, np.number):
        raise ValueError('Texture pixels must be numeric')
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgb = np.clip(pixels[::-1, :, :3], 0, 1)
    if not is_normal and not linear_space:
        from SourceIO.library.utils.math_utilities import linear_to_srgb
        rgb = linear_to_srgb(rgb, 1)
    rgba[..., :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    alpha = pixels[::-1, :, 3] if channels == 4 else 1.0
    rgba[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    if is_normal:
        flags = TextureFlags.NORMAL
    else:
        flags = 0 if linear_space else TextureFlags.SRGB
    vtf = VTFFile()
    vtf.create_from_data(
        rgba.tobytes(), width, height,
        src_image_format=ImageFormat.RGBA8888, dst_image_format=fmt,
        filter_mode=mip_filter, flags=flags,
        generate_mipmaps=generate_mipmaps, generate_thumbnail=True,
        resize_to_pow2=2)
    vtf.set_flag(TextureFlags.NORMAL, is_normal)
    vtf.set_flag(TextureFlags.SRGB, not is_normal and not linear_space)
    return vtf.to_bytes()


def render_vtf(source: (ImageSourceRecipe | ExternalTextureSourceRecipe
                        | bpy.types.Image | np.ndarray),
               name: str, image_format: str = 'DXT5',
               mip_filter: str = 'CATROM', generate_mipmaps: bool = True,
               is_normal: bool = False,
               linear_space: bool = True) -> VTFRecipe:
    """Resolve an image or external path into VTF bytes without writing
    artifacts.

    External .vtf files are copied bit-for-bit; other files are decoded with
    a temporary Blender Image and re-encoded.
    """
    destination = material_asset_path(name, '.vtf')
    if isinstance(source, ExternalTextureSourceRecipe):
        path = Path(bpy.path.abspath(source.path))
        if path.suffix.lower() == '.vtf':
            data = path.read_bytes()
            if not data.startswith(b'VTF\x00'):
                raise ValueError(f'External texture {path} is not a VTF file')
            return VTFRecipe(BinaryArtifact(destination, data))
        pixels = texture_pixels(source)
    elif isinstance(source, ImageSourceRecipe):
        pixels = texture_pixels(source)
    elif isinstance(source, bpy.types.Image):
        pixels = _image_pixels(source)
    elif isinstance(source, np.ndarray):
        pixels = source
    else:
        raise ValueError('VTF input must be an Image, path recipe or pixel '
                         f'array, got {type(source).__name__}')
    try:
        fmt, filt = ImageFormat[image_format], MipFilter[mip_filter]
    except KeyError as exc:
        raise ValueError(f'Unknown VTF format or mip filter: {exc}') from exc
    return VTFRecipe(BinaryArtifact(destination, _encode_vtf(
        pixels, fmt, filt, generate_mipmaps, is_normal, linear_space)))


def render_material(material: bpy.types.Material, path: str, shader: str,
                    params: MaterialParameters,
                    textures: tuple[tuple[str, VTFRecipe], ...] = (),
                    typed_pairs: tuple[KVPair, ...] = (),
                    raw_pairs: tuple[KVPair, ...] = (),
                    proxies: tuple[KVPair, ...] = ()) -> MaterialRecipe:
    """Render a VMT and collect the VTF artifacts its textures reference."""
    if material is None or not material.name:
        raise ValueError('VMT requires a Blender Material binding')
    if not shader or any(c in shader for c in '\r\n"{}'):
        raise ValueError('VMT shader name must be a single KeyValues name')
    destination = material_asset_path(path, '.vmt')
    entries = KV1Block()
    generated: set[str] = set()
    artifacts_by_path: dict[str, BinaryArtifact] = {}

    def append_generated(pair: KVPair) -> None:
        key = pair.key.lower()
        if key == 'proxies' or key in generated:
            raise ValueError(f'VMT parameter {pair.key!r} conflicts with a '
                             'generated parameter')
        _append_pairs(entries, (pair,))
        generated.add(key)

    for key, texture in textures:
        append_generated(KVPair(key, texture.reference))
        artifact = texture.artifact
        previous = artifacts_by_path.get(artifact.path)
        if previous is not None and previous.data != artifact.data:
            raise ValueError(f'Conflicting textures at {artifact.path}')
        artifacts_by_path[artifact.path] = artifact

    for key, enabled in (('$model', params.model),
                         ('$translucent', params.translucent),
                         ('$alphatest', params.alphatest),
                         ('$nocull', params.nocull),
                         ('$selfillum', params.selfillum),
                         ('$phong', params.phong)):
        if enabled:
            append_generated(KVPair(key, '1'))
    if params.alphatest:
        append_generated(KVPair('$alphatestreference',
                                format(params.alphatest_reference, '.9g')))
    if params.phong:
        append_generated(KVPair('$phongboost',
                                format(params.phong_boost, '.9g')))
        # $phongexponent overrides the per-texel exponent, so it is omitted
        # when the texture is present.
        if not any(key.lower() == '$phongexponenttexture'
                   for key, _ in textures):
            append_generated(KVPair('$phongexponent',
                                    format(params.phong_exponent, '.9g')))
    if params.envmap:
        append_generated(KVPair('$envmap', params.envmap))
    if params.color != (1.0, 1.0, 1.0):
        color = ' '.join(format(x, '.9g') for x in params.color)
        append_generated(KVPair('$color2', f'[{color}]'))
    for pair in typed_pairs:
        append_generated(pair)
    for pair in raw_pairs:
        if pair.key.lower() in generated or pair.key.lower() == 'proxies':
            raise ValueError(f'Raw VMT parameter {pair.key!r} conflicts '
                             'with a generated parameter')
    _append_pairs(entries, raw_pairs)
    if proxies:
        proxy_block = KV1Block()
        _append_pairs(proxy_block, proxies)
        entries.append('Proxies', proxy_block)
    root = KV1Block()
    root.append(shader, entries)
    return MaterialRecipe(material, destination, dumps(root),
                          tuple(artifacts_by_path.values()))
