"""Texture decoding keeps stored byte values (not Blender's linear floats)."""

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import bpy
import numpy as np

from SourceIO.blender_bindings.source1.model_export.material_assets import (
    ExternalTextureSourceRecipe, ImageSourceRecipe, render_vtf, texture_pixels)
from SourceIO.library.utils.math_utilities import linear_to_srgb
from SourceIO.library.utils.pylib.vtf import TextureFlags, VTFFile, load_vtf_texture


def write_png16(path: Path, rgba: tuple[int, int, int, int], size: int = 4):
    def chunk(kind: bytes, body: bytes) -> bytes:
        return (struct.pack('>I', len(body)) + kind + body
                + struct.pack('>I', zlib.crc32(kind + body)))

    row = b'\x00' + struct.pack('>4H', *rgba) * size
    path.write_bytes(
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>2I5B', size, size, 16, 6, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(row * size))
        + chunk(b'IEND', b''))


class TextureColorTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_external_16bit_png_keeps_stored_values(self):
        path = self.root / 'grey16.png'
        write_png16(path, (32768, 32768, 65535, 65535))
        before = len(bpy.data.images)
        source = ExternalTextureSourceRecipe(str(path))
        np.testing.assert_allclose(
            texture_pixels(source)[0, 0], [0.5, 0.5, 1, 1], atol=1 / 255)
        data = render_vtf(source, 'grey16', 'RGBA8888',
                          generate_mipmaps=False).artifact.data
        payload, *_ = load_vtf_texture(data)
        self.assertLessEqual(abs(int(payload[0]) - 128), 1)
        self.assertEqual(len(bpy.data.images), before)

    def test_float_srgb_image_is_encoded_without_changing_settings(self):
        image = bpy.data.images.new('float_srgb', 2, 2, float_buffer=True)
        self.addCleanup(bpy.data.images.remove, image)
        image.colorspace_settings.name = 'sRGB'
        image.pixels[:] = [0.2, 0.2, 0.2, 0.5] * 4
        self.assertEqual(image.colorspace_settings.name, 'sRGB')
        pixels = texture_pixels(ImageSourceRecipe(image))
        np.testing.assert_allclose(
            pixels[0, 0], [*linear_to_srgb(np.full(3, 0.2, np.float32), 1),
                           0.5], atol=1e-4)
        self.assertEqual(image.colorspace_settings.name, 'sRGB')
        self.assertFalse(image.colorspace_settings.is_data)

    def test_srgb_vtf_keeps_bytes_through_the_graph(self):
        pixels = np.full((4, 4, 4), 187 / 255, np.float32)
        vtf = VTFFile()
        vtf.load_bytes(render_vtf(pixels, 'src', 'RGBA8888',
                                  generate_mipmaps=False).artifact.data)
        vtf.set_flag(TextureFlags.SRGB, True)
        path = self.root / 'srgb.vtf'
        path.write_bytes(vtf.to_bytes())
        source = ExternalTextureSourceRecipe(str(path))
        np.testing.assert_allclose(texture_pixels(source), pixels, atol=1e-6)
        data = render_vtf(texture_pixels(source), 'out', 'RGBA8888',
                          generate_mipmaps=False).artifact.data
        payload, *_ = load_vtf_texture(data)
        self.assertEqual(int(payload[0]), 187)


    def test_npot_vtf_uses_nearest_power_of_two(self):
        for size, expected in ((47, 32), (49, 64)):
            with self.subTest(size=size):
                pixels = np.full((size, size, 4), 0.5, np.float32)
                data = render_vtf(pixels, 'npot', 'RGBA8888',
                                  generate_mipmaps=False).artifact.data
                _, width, height, _ = load_vtf_texture(data)
                self.assertEqual((width, height), (expected, expected))

    def test_byte_encoding_rounds_half_steps(self):
        pixels = np.full((4, 4, 4), 0.5, np.float32)
        data = render_vtf(pixels, 'half', 'RGBA8888',
                          generate_mipmaps=False).artifact.data
        payload, *_ = load_vtf_texture(data)
        self.assertEqual(tuple(payload[:4]), (128, 128, 128, 128))

if __name__ == '__main__':
    unittest.main()
