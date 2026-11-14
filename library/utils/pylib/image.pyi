from __future__ import annotations

from typing import Any

"""
Image encoding (PNG, EXR) and block-compressed texture decoding.
"""
def decode_texture(image_data: Any, width: Any, height: Any, format: Any) -> Any:
    """
    Decode block-compressed texture data and return the pixels as bytes.
    
    image_data is any bytes-like object holding at least ceil(width/4)*ceil(height/4)
    4x4 blocks (extra trailing bytes are ignored). width must be in 1..536870911,
    height in 1..2147483647.
    
    Formats (block size) and output layout (tightly packed rows, width*height pixels):
      'BC1'/'DXT1' (8), 'BC2'/'DXT3' (16), 'BC3'/'DXT5' (16), 'BC7' (16),
      'ETC1' (8), 'ETC2' (8, RGB, A = 255): RGBA8 (uint8 x4).
      'ETC2_A1'/'ETC2_RGB_A1' (8): RGBA8 with 1-bit alpha (A is 0 or 255).
      'ETC2_EAC'/'ETC2_RGBA' (16): RGBA8 with 8-bit alpha.
      'EAC_R11' (8): RGBA8 with the value in R, G = B = 0, A = 255 (11-bit value
          reduced to 8 bits).
      'EAC_RG11' (16): RGBA8 with the values in R and G, B = 0, A = 255.
      'EAC_R11S' (8), 'EAC_RG11S' (16): as EAC_R11/EAC_RG11 for signed data, stored
          biased as unsigned bytes (0 ~ -1.0, 127 ~ 0.0, 254 ~ +1.0).
      'BC4'/'ATI1N' (8): R8 unsigned (uint8 x1).
      'BC4S'/'ATI1N_SNORM' (8): R8 signed (int8 two's-complement x1, range -127..127).
      'BC5'/'ATI2N' (16): RG8 unsigned (uint8 x2).
      'BC5S'/'ATI2N_SNORM' (16): RG8 signed (int8 two's-complement x2, range -127..127).
      'BC6H'/'BC6H_UF16' (16): RGB unsigned half floats (float16 x3).
      'BC6H_SF16' (16): RGB signed half floats (float16 x3).
    Raises ValueError for an unknown format, bad dimensions or too little data.
    """
    ...

def encode_exr(image_data: Any, width: Any, height: Any, channels: Any) -> Any:
    """
    Encode float image data as EXR (stored as half floats) and return it as bytes.
    
    image_data is any bytes-like object holding width*height*channels native-endian
    float32 values, interleaved; channels is 1 (Y), 3 (RGB) or 4 (RGBA).
    Raises ValueError on bad dimensions/length, OverflowError on oversized
    dimensions, RuntimeError if encoding fails.
    """
    ...

def encode_png(image_data: Any, width: Any, height: Any, channels: Any) -> Any:
    """
    Encode 8-bit image data as PNG and return the file contents as bytes.
    
    image_data is any bytes-like object of width*height*channels bytes (rows
    tightly packed, top to bottom); channels is 1 (Y), 2 (YA), 3 (RGB) or 4 (RGBA).
    Raises ValueError on bad dimensions/length, OverflowError if the image exceeds
    the encoder's int limits, MemoryError/RuntimeError if encoding fails.
    """
    ...

def save_exr(image_data: Any, width: Any, height: Any, channels: Any, file_path: Any) -> Any:
    """
    Encode float image data as EXR (stored as half floats) and write it to file_path.
    
    image_data is any bytes-like object holding width*height*channels native-endian
    float32 values, interleaved; channels is 1 (Y), 3 (RGB) or 4 (RGBA).
    file_path must be a str. Returns None.
    Raises ValueError on bad dimensions/length, OverflowError on oversized
    dimensions, OSError if the file cannot be written (a partially written file
    is removed), RuntimeError if encoding fails.
    """
    ...

def save_png(image_data: Any, width: Any, height: Any, channels: Any, file_path: Any) -> Any:
    """
    Encode 8-bit image data as PNG and write it to file_path.
    
    image_data is any bytes-like object of width*height*channels bytes (rows
    tightly packed, top to bottom); channels is 1 (Y), 2 (YA), 3 (RGB) or 4 (RGBA).
    file_path must be a str. Returns None.
    Raises ValueError on bad dimensions/length, OverflowError if the image exceeds
    the encoder's int limits, OSError if the file cannot be written (a partially
    written file is removed), RuntimeError if encoding fails.
    """
    ...

