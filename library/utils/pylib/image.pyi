from __future__ import annotations

from typing import Any

"""
SourceIO image module
"""
def decode_texture(image_data: Any, width: Any, height: Any, format: Any) -> Any:
    """
    Decode texture data block compressed format.
    """
    ...

def encode_exr(image_data: Any, width: Any, height: Any, channels: Any) -> Any:
    """
    Encode image data to EXR format and return as bytes.
    """
    ...

def encode_png(image_data: Any, width: Any, height: Any, channels: Any) -> Any:
    """
    Encode image data to PNG format and return as bytes.
    """
    ...

def save_exr(image_data: Any, width: Any, height: Any, channels: Any, file_path: Any) -> Any:
    """
    Save image data as EXR file.
    """
    ...

def save_png(image_data: Any, width: Any, height: Any, channels: Any, file_path: Any) -> Any:
    """
    Save image data as PNG file.
    """
    ...

