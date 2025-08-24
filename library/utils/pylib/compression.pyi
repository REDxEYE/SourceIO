from __future__ import annotations

from typing import Any

"""
SourceIO compression module
"""
class LZ4ChainDecoder:
    """
    Chain decoder for LZ4 blocks. Call decompress(src, block_size) to get bytes.
    """
    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def decompress(self: Any, src: Any, block_size: Any) -> Any:
        """
        Decompress one chain block from 'src' into an output buffer of size 'block_size' and return bytes.
        """
        ...

class LZ4ChainEncoder:
    """
    Chain encoder for raw LZ4 blocks with dependent-block dictionary.
    """
    block_size: Any
    extra_blocks: Any

    def __init__(self, block_size, extra_blocks) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def compress(self: Any, src: Any, accel: Any = ...) -> Any:
        """
        Compress one uncompressed block 'src' using the rolling dictionary.
        Returns compressed bytes. 'accel' is LZ4 acceleration (1=faster&denser, higher=faster&weaker).
        """
        ...

def lz4_compress(data: Any) -> Any:
    """
    Compress data using LZ4 compression.
    """
    ...

def lz4_decompress(data: Any, decompressed_size: Any) -> Any:
    """
    Decompress data using LZ4 compression.
    """
    ...

def lz4_decompress_continue(context: Any, data: Any, decompressed_size: Any) -> Any:
    """
    Continue decompressing data using LZ4 compression.
    """
    ...

def zstd_compress(data: Any, compression_level: Any = ...) -> Any:
    """
    Compress data using zstd compression.
    The compression level can be between 1 and 22, default is 3.
    """
    ...

def zstd_compress_stream(data: Any, compression_level: Any = ...) -> Any:
    """
    Compress data using zstd compression in a streaming manner.
    The compression level can be between 1 and 22, default is 3.
    """
    ...

def zstd_decompress(data: Any, decompressed_size: Any) -> Any:
    """
    Decompress data using zstd compression.
    """
    ...

def zstd_decompress_stream(data: Any) -> Any:
    """
    Decompress data using zstd compression in a streaming manner.
    """
    ...

