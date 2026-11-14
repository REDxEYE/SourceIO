from __future__ import annotations

from typing import Any

"""
SourceIO compression module
"""
class LZ4ChainDecoder:
    """
    Chain decoder for dependent LZ4 blocks with a rolling 64 KiB dictionary.
    block_size is rounded up to a multiple of 1024 (minimum 1024).
    """
    block_size: Any
    extra_blocks: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def decompress(self: Any, src: Any, block_size: Any) -> Any:
        """
        Decompress one dependent LZ4 block from 'src' into a bytes object of length 'block_size'.
        Raises ValueError if the data is corrupt or a size exceeds INT_MAX or the decoder capacity.
        """
        ...

class LZ4ChainEncoder:
    """
    Chain encoder for raw LZ4 blocks with a dependent-block dictionary; output is
    decodable by LZ4ChainDecoder with the same parameters.
    block_size is rounded up to a multiple of 1024 (minimum 1024).
    """
    block_size: Any
    extra_blocks: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def compress(self: Any, src: Any, accel: Any = ...) -> Any:
        """
        Compress one block 'src' as a dependent LZ4 block using the rolling 64 KiB dictionary.
        Returns compressed bytes. 'accel' is the LZ4 acceleration (values <= 0 mean 1;
        higher is faster but compresses less). Raises ValueError if 'src' exceeds the
        encoder capacity.
        """
        ...

def lz4_compress(data: Any) -> Any:
    """
    Compress data into a raw LZ4 block. Raises ValueError if data exceeds LZ4_MAX_INPUT_SIZE.
    """
    ...

def lz4_decompress(data: Any, decompressed_size: Any) -> Any:
    """
    Decompress a raw LZ4 block that must decode to exactly 'decompressed_size' bytes.
    Raises ValueError on corrupt data, size mismatch, or sizes above INT_MAX.
    """
    ...

def zstd_compress(data: Any, compression_level: Any = ...) -> Any:
    """
    Compress data into a single zstd frame.
    compression_level ranges from zstd's minimum (negative, fastest) to its maximum
    (22); 0 selects the zstd default. Raises ValueError for out-of-range levels.
    """
    ...

def zstd_compress_stream(data: Any, compression_level: Any = ...) -> Any:
    """
    Compress data using the zstd streaming API.
    compression_level ranges from zstd's minimum (negative, fastest) to its maximum
    (22); 0 selects the zstd default. Raises ValueError for out-of-range levels.
    """
    ...

def zstd_decompress(data: Any, decompressed_size: Any) -> Any:
    """
    Decompress a zstd frame into at most 'decompressed_size' bytes and return them.
    Raises ValueError on corrupt data or a negative size.
    """
    ...

def zstd_decompress_stream(data: Any) -> Any:
    """
    Decompress one or more concatenated zstd frames using the streaming API.
    Raises ValueError on corrupt or truncated data.
    """
    ...

