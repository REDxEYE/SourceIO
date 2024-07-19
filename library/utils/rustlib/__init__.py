import platform


class UnsupportedSystem(Exception):
    pass


platform_info = platform.uname()
if platform_info.system == "Windows":
    from .windows_x64.rustlib import (Vpk,
                                      decode_index_buffer, decode_vertex_buffer,
                                      zstd_compress, zstd_compress_stream,
                                      zstd_decompress, zstd_decompress_stream,
                                      LZ4ChainDecoder, lz4_compress, lz4_decompress,
                                      load_vtf_texture, save_vtf_texture, save_exr, encode_exr, save_png, encode_png,
                                      decode_texture)
elif platform_info.system == 'Linux':
    from .linux_x64.rustlib import (Vpk,
                                    decode_index_buffer, decode_vertex_buffer,
                                    zstd_compress, zstd_compress_stream,
                                    zstd_decompress, zstd_decompress_stream,
                                    LZ4ChainDecoder, lz4_compress, lz4_decompress,
                                    load_vtf_texture, save_vtf_texture, save_exr, encode_exr, save_png, encode_png,
                                    decode_texture)
elif platform_info.system == 'Darwin':
    if platform.machine().lower() == 'arm64':
        from .macos_arm.rustlib import (Vpk,
                                        decode_index_buffer, decode_vertex_buffer,
                                        zstd_compress, zstd_compress_stream,
                                        zstd_decompress, zstd_decompress_stream,
                                        LZ4ChainDecoder, lz4_compress, lz4_decompress,
                                        load_vtf_texture, save_vtf_texture, save_exr, encode_exr, save_png, encode_png,
                                        decode_texture)
    else:
        from .macos_x64.rustlib import (Vpk,
                                        decode_index_buffer, decode_vertex_buffer,
                                        zstd_compress, zstd_compress_stream,
                                        zstd_decompress, zstd_decompress_stream,
                                        LZ4ChainDecoder, lz4_compress, lz4_decompress,
                                        load_vtf_texture, save_vtf_texture, save_exr, encode_exr, save_png, encode_png,
                                        decode_texture)
else:
    raise UnsupportedSystem(f'System {platform_info} not suppported')
