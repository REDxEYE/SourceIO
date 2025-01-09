from enum import IntEnum, IntFlag


class VTexFlags(IntFlag):
    SUGGEST_CLAMPS = 0x00000001
    SUGGEST_CLAMPT = 0x00000002
    SUGGEST_CLAMPU = 0x00000004
    NO_LOD = 0x00000008
    CUBE_TEXTURE = 0x00000010
    VOLUME_TEXTURE = 0x00000020
    TEXTURE_ARRAY = 0x00000040


class VTexFormat(IntEnum):
    UNKNOWN = 0
    DXT1 = 1
    DXT5 = 2
    I8 = 3
    RGBA8888 = 4
    R16 = 5
    RG1616 = 6
    RGBA16161616 = 7
    R16F = 8
    RG1616F = 9
    RGBA16161616F = 10
    R32F = 11
    RG3232F = 12
    RGB323232F = 13
    RGBA32323232F = 14
    JPEG_RGBA8888 = 15
    PNG_RGBA8888 = 16
    JPEG_DXT5 = 17
    PNG_DXT5 = 18
    BC6H = 19
    BC7 = 20
    ATI2N = 21
    IA88 = 22
    ETC2 = 23
    ETC2_EAC = 24
    R11_EAC = 25
    RG11_EAC = 26
    ATI1N = 27
    BGRA8888 = 28

    @staticmethod
    def block_size(fmt):
        return {
            VTexFormat.DXT1: 8,
            VTexFormat.DXT5: 16,
            VTexFormat.RGBA8888: 4,
            VTexFormat.R16: 2,
            VTexFormat.I8: 1,
            VTexFormat.RG1616: 4,
            VTexFormat.RGBA16161616: 8,
            VTexFormat.R16F: 2,
            VTexFormat.RG1616F: 4,
            VTexFormat.RGBA16161616F: 8,
            VTexFormat.R32F: 4,
            VTexFormat.RG3232F: 8,
            VTexFormat.RGB323232F: 12,
            VTexFormat.RGBA32323232F: 16,
            VTexFormat.BC6H: 16,
            VTexFormat.BC7: 16,
            VTexFormat.IA88: 2,
            VTexFormat.ETC2: 8,
            VTexFormat.ETC2_EAC: 16,
            VTexFormat.BGRA8888: 4,
            VTexFormat.ATI1N: 8,
            VTexFormat.ATI2N: 16,
        }[fmt]


class VTexExtraData(IntEnum):
    UNKNOWN = 0
    FALLBACK_BITS = 1
    SHEET = 2
    FILL_TO_POWER_OF_TWO = 3
    COMPRESSED_MIP_SIZE = 4
    CUBEMAP_RADIANCE_SH = 5
