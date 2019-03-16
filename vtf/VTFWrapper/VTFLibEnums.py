from ctypes import c_uint32


class EnumerationType(type(c_uint32)):
    def __new__(metacls, name, bases, dict):
        if not "_members_" in dict:
            _members_ = {}
            for key, value in dict.items():
                if not key.startswith("_"):
                    _members_[key] = value

            dict["_members_"] = _members_
        else:
            _members_ = dict["_members_"]

        dict["_reverse_map_"] = {v: k for k, v in _members_.items()}
        cls = type(c_uint32).__new__(metacls, name, bases, dict)
        for key, value in cls._members_.items():
            globals()[key] = value
        return cls

    def __repr__(self):
        return "<Enumeration %s>" % self.__name__


class CEnumeration(c_uint32, metaclass=EnumerationType):
    _members_ = {}

    def __repr__(self):
        value = self.value
        return "<{} {}>".format(self.__class__.__name__, self.name)

    @property
    def name(self):
        return self._reverse_map_.get(self.value, '(unknown)')

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other

        return isinstance(self, type(other)) and self.value == other.value

    @classmethod
    def from_param(self, value):

        # print('lel',value)
        self.value = value


class CFlag(c_uint32, metaclass=EnumerationType):
    _members_ = {}

    def __repr__(self):
        value = self.value
        keys = []
        for val, name in self._reverse_map_.items():
            if value & val:
                keys.append(name)
        return "<{} {}>".format(self.__class__.__name__, " | ".join(keys))

    def get_flag(self, flag):
        return bool(self.value & flag)

    @property
    def name(self):
        return self._reverse_map_.get(self.value, '(unknown)')

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other

        return isinstance(self, type(other)) and self.value == other.value

    @classmethod
    def from_param(self, value):

        # print('lel',value)
        self.value = value


class Option(CEnumeration):
    OptionDXTQuality = 0

    OptionLuminanceWeightR = 1
    OptionLuminanceWeightG = 2
    OptionLuminanceWeightB = 3

    OptionBlueScreenMaskR = 4
    OptionBlueScreenMaskG = 5
    OptionBlueScreenMaskB = 6

    OptionBlueScreenClearR = 7
    OptionBlueScreenClearG = 8
    OptionBlueScreenClearB = 9

    OptionFP16HDRKey = 10
    OptionFP16HDRShift = 11
    OptionFP16HDRGamma = 12

    OptionUnsharpenRadius = 13
    OptionUnsharpenAmount = 14
    OptionUnsharpenThreshold = 15

    OptionXSharpenStrength = 16
    OptionXSharpenThreshold = 17

    OptionVMTParseMode = 18


class ImageFormat(CEnumeration):
    ImageFormatRGBA8888 = 0
    ImageFormatABGR8888 = 1
    ImageFormatRGB888 = 2
    ImageFormatBGR888 = 3
    ImageFormatRGB565 = 4
    ImageFormatI8 = 5
    ImageFormatIA88 = 6
    ImageFormatP8 = 7
    ImageFormatA8 = 8
    ImageFormatRGB888BlueScreen = 9
    ImageFormatBGR888BlueScreen = 10
    ImageFormatARGB8888 = 11
    ImageFormatBGRA8888 = 12
    ImageFormatDXT1 = 13
    ImageFormatDXT3 = 14
    ImageFormatDXT5 = 15
    ImageFormatBGRX8888 = 16
    ImageFormatBGR565 = 17
    ImageFormatBGRX5551 = 18
    ImageFormatBGRA4444 = 19
    ImageFormatDXT1OneBitAlpha = 20
    ImageFormatBGRA5551 = 21
    ImageFormatUV88 = 22
    ImageFormatUVWQ8888 = 23
    ImageFormatRGBA16161616F = 24
    ImageFormatRGBA16161616 = 25
    ImageFormatUVLX8888 = 26
    ImageFormatI32F = 27
    ImageFormatRGB323232F = 28
    ImageFormatRGBA32323232F = 29
    ImageFormatCount = 30
    ImageFormatNone = -1


class ImageFlag(CFlag):
    ImageFlagNone = 0x00000000
    ImageFlagPointSample = 0x00000001
    ImageFlagTrilinear = 0x00000002
    ImageFlagClampS = 0x00000004
    ImageFlagClampT = 0x00000008
    ImageFlagAnisotropic = 0x00000010
    ImageFlagHintDXT5 = 0x00000020
    ImageFlagSRGB = 0x00000040
    ImageFlagNormal = 0x00000080
    ImageFlagNoMIP = 0x00000100
    ImageFlagNoLOD = 0x00000200
    ImageFlagMinMIP = 0x00000400
    ImageFlagProcedural = 0x00000800
    ImageFlagOneBitAlpha = 0x00001000
    ImageFlagEightBitAlpha = 0x00002000
    ImageFlagEnviromentMap = 0x00004000
    ImageFlagRenderTarget = 0x00008000
    ImageFlagDepthRenderTarget = 0x00010000
    ImageFlagNoDebugOverride = 0x00020000
    ImageFlagSingleCopy = 0x00040000
    ImageFlagUnused0 = 0x00080000
    ImageFlagUnused1 = 0x00100000
    ImageFlagUnused2 = 0x00200000
    ImageFlagUnused3 = 0x00400000
    ImageFlagNoDepthBuffer = 0x00800000
    ImageFlagUnused4 = 0x01000000
    ImageFlagClampU = 0x02000000
    ImageFlagVertexTexture = 0x04000000
    ImageFlagSSBump = 0x08000000
    ImageFlagUnused5 = 0x10000000
    ImageFlagBorder = 0x20000000
    ImageFlagCount = 30


class CubemapFace(CEnumeration):
    CubemapFaceRight = 0
    CubemapFaceLeft = 1
    CubemapFaceBack = 2
    CubemapFaceFront = 3
    CubemapFaceUp = 4
    CubemapFaceDown = 5
    CubemapFaceSphereMap = 6
    CubemapFaceCount = 7


class MipmapFilter(CEnumeration):
    MipmapFilterPoint = 0
    MipmapFilterBox = 1
    MipmapFilterTriangle = 2
    MipmapFilterQuadratic = 3
    MipmapFilterCubic = 4
    MipmapFilterCatrom = 5
    MipmapFilterMitchell = 6
    MipmapFilterGaussian = 7
    MipmapFilterSinC = 8
    MipmapFilterBessel = 9
    MipmapFilterHanning = 10
    MipmapFilterHamming = 11
    MipmapFilterBlackman = 12
    MipmapFilterKaiser = 13
    MipmapFilterCount = 14


class SharpenFilter(CEnumeration):
    SharpenFilterNone = 0
    SharpenFilterNegative = 1
    SharpenFilterLighter = 2
    SharpenFilterDarker = 3
    SharpenFilterContrastMore = 4
    SharpenFilterContrastLess = 5
    SharpenFilterSmoothen = 6
    SharpenFilterSharpenSoft = 7
    SharpenFilterSharpenMeium = 8
    SharpenFilterSharpenStrong = 9
    SharpenFilterFindEdges = 10
    SharpenFilterContour = 11
    SharpenFilterEdgeDetect = 12
    SharpenFilterEdgeDetectSoft = 13
    SharpenFilterEmboss = 14
    SharpenFilterMeanRemoval = 15
    SharpenFilterUnsharp = 16
    SharpenFilterXSharpen = 17
    SharpenFilterWarpSharp = 18
    SharpenFilterCount = 19


class DXTQuality(CEnumeration):
    DXTQualityLow = 0
    DXTQualityMedium = 1
    DXTQualityHigh = 2
    DXTQualityHighest = 3
    DXTQualityCount = 4


class KernelFilter(CEnumeration):
    KernelFilter4x = 0
    KernelFilter3x3 = 1
    KernelFilter5x5 = 2
    KernelFilter7x7 = 3
    KernelFilter9x9 = 4
    KernelFilterDuDv = 5
    KernelFilterCount = 6


class HeightConversionMethod(CEnumeration):
    HeightConversionMethodAlpha = 0
    HeightConversionMethodAverageRGB = 1
    HeightConversionMethodBiasedRGB = 2
    HeightConversionMethodRed = 3
    HeightConversionMethodGreed = 4
    HeightConversionMethodBlue = 5
    HeightConversionMethodMaxRGB = 6
    HeightConversionMethodColorSspace = 7
    # HeightConversionMethodNormalize = auto()
    HeightConversionMethodCount = 8


class NormalAlphaResult(CEnumeration):
    NormalAlphaResultNoChange = 0
    NormalAlphaResultHeight = 1
    NormalAlphaResultBlack = 2
    NormalAlphaResultWhite = 3
    NormalAlphaResultCount = 4


class ResizeMethod(CEnumeration):
    ResizeMethodNearestPowerTwo = 0
    ResizeMethodBiggestPowerTwo = 1
    ResizeMethodSmallestPowerTwo = 2
    ResizeMethodSet = 3
    ResizeMethodCount = 4


class ResourceFlag(CEnumeration):
    ResourceFlagNoDataChunk = 0x02
    ResourceFlagCount = 1


class ResourceType(CEnumeration):
    ResourceTypeLowResolutionImage = 0x01
    ResourceTypeImage = 0x30
    ResourceTypeSheet = 0x10
    ResourceTypeCRC = ord('C') | (
        ord('R') << 8) | (
        ord('C') << 24) | (
            ResourceFlag.ResourceFlagNoDataChunk << 32)
    ResourceTypeLODControl = ord('L') | (ord('O') << 8) | (ord('D') << 24) | (
        ResourceFlag.ResourceFlagNoDataChunk << 32)
    ResourceTypeTextureSettingsEx = ord('T') | (ord('S') << 8) | (ord('O') << 24) | (
        ResourceFlag.ResourceFlagNoDataChunk << 32)
    ResourceTypeKeyValueData = ord('K') | (ord('V') << 8) | (ord('D') << 24)


class Proc(CEnumeration):
    ProcReadClose = 0
    ProcReadOpen = 1
    ProcReadRead = 2
    ProcReadSeek = 3
    ProcReadTell = 4
    ProcReadSize = 5
    ProcWriteClose = 6
    ProcWriteOpen = 7
    ProcWriteWrite = 8
    ProcWriteSeek = 9
    ProcWriteSize = 10
    ProcWriteTell = 11


class SeekMode(CEnumeration):
    Begin = 0
    Current = 1
    End = 2


if __name__ == '__main__':
    a = Proc(5)
    print(a)
    a.value = a.ProcWriteTell
    print(a)
    print(a.ProcWriteTell)
