import logging
from collections import defaultdict

import numpy as np
import numpy.typing as npt

from SourceIO.library.source2.blocks.kv3_block import KVBlock
from SourceIO.library.source2.keyvalues3.types import AnyKVType


class MorphBlock(KVBlock):

    def __init__(self, data: dict[str, AnyKVType] = None):
        super().__init__(data or {})
        self._morph_datas: dict[int, dict[str, npt.NDArray[np.float32]]] = defaultdict(dict)
        self._morph_name_map: dict[str, AnyKVType] = {}
        self._vmorf_texture = None

    @staticmethod
    def _struct_name():
        return 'MorphSetData_t'

    @property
    def lookup_type(self):
        return self.get("m_nLookupType", "LOOKUP_TYPE_VERTEX_ID")

    @property
    def encoding_type(self):
        return self.get("m_nEncodingType", "ENCODING_TYPE_OBJECT_SPACE")

    @property
    def bundles(self) -> list[str]:
        return self['m_bundleTypes']

    def get_bundle_id(self, bundle_name):
        if bundle_name in self.bundles:
            return self.bundles.index(bundle_name)
        return None

    def get_morph_data(self, flex_name: str, bundle_id: int, texture):
        bundle_data = self._morph_datas[bundle_id]
        cached = bundle_data.get(flex_name)
        if cached is not None:
            return cached

        assert self.lookup_type == 'LOOKUP_TYPE_VERTEX_ID'
        assert self.encoding_type == 'ENCODING_TYPE_OBJECT_SPACE'

        t_width, t_height = texture.shape[:2]
        width = self['m_nWidth']
        height = self['m_nHeight']

        name_map = self._morph_name_map
        if name_map is None:
            name_map = {md['m_name']: md for md in self['m_morphDatas']}
            self._morph_name_map = name_map

        morph_data = name_map.get(flex_name)
        if morph_data is None:
            logging.error(f'Failed to find morph data for {flex_name!r} flex')
            return None

        out = bundle_data[flex_name] = np.zeros((height, width, 4), dtype=np.float32)

        for n, rect in enumerate(morph_data['m_morphRectDatas']):
            bundle = rect['m_bundleDatas'][bundle_id]

            rw = round(rect['m_flUWidthSrc'] * t_width)
            rh = round(rect['m_flVHeightSrc'] * t_height)
            dx = rect['m_nXLeftDst']
            dy = rect['m_nYTopDst']
            ru = round(bundle['m_flULeftSrc'] * t_width)
            rv = round(bundle['m_flVTopSrc'] * t_height)
            src = texture[rv:rv + rh, ru:ru + rw, :4].astype(np.float32, copy=False)

            dst = out[dy:(dy + rh), dx:(dx + rw), :4]

            np.multiply(src, np.asarray(bundle['m_ranges'], dtype=np.float32), out=dst, casting='unsafe')
            np.add(dst, np.asarray(bundle['m_offsets'], dtype=np.float32), out=dst, casting='unsafe')

        return out
