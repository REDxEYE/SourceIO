from .data_block import DATA
import numpy as np


class MRPH(DATA):

    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.flex_data = {}

    def read_morphs(self):
        if self.data['m_pTextureAtlas'] in self._valve_file.available_resources:
            morph_atlas = self._valve_file.parse_new(self._valve_file.available_resources[self.data['m_pTextureAtlas']])
            morph_atlas.read_block_info()
            morph_atlas_data = morph_atlas.get_data_block(block_name="DATA")[0]
            morph_atlas_data.read_image(False)
            raw_flex_data = np.array(list(morph_atlas_data.image_data), dtype=np.uint8)
            width = self.data['m_nWidth']
            height = self.data['m_nHeight']
            encoding_type = self.data['m_nEncodingType']
            lookup_type = self.data['m_nLookupType']
            assert lookup_type == 'LOOKUP_TYPE_VERTEX_ID', "Unknown lookup type"
            assert encoding_type == 'ENCODING_TYPE_OBJECT_SPACE', "Unknown encoding type"
            bundle_types = self.data['m_bundleTypes']
            raw_flex_data = raw_flex_data.reshape((morph_atlas_data.width, morph_atlas_data.height, 4))

            for morph_datas in self.data['m_morphDatas']:
                self.flex_data[morph_datas['m_name']] = np.zeros((len(bundle_types),
                                                                  height, width,
                                                                  4),
                                                                 dtype=np.float32)
                for n, rect in enumerate(morph_datas['m_morphRectDatas']):
                    rect_width = round(rect['m_flUWidthSrc'] * morph_atlas_data.width)
                    rect_height = round(rect['m_flVHeightSrc'] * morph_atlas_data.height)
                    dst_x = rect['m_nXLeftDst']
                    dst_y = rect['m_nYTopDst']
                    for c, bundle in enumerate(rect['m_bundleDatas']):
                        rect_u = round(bundle['m_flULeftSrc'] * morph_atlas_data.width)
                        rect_v = round(bundle['m_flVTopSrc'] * morph_atlas_data.height)
                        morph_data_rect = raw_flex_data[rect_v:rect_v + rect_height, rect_u:rect_u + rect_width, :]
                        vec_offset = np.array(bundle['m_offsets'].as_list)
                        vec_range = np.array(bundle['m_ranges'].as_list)

                        transformed_data = np.divide(morph_data_rect, 255)
                        transformed_data = np.multiply(transformed_data, vec_range)
                        transformed_data = np.add(transformed_data, vec_offset)
                        transformed_data = np.round(transformed_data, 6)

                        self.flex_data[morph_datas['m_name']][c, dst_y: dst_y + rect_height, dst_x: dst_x + rect_width,
                        :] = transformed_data
            for k, v in self.flex_data.items():
                self.flex_data[k] = v.reshape((len(bundle_types), width * height, 4))
            return True
        else:
            return False
