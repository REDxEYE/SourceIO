import struct

from .data_block import DATA
import numpy as np
from ...source1.mdl.flex_expressions import *


class MRPH(DATA):

    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.flex_data = {}

    def read_morphs(self):
        if self.data['m_pTextureAtlas'] not in self._valve_file.available_resources:
            return False
        morph_atlas = self._valve_file.parse_new(self._valve_file.available_resources[self.data['m_pTextureAtlas']])
        morph_atlas.read_block_info()
        morph_atlas_data = morph_atlas.get_data_block(block_name="DATA")[0]
        morph_atlas_data.read_image(False)
        raw_flex_data = np.frombuffer(morph_atlas_data.image_data, dtype=np.uint8)
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
                    vec_offset = bundle['m_offsets']
                    vec_range = bundle['m_ranges']

                    transformed_data = np.divide(morph_data_rect, 255)
                    transformed_data = np.multiply(transformed_data, vec_range)
                    transformed_data = np.add(transformed_data, vec_offset)
                    transformed_data = transformed_data

                    self.flex_data[morph_datas['m_name']][c, dst_y: dst_y + rect_height, dst_x: dst_x + rect_width,
                    :] = transformed_data
        for k, v in self.flex_data.items():
            self.flex_data[k] = v.reshape((len(bundle_types), width * height, 4))
        return True

    def rebuild_flex_expressions(self):
        flex_rules = {}

        def get_flex_desc(index):
            return self.data['m_FlexDesc'][index]['m_szFacs']

        def get_flex_cnt(index):
            return self.data['m_FlexControllers'][index]['m_szName']

        for rule in self.data['m_FlexRules']:
            stack = []
            # try:
            for op in rule['m_FlexOps']:
                flex_op = op['m_OpCode']
                index = op['m_Data']
                value = struct.unpack('f', struct.pack('i', index))[0]
                if flex_op == "FLEX_OP_ADD":
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Add(left, right))
                elif flex_op == "FLEX_OP_COMBO":
                    count = index
                    values = [stack.pop(-1) for _ in range(count)]
                    combo = Combo(*values)
                    stack.append(combo)
                elif flex_op == "FLEX_OP_CONST":
                    stack.append(Value(value))
                elif flex_op == "FLEX_OP_DIV":
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Div(left, right))
                elif flex_op in [
                    "FLEX_OP_DME_UPPER_EYELID",
                    "FLEX_OP_DME_LOWER_EYELID",
                ]:
                    stack.pop(-1)
                    stack.pop(-1)
                    stack.pop(-1)
                    stack.append(Value(1.0))
                elif flex_op == "FLEX_OP_DOMINATE":
                    count = index + 1
                    values = [stack.pop(-1) for _ in range(count)]
                    dom = Dominator(*values)
                    stack.append(dom)
                elif flex_op == "FLEX_OP_FETCH1":
                    stack.append(FetchController(get_flex_cnt(index)))
                elif flex_op == "FLEX_OP_FETCH2":
                    stack.append(FetchFlex(get_flex_desc(index)))
                elif flex_op == "FLEX_OP_MAX":
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Max(left, right))
                elif flex_op == "FLEX_OP_MIN":
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Min(left, right))
                elif flex_op == "FLEX_OP_MUL":
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Mul(left, right))
                elif flex_op == "FLEX_OP_NEG":
                    stack.append(Neg(stack.pop(-1)))
                elif flex_op == "FLEX_OP_NWAY":
                    flex_cnt_value = int(stack.pop(-1).value)
                    flex_cnt = FetchController(get_flex_cnt(flex_cnt_value))
                    f_w = stack.pop(-1)
                    f_z = stack.pop(-1)
                    f_y = stack.pop(-1)
                    f_x = stack.pop(-1)
                    gtx = Min(Value(1.0), Neg(Min(Value(0.0), Sub(f_x, flex_cnt))))
                    lty = Min(Value(1.0), Neg(Min(Value(0.0), Sub(flex_cnt, f_y))))
                    remap_x = Min(Max(Div(Sub(flex_cnt, f_x), (Sub(f_y, f_x))), Value(0.0)), Value(1.0))
                    gtey = Neg(Sub(Min(Value(1.0), Neg(Min(Value(0.0), Sub(flex_cnt, f_y)))), Value(1.0)))
                    ltez = Neg(Sub(Min(Value(1.0), Neg(Min(Value(0.0), Sub(f_z, flex_cnt)))), Value(1.0)))
                    gtz = Min(Value(1.0), Neg(Min(Value(0.0), Sub(f_z, flex_cnt))))
                    ltw = Min(Value(1.0), Neg(Min(Value(0.0), Sub(flex_cnt, f_w))))
                    remap_z = Sub(Value(1.0),
                                  Min(Max(Div(Sub(flex_cnt, f_z), (Sub(f_w, f_z))), Value(0.0)), Value(1.0)))
                    final_expr = Add(Add(Mul(Mul(gtx, lty), remap_x), Mul(gtey, ltez)), Mul(Mul(gtz, ltw), remap_z))

                    final_expr = Mul(final_expr, FetchController(get_flex_cnt(index)))
                    stack.append(final_expr)
                elif flex_op == "FLEX_OP_SUB":
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Sub(left, right))
                elif flex_op == "FLEX_OP_TWO_WAY_0":
                    mx = Max(Add(FetchController(get_flex_cnt(index)), Value(1.0)), Value(0.0))
                    mn = Min(mx, Value(1.0))
                    res = Sub(1, mn)
                    stack.append(res)
                elif flex_op == "FLEX_OP_TWO_WAY_1":
                    mx = Max(FetchController(get_flex_cnt(index)), Value(0.0))
                    mn = Min(mx, Value(1.0))
                    stack.append(mn)
                else:
                    print("Unknown OP", op)
            if len(stack) > 1 or not stack:
                print(f"failed to parse ({get_flex_desc(rule['m_nFlex'])}) flex rule")
                print(stack)
                continue
            final_expr = stack.pop(-1)
            # name = self.get_value('stereo_flexes').get(rule.flex_index, self.flex_names[rule.flex_index])
            name = get_flex_desc(rule['m_nFlex'])
            flex_rules[name] = final_expr
            # except:
            #     pass
        return flex_rules
