import struct
from collections import defaultdict

from .data_block import DATA
import numpy as np

from ...source1.mdl.v49.flex_expressions import *
from ...shared.content_providers.content_manager import ContentManager


class MRPH(DATA):

    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.flex_data = {}

    def read_morphs(self):
        from ..data_blocks import TEXR
        from ..resource_types import ValveCompiledTexture
        if self.data['m_pTextureAtlas'] not in self._valve_file.available_resources:
            return False
        vmorf_actual_path = self._valve_file.available_resources.get(self.data['m_pTextureAtlas'], None)
        if not vmorf_actual_path:
            return False
        vmorf_path = ContentManager().find_file(vmorf_actual_path)
        if not vmorf_path:
            return False
        morph_atlas = ValveCompiledTexture(vmorf_path)
        morph_atlas.read_block_info()
        morph_atlas_data: TEXR = morph_atlas.get_data_block(block_name="DATA")[0]
        morph_atlas_data.read_image(False)
        raw_flex_data = np.frombuffer(morph_atlas_data.image_data, dtype=np.uint8)
        width = self.data['m_nWidth']
        height = self.data['m_nHeight']
        encoding_type = self.data.get('m_nEncodingType', 'ENCODING_TYPE_OBJECT_SPACE')
        lookup_type = self.data.get('m_nLookupType', 'LOOKUP_TYPE_VERTEX_ID')
        if isinstance(encoding_type, tuple):
            encoding_type = encoding_type[0].split('::')[-1]
            lookup_type = lookup_type[0].split('::')[-1]
        assert lookup_type == 'LOOKUP_TYPE_VERTEX_ID', "Unknown lookup type"
        assert encoding_type == 'ENCODING_TYPE_OBJECT_SPACE', "Unknown encoding type"
        bundle_types = self.data['m_bundleTypes']
        raw_flex_data = raw_flex_data.reshape((morph_atlas_data.width, morph_atlas_data.height, 4))

        for morph_datas in self.data['m_morphDatas']:
            morph_name = morph_datas['m_name']
            self.flex_data[morph_name] = np.zeros((len(bundle_types), height, width, 4), dtype=np.float32)
            for n, rect in enumerate(morph_datas['m_morphRectDatas']):
                rect_width = round(rect['m_flUWidthSrc'] * morph_atlas_data.width)
                rect_height = round(rect['m_flVHeightSrc'] * morph_atlas_data.height)
                dst_x = rect['m_nXLeftDst']
                dst_y = rect['m_nYTopDst']
                for c, bundle in enumerate(rect['m_bundleDatas']):
                    rect_u = round(bundle['m_flULeftSrc'] * morph_atlas_data.width)
                    rect_v = round(bundle['m_flVTopSrc'] * morph_atlas_data.height)
                    morph_data_rect = raw_flex_data[rect_v:rect_v + rect_height, rect_u:rect_u + rect_width, :]

                    transformed_data = np.divide(morph_data_rect, 255)
                    transformed_data = np.multiply(transformed_data, bundle['m_ranges'])
                    transformed_data = np.add(transformed_data, bundle['m_offsets'])
                    y_slice = slice(dst_y, dst_y + rect_height)
                    x_slice = slice(dst_x, dst_x + rect_width)
                    self.flex_data[morph_name][c, y_slice, x_slice, :] = transformed_data
        for k, v in self.flex_data.items():
            self.flex_data[k] = v.reshape((len(bundle_types), width * height, 4))
        return True

    def rebuild_flex_expressions(self):
        flex_rules = defaultdict(list)

        def get_flex_desc(index):
            return self.data['m_FlexDesc'][index]['m_szFacs']

        def get_flex_cnt(index):
            return self.data['m_FlexControllers'][index]['m_szName']

        for rule in self.data['m_FlexRules']:
            stack = []
            inputs = []
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
                elif flex_op == "FLEX_OP_DME_UPPER_EYELID":
                    close_lid_v_controller = get_flex_cnt(index)
                    inputs.append((close_lid_v_controller.name, 'DUE'))
                    close_lid_v = CustomFunction('rclamped', FetchController(close_lid_v_controller.name),
                                                 close_lid_v_controller.min, close_lid_v_controller.max,
                                                 0, 1)

                    flex_cnt_value = int(stack.pop(-1).value)
                    close_lid_controller = get_flex_cnt(flex_cnt_value)
                    inputs.append((close_lid_controller.name, 'DUE'))
                    close_lid = CustomFunction('rclamped', FetchController(close_lid_controller.name),
                                               close_lid_controller.min, close_lid_controller.max,
                                               0, 1)

                    blink_index = int(stack.pop(-1).value)
                    # blink = Value(0.0)
                    # if blink_index >= 0:
                    #     blink_controller = self.flex_controllers[blink_index]
                    #     inputs.append((blink_controller.name, 'DUE'))
                    #     blink_fetch = FetchController(blink_controller.name)
                    #     blink = CustomFunction('rclamped', blink_fetch,
                    #                            blink_controller.min, blink_controller.max,
                    #                            0, 1)

                    eye_up_down_index = int(stack.pop(-1).value)
                    eye_up_down = Value(0.0)
                    if eye_up_down_index >= 0:
                        eye_up_down_controller = get_flex_cnt(eye_up_down_index)
                        inputs.append((eye_up_down_controller.name, 'DUE'))
                        eye_up_down_fetch = FetchController(eye_up_down_controller.name)
                        eye_up_down = CustomFunction('rclamped', eye_up_down_fetch,
                                                     eye_up_down_controller.min, eye_up_down_controller.max,
                                                     -1, 1)

                    stack.append(CustomFunction('upper_eyelid_case', eye_up_down, close_lid_v, close_lid))
                elif flex_op == "FLEX_OP_DME_LOWER_EYELID":
                    close_lid_v_controller = get_flex_cnt(index)
                    inputs.append((close_lid_v_controller.name, 'DUE'))
                    close_lid_v = CustomFunction('rclamped', FetchController(close_lid_v_controller.name),
                                                 close_lid_v_controller.min, close_lid_v_controller.max,
                                                 0, 1)

                    flex_cnt_value = int(stack.pop(-1).value)
                    close_lid_controller = get_flex_cnt(flex_cnt_value)
                    inputs.append((close_lid_controller.name, 'DUE'))
                    close_lid = CustomFunction('rclamped', FetchController(close_lid_controller.name),
                                               close_lid_controller.min, close_lid_controller.max,
                                               0, 1)

                    blink_index = int(stack.pop(-1).value)
                    # blink = Value(0.0)
                    # if blink_index >= 0:
                    #     blink_controller = self.flex_controllers[blink_index]
                    #     inputs.append((blink_controller.name, 'DUE'))
                    #     blink_fetch = FetchController(blink_controller.name)
                    #     blink = CustomFunction('rclamped', blink_fetch,
                    #                            blink_controller.min, blink_controller.max,
                    #                            0, 1)

                    eye_up_down_index = int(stack.pop(-1).value)
                    eye_up_down = Value(0.0)
                    if eye_up_down_index >= 0:
                        eye_up_down_controller = get_flex_cnt(eye_up_down_index)
                        inputs.append((eye_up_down_controller.name, 'DUE'))
                        eye_up_down_fetch = FetchController(eye_up_down_controller.name)
                        eye_up_down = CustomFunction('rclamped', eye_up_down_fetch,
                                                     eye_up_down_controller.min, eye_up_down_controller.max,
                                                     -1, 1)

                    stack.append(CustomFunction('lower_eyelid_case', eye_up_down, close_lid_v, close_lid))
                elif flex_op == "FLEX_OP_DOMINATE":
                    count = index + 1
                    values = [stack.pop(-1) for _ in range(count)]
                    dom = Dominator(*values)
                    stack.append(dom)
                elif flex_op == "FLEX_OP_FETCH1":
                    inputs.append((get_flex_cnt(index), 'fetch1'))
                    stack.append(FetchController(get_flex_cnt(index)))
                elif flex_op == "FLEX_OP_FETCH2":
                    inputs.append((get_flex_desc(index), 'fetch2'))
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

                    inputs.append((get_flex_cnt(index), 'NWAY'))
                    flex_cnt = FetchController(get_flex_cnt(index))

                    inputs.append((get_flex_cnt(int(stack.pop(-1).value)), 'NWAY'))
                    multi_cnt = FetchController(get_flex_cnt(int(stack.pop(-1).value)))

                    f_w = stack.pop(-1)
                    f_z = stack.pop(-1)
                    f_y = stack.pop(-1)
                    f_x = stack.pop(-1)
                    final_expr = CustomFunction('nway', multi_cnt, flex_cnt, f_x, f_y, f_z, f_w)

                    final_expr = Mul(final_expr, FetchController(get_flex_cnt(index)))
                    stack.append(final_expr)
                elif flex_op == "FLEX_OP_SUB":
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Sub(left, right))
                elif flex_op == "FLEX_OP_TWO_WAY_0":
                    inputs.append((get_flex_cnt(index), '2WAY0'))
                    res = CustomFunction('rclamped', FetchController(get_flex_cnt(index)), -1, 0, 1, 0)
                    stack.append(res)
                elif flex_op == "FLEX_OP_TWO_WAY_1":
                    inputs.append((get_flex_cnt(index), '2WAY1'))
                    res = CustomFunction('clamp', FetchController(get_flex_cnt(index)), 0, 1)
                    stack.append(res)
                else:
                    print("Unknown OP", op)
            if len(stack) > 1 or not stack:
                print(f"failed to parse ({get_flex_desc(rule['m_nFlex'])}) flex rule")
                print(stack)
                continue
            final_expr = stack.pop(-1)
            name = get_flex_desc(rule['m_nFlex'])
            if final_expr not in flex_rules[name]:
                flex_rules[name].append(final_expr)

        for name in flex_rules.keys():
            flex_rules[name] = flex_rules[name][-1]

        return flex_rules
