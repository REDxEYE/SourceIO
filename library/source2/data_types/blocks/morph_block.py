import logging
import struct
from collections import defaultdict
from enum import Enum
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from ....shared.content_providers.content_manager import ContentManager
from ....source1.mdl.v49.flex_expressions import Value, FetchController, FetchFlex, Add, Sub, Mul, Div, Neg, Max, Min, \
    Combo, Dominator
from ....utils import Buffer
from ...resource_types.compiled_texture_resource import \
    CompiledTextureResource
from ...resource_types.compiled_resource import CompiledResource
from .kv3_block import KVBlock


class FlexOpCode(Enum):
    FLEX_OP_CONST = "FLEX_OP_CONST"
    FLEX_OP_FETCH1 = "FLEX_OP_FETCH1"
    FLEX_OP_FETCH2 = "FLEX_OP_FETCH2"
    FLEX_OP_ADD = "FLEX_OP_ADD"
    FLEX_OP_SUB = "FLEX_OP_SUB"
    FLEX_OP_MUL = "FLEX_OP_MUL"
    FLEX_OP_DIV = "FLEX_OP_DIV"
    FLEX_OP_NEG = "FLEX_OP_NEG"
    FLEX_OP_EXP = "FLEX_OP_EXP"
    FLEX_OP_OPEN = "FLEX_OP_OPEN"
    FLEX_OP_CLOSE = "FLEX_OP_CLOSE"
    FLEX_OP_COMMA = "FLEX_OP_COMMA"
    FLEX_OP_MAX = "FLEX_OP_MAX"
    FLEX_OP_MIN = "FLEX_OP_MIN"
    FLEX_OP_2WAY_0 = "FLEX_OP_2WAY_0"
    FLEX_OP_2WAY_1 = "FLEX_OP_2WAY_1"
    FLEX_OP_NWAY = "FLEX_OP_NWAY"
    FLEX_OP_COMBO = "FLEX_OP_COMBO"
    FLEX_OP_DOMINATE = "FLEX_OP_DOMINATE"
    FLEX_OP_DME_LOWER_EYELID = "FLEX_OP_DME_LOWER_EYELID"
    FLEX_OP_DME_UPPER_EYELID = "FLEX_OP_DME_UPPER_EYELID"
    FLEX_OP_SQRT = "FLEX_OP_SQRT"
    FLEX_OP_REMAPVALCLAMPED = "FLEX_OP_REMAPVALCLAMPED"
    FLEX_OP_SIN = "FLEX_OP_SIN"
    FLEX_OP_COS = "FLEX_OP_COS"
    FLEX_OP_ABS = "FLEX_OP_ABS"


class MorphBlock(KVBlock):

    def __init__(self, buffer: Buffer, resource: CompiledResource):
        super().__init__(buffer, resource)
        self._morph_datas: Dict[int, Dict[str, npt.NDArray[np.float32]]] = defaultdict(dict)
        self._vmorf_texture = None

    @staticmethod
    def _get_struct(ntro):
        return ntro.struct_by_name('MorphSetData_t')

        # encoding_type = encoding_type[0].split('::')[-1]
        # lookup_type = lookup_type[0].split('::')[-1]

    @property
    def lookup_type(self):
        return self.get("m_nLookupType", "LOOKUP_TYPE_VERTEX_ID")

    @property
    def encoding_type(self):
        return self.get("m_nEncodingType", "ENCODING_TYPE_OBJECT_SPACE")

    @property
    def bundles(self) -> List[str]:
        return self['m_bundleTypes']

    def get_bundle_id(self, bundle_name):
        if bundle_name in self.bundles:
            return self.bundles.index(bundle_name)

    def get_morph_data(self, flex_name: str, bundle_id: int, cm: ContentManager):
        bundle_data = self._morph_datas[bundle_id]
        if flex_name in bundle_data:
            return bundle_data[flex_name]
        assert self.lookup_type == 'LOOKUP_TYPE_VERTEX_ID'
        assert self.encoding_type == 'ENCODING_TYPE_OBJECT_SPACE'
        if self._vmorf_texture is None:
            vmorf = self._resource.get_child_resource(self['m_pTextureAtlas'], cm, CompiledTextureResource)
            if not vmorf:
                logging.error(f'Failed to find {self["m_pTextureAtlas"]!r} morf texture')
                return None
            self._vmorf_texture = texture, (t_width, t_height) = vmorf.get_texture_data(0, False)
        else:
            texture, (t_width, t_height) = self._vmorf_texture

        width = self['m_nWidth']
        height = self['m_nHeight']

        for morph_data_ in self['m_morphDatas']:
            if morph_data_['m_name'] == flex_name:
                morph_data = morph_data_
                break
        else:
            logging.error(f'Failed to find morph data for {flex_name!r} flex')
            return None
        rect_flex_data = bundle_data[morph_data['m_name']] = np.zeros((height, width, 4), dtype=np.float32)

        for n, rect in enumerate(morph_data['m_morphRectDatas']):
            bundle = rect['m_bundleDatas'][bundle_id]

            rect_width = round(rect['m_flUWidthSrc'] * t_width)
            rect_height = round(rect['m_flVHeightSrc'] * t_height)
            dst_x = rect['m_nXLeftDst']
            dst_y = rect['m_nYTopDst']
            y_slice = slice(dst_y, dst_y + rect_height)
            x_slice = slice(dst_x, dst_x + rect_width)
            rect_u = round(bundle['m_flULeftSrc'] * t_width)
            rect_v = round(bundle['m_flVTopSrc'] * t_height)
            morph_data_rect = texture[rect_v:rect_v + rect_height, rect_u:rect_u + rect_width, :]

            transformed_data = morph_data_rect
            transformed_data = np.multiply(transformed_data, bundle['m_ranges'])
            transformed_data = np.add(transformed_data, bundle['m_offsets'])
            rect_flex_data[y_slice, x_slice, :] = transformed_data

        return rect_flex_data

    def decompile_flex_rules(self):
        rules = []
        m_flex_controllers = self["m_FlexControllers"]
        flex_descs = self["m_FlexDesc"]
        for m_flex_rule in self["m_FlexRules"]:
            flex_desc = flex_descs[m_flex_rule["m_nFlex"]]
            expr = ""
            stack = []
            for op in m_flex_rule["m_FlexOps"]:
                op_code = FlexOpCode(op["m_OpCode"])
                data = op["m_Data"]

                if op_code == FlexOpCode.FLEX_OP_CONST:
                    stack.append(Value(struct.unpack("f", struct.pack("i", data))[0]))
                elif op_code == FlexOpCode.FLEX_OP_FETCH1:
                    stack.append(FetchController(m_flex_controllers[data]["m_szName"]))
                elif op_code == FlexOpCode.FLEX_OP_FETCH2:
                    stack.append(FetchFlex(flex_descs[data]["m_szFacs"]))
                elif op_code == FlexOpCode.FLEX_OP_ADD:
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Add(left, right))
                elif op_code == FlexOpCode.FLEX_OP_SUB:
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Sub(left, right))
                elif op_code == FlexOpCode.FLEX_OP_MUL:
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Mul(left, right))
                elif op_code == FlexOpCode.FLEX_OP_DIV:
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Div(left, right))
                elif op_code == FlexOpCode.FLEX_OP_NEG:
                    stack.append(Neg(stack.pop(-1)))
                elif op_code == FlexOpCode.FLEX_OP_MAX:
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Max(left, right))
                elif op_code == FlexOpCode.FLEX_OP_MIN:
                    right = stack.pop(-1)
                    left = stack.pop(-1)
                    stack.append(Min(left, right))
                elif op_code == FlexOpCode.FLEX_OP_COMBO:
                    count = data
                    values = [stack.pop(-1) for _ in range(count)]
                    combo = Combo(*values)
                    stack.append(combo)
                elif op_code == FlexOpCode.FLEX_OP_DOMINATE:
                    count = data + 1
                    values = [stack.pop(-1) for _ in range(count)]
                    dom = Dominator(*values)
                    stack.append(dom)
                elif op_code == FlexOpCode.FLEX_OP_2WAY_0:
                    mx = Max(Add(FetchController(m_flex_controllers[data]["m_szName"]), Value(1.0)), Value(0.0))
                    mn = Min(mx, Value(1.0))
                    res = Sub(1, mn)
                    stack.append(res)
                elif op_code == FlexOpCode.FLEX_OP_2WAY_1:
                    mx = Max(FetchController(m_flex_controllers[data]["m_szName"]), Value(0.0))
                    mn = Min(mx, Value(1.0))
                    stack.append(mn)
                elif op_code == FlexOpCode.FLEX_OP_NWAY:
                    flex_cnt_value = int(stack.pop(-1).value)
                    flex_cnt = FetchController(m_flex_controllers[flex_cnt_value]["m_szName"])
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

                    final_expr = Mul(final_expr, FetchController(m_flex_controllers[data]["m_szName"]))
                    stack.append(final_expr)
                elif op_code == FlexOpCode.FLEX_OP_DME_UPPER_EYELID:
                    stack.pop(-1)
                    stack.pop(-1)
                    stack.pop(-1)
                    stack.append(Value(1.0))
                elif op_code == FlexOpCode.FLEX_OP_DME_LOWER_EYELID:
                    stack.pop(-1)
                    stack.pop(-1)
                    stack.pop(-1)
                    stack.append(Value(1.0))
                else:
                    raise NotImplementedError(f"Opcode {op_code}({data}) not implemented.")
            assert len(stack) == 1
            print(flex_desc, m_flex_rule)
            print(stack[0])
            rules.append((flex_desc["m_szFacs"], stack[0].as_simple(), []))
        return rules
