import struct


def int16_to_float(value):
    sign = (value & 0x8000) >> 15
    float_sign = -1 if sign == 1 else 1
    mantissa = value & 0x3FF
    biased_exponent = (value & 0x7C00) >> 10

    if (biased_exponent == 31) and (mantissa == 0):
        return 65504.0 * float_sign
    if biased_exponent == 31:
        return 0

    if biased_exponent == 0 and mantissa != 0:
        float_mantissa = mantissa / 1024.0
        return float_sign * float_mantissa * (1.0 / 16384.0)
    else:
        result_mantissa = mantissa << (23 - 10)
        if biased_exponent == 0:
            result_biased_exponent = 0
        else:
            result_biased_exponent = (biased_exponent - 15 + 127) << 23
        result_sign = sign << 31

        return struct.unpack('!f', struct.pack('!I', (result_sign | result_biased_exponent | result_mantissa)))[0]
