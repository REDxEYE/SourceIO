import array

uint32_t = 'I'


def murmur_hash2(data: str, seed=0):
    """
    Generate a 32-bit hash from a string using the MurmurHash2 algorithm
    takes a bytestring!
    Pure-python implementation.
    """
    data = data.encode("ascii")
    input_len = len(data)

    # m and r are mixing constants generated offline
    # They're not really magic, they just happen to work well
    m = 0x5bd1e995
    # r = 24

    # Initialize the hash to a "random" value
    h = seed ^ input_len

    # Mix 4 bytes at a time into the hash
    x = input_len % 4
    o = input_len - x

    for k in array.array(uint32_t, data[:o]):
        # Original Algorithm
        # k *= m;
        # k ^= k >> r;
        # k *= m;

        # h *= m;
        # h ^= k;

        # My Algorithm
        k = (k * m) & 0xFFFFFFFF
        h = (((k ^ (k >> 24)) * m) ^ (h * m)) & 0xFFFFFFFF

        # Explanation: We need to keep it 32-bit. There are a few rules:
        # 1. Inputs to >> must be truncated, it never overflows
        # 2. Inputs to * must be truncated, it may overflow
        # 3. Inputs to ^ may be overflowed, it overflows if any input was overflowed
        # 4. The end result must be truncated
        # Therefore:
        # b = k * m -> may overflow, we truncate it because b >> r cannot take overflowed data
        # c = b ^ (b >> r) -> never overflows, as b is truncated and >> never does
        # h = (c * m) ^ (h * m) -> both inputs to ^ may overflow, but since ^ can take it, we truncate once afterwards.

    # Handle the last few bytes of the input array
    if x > 0:
        if x > 2:
            h ^= data[o + 2] << 16
        if x > 1:
            h ^= data[o + 1] << 8
        h = ((h ^ data[o]) * m) & 0xFFFFFFFF

    # Do a few final mixes of the hash to ensure the last few
    # bytes are well incorporated

    # Original:
    # h ^= h >> 13;
    # h *= m;
    # h ^= h >> 15;

    h = ((h ^ (h >> 13)) * m) & 0xFFFFFFFF
    return h ^ (h >> 15)


if __name__ == '__main__':
    a = murmur_hash2('model', 0x31415926)
    print(a)
