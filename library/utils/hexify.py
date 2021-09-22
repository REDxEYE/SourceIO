from binascii import b2a_hex


def rhex(data):
    "Hex (XX XX XX XX) representation of bytes"
    a = [b2a_hex(p.to_bytes(1,'little')).decode() for p in data]
    return " ".join(a)


if __name__ == '__main__':
    t = rhex(b"asdasdasdsadasd")
    print(t)
