class InvalidFileMagic(Exception):
    def __init__(self, message:str, expected:bytes, actual:bytes, *args):
        super().__init__(f"{message}: expected {expected!r}, got {actual!r}", *args)