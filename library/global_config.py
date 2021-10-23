from .utils.singleton import SingletonMeta


class GoldSrcConfig(metaclass=SingletonMeta):
    def __init__(self):
        self.use_hd = False
