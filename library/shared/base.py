class Base:
    storage = {}

    @classmethod
    def store_value(cls, key, value):
        cls.storage[key] = value

    @classmethod
    def get_value(cls, key):
        return cls.storage.get(key, None)

    @property
    def mdl_version(self):
        mdl = self.get_value('MDL')
        return mdl.header.version
