from pathlib import Path
from .murmurhash2 import murmur_hash2

MURMUR2SEED = 0x31415926


class EntityKeyValuesKeys:
    lookup_table = {}
    _all_keys = []

    def __init__(self, key_list: Path = Path(__file__).parent / Path("./entitykeyvalues_list.txt")):
        if not self.lookup_table:
            self._all_keys = key_list.open('r').readlines()
            self.precompute_keys()

    def precompute_keys(self):
        for skey in self._all_keys:
            mhash = murmur_hash2(skey.strip('\n'), MURMUR2SEED)
            self.lookup_table[mhash] = skey.strip('\n')

    def get(self, key_hash):
        return self.lookup_table.get(key_hash, key_hash)
