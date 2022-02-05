from math import radians
from typing import Tuple, Union, List, IO

from ....logger import SLoggingManager
from ...shared.content_providers.content_manager import ContentManager
from ...utils.kv_parser import ValveKeyValueParser, _KVDataProxy

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::VMT')


class VMT:
    def __init__(self, buffer: IO, filename: str):
        data = buffer.read()
        if isinstance(data, bytes):
            data = data.decode('latin1')
        parser = ValveKeyValueParser(buffer_and_name=(data, filename), self_recover=True)
        parser.parse()
        self.shader, self.data = parser.tree.top()
        del parser
        self._postprocess()

    def _postprocess(self):
        content_manager = ContentManager()
        if self.shader == 'patch':
            original_material = content_manager.find_file(self.get_string('include'))
            if not original_material:
                logger.error(f'Failed to find original material {self.get_string("include")!r}')
                return
            patched_vmt = VMT(original_material, self.get_string('include'))
            if 'insert' in self:
                patch_data = self.get('insert', {})
                patched_vmt.data.merge(patch_data)
            if 'replace' in self:
                patch_data = self.get('replace', {})
                patched_vmt.data.merge(patch_data)
            self.shader = patched_vmt.shader
            self.data = patched_vmt.data

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item) -> Union[_KVDataProxy, str]:
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, name, default=None) -> Union[_KVDataProxy, str]:
        return self.data.get(name, default)

    def get_vector(self, name, default=(0, 0, 0)):
        raw_value = self.get(name, None)
        if raw_value is None:
            return default, None

        if raw_value[0] == '{':
            converter = int
            pass
        elif raw_value[0] == '[':
            converter = float
        else:
            return [float(raw_value)], float
            # raise ValueError(f'Not a vector value: {raw_value}')

        values = raw_value[1:-1].split()
        return tuple(map(converter, values)), converter

    def get_string(self, name, default='invalid'):
        raw_value = self.get(name, default)
        return str(raw_value) if raw_value is not None else None

    def get_int(self, name, default=0):
        raw_value = self.get(name, None)
        if raw_value is None:
            return default
        if raw_value and '.' in raw_value:
            return int(float(raw_value))
        return int(raw_value)

    def get_float(self, name, default=0.0):
        raw_value = self.get(name, None)
        if raw_value is None:
            return default
        return float(raw_value)

    def get_transform_matrix(self, name, default=None):
        if default is None:
            default = {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1.0), 'rotate': (0.0, 0.0, 0.0),
                       'translate': (0.0, 0.0, 0.0)}
        raw_value = self.get(name)
        if raw_value is None:
            return None
        matrix = default
        tokens = raw_value.split()
        while tokens:
            name = tokens.pop(0)
            if name == 'center':
                matrix[name] = float(tokens.pop(0)), float(tokens.pop(0)), 0.0
            elif name == 'scale':
                matrix[name] = float(tokens.pop(0)), float(tokens.pop(0)), 1.0
            elif name == 'rotate':
                matrix[name] = 0, 0, radians(float(tokens.pop(0)))
            elif name == 'translate':
                matrix[name] = float(tokens.pop(0)), float(tokens.pop(0)), 0.0
            else:
                print(f'Unhandled {name}')

        return matrix
