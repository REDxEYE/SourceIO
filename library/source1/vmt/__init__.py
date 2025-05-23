import traceback
from math import radians
from typing import Union

from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import Buffer, TinyPath
from SourceIO.logger import SourceLogMan
from SourceIO.library.utils.kv_parser import ValveKeyValueParser, KVDataProxy, KVLexerException

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::VMT')


class VMT:
    def __init__(self, buffer: Buffer, filename: str, content_manager: ContentManager):
        self._usage_report = set()
        data = buffer.read()
        if isinstance(data, bytes):
            data = data.decode('latin1')
        parser = ValveKeyValueParser(buffer_and_name=(data, filename), self_recover=True)
        try:
            parser.parse()
            self.shader, self.data = parser.tree.top()
            del parser
            self._postprocess(content_manager)
        except KVLexerException as ex:
            logger.exception("Failed to parse material due to", ex)
            traceback.print_exc()
            self.shader = "FAILED_TO_LOAD"
            self.data = KVDataProxy([])


    def _lookup_material(self, content_manager: ContentManager, look_for: TinyPath):
        original_material = content_manager.find_file(look_for)
        # An extra lookup helps VtMB materials to be found
        if not original_material:
            original_material = content_manager.find_file(TinyPath('materials') / look_for.with_suffix('.vmt'))
        return original_material


    def _postprocess(self, content_manager: ContentManager):
        if self.shader == 'patch':
            look_for = TinyPath(self.get_string('include'))
            original_material = self._lookup_material(content_manager, look_for)
            if not original_material:
                logger.error(f'Failed to find original material: {look_for!r}')
                return
            patched_vmt = VMT(original_material, look_for, content_manager)
            if 'insert' in self:
                patch_data = self.get('insert', {})
                patched_vmt.data.merge(patch_data)
            if 'replace' in self:
                patch_data = self.get('replace', {})
                patched_vmt.data.merge(patch_data)
            self.shader = patched_vmt.shader
            self.data = patched_vmt.data
        try:
            self._resolve_expressions(self.data)
        except Exception as ex:
            logger.exception(f"Failed to resolve expression in material", ex)

    def _resolve_expressions(self, node: KVDataProxy):
        for key, value in node.items():
            if key in node.known_conditions and node.known_conditions[key]:
                if isinstance(value, (KVDataProxy, dict, list)):
                    for e_key, e_value in value.items():
                        node[e_key] = e_value
                del node[key]
                self._resolve_expressions(node)
                return
            if key in node.known_conditions and not node.known_conditions[key]:
                del node[key]
                self._resolve_expressions(node)
                return
            if isinstance(value, KVDataProxy):
                self._resolve_expressions(value)

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item) -> Union[KVDataProxy, str]:
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get_unvisited_params(self):
        unvisited_params = {}
        for k, v in self.data.items():
            if k not in self._usage_report:
                unvisited_params[k] = v
        return unvisited_params

    def get(self, name, default=None) -> Union[KVDataProxy, str]:
        value = self.data.get(name, default)
        if value == "":
            return default
        self._usage_report.add(name)
        return value

    def get_vector(self, name, default=(0, 0, 0)):
        raw_value = self.get(name, None)
        if raw_value is None:
            return default, None

        if raw_value[0] == '{':
            converter = int
        elif raw_value[0] == '[':
            converter = float
        elif len(raw_value.split()) > 1:
            return tuple(map(float, raw_value.split())), float
        else:
            return [float(raw_value)], float

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
        return float(raw_value.replace("[", "").replace("]", ""))

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
