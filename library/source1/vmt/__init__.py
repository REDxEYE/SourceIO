import re
from math import radians
from typing import Union

from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import Buffer, TinyPath, kv1
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::VMT')

#: Shared evaluator for condition-named blocks. Its symbol table is the single place
#: that decides which content branch a material takes.
CONDITIONS = kv1.ConditionContext()


class VMT:
    def __init__(self, buffer: Buffer, filename: str, content_manager: ContentManager):
        self._usage_report = set()
        data = buffer.read()
        text = kv1.decode_text(data, str(filename)) if isinstance(data, bytes) else data
        # The parser recovers from damage rather than raising, so there is no parse
        # failure to catch -- an unreadable material yields an empty tree instead.
        root = kv1.loads(text, str(filename))
        self.shader, self.data = root.top()
        if not isinstance(self.data, kv1.KV1Block):
            logger.warn(f'{filename}: shader {self.shader!r} has a value where a block '
                        f'was expected')
            self.shader, self.data = 'FAILED_TO_LOAD', kv1.KV1Block()
        elif not self.shader:
            self.shader = 'FAILED_TO_LOAD'
        self._postprocess(content_manager)


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
            self._resolve_prefixed_conditions(self.data)
            self._resolve_conditional_blocks(self.data)
        except Exception as ex:
            logger.exception(f"Failed to resolve expression in material", ex)

    def _resolve_prefixed_conditions(self, node: kv1.KV1Block):
        """Apply ``<condition>?<key>`` parameters, e.g. ``!gameconsole?$phong``.

        The material system reads these as "set ``$phong`` only when not on a console".
        1,001 of them appear across 658 shipped materials, gating real shading inputs
        (``$phong``, ``$envmap``, ``$alpha``), so ignoring the prefix means ignoring
        the parameter.

        A resolved parameter *replaces* any plain key of the same name, since the
        conditional form is the more specific instruction. The prefix is only treated
        as a condition when every symbol in it is recognised -- an unrecognised one is
        far more likely to be a key that merely contains ``?``.
        """
        kept, apply = [], []
        for entry in node.entries:
            condition, separator, real_key = entry.key.rpartition('?')
            if not separator or not condition or not real_key:
                kept.append(entry)
                continue
            if not CONDITIONS.is_fully_known(condition):
                logger.debug(f'Leaving {entry.key!r} alone: {condition!r} is not a '
                             f'condition we recognise')
                kept.append(entry)
                continue
            result, error = CONDITIONS.evaluate(condition)
            if error:
                logger.debug(f'Keeping {entry.key!r}: {error}')
                kept.append(entry)
            elif result:
                apply.append((real_key, entry.value))
            # a false condition drops the parameter, as Source does
        if len(kept) != len(node.entries):
            node.entries[:] = kept
            for key, value in apply:
                node[key] = value

        for _, value in node.blocks():
            self._resolve_prefixed_conditions(value)

    def _resolve_conditional_blocks(self, node: kv1.KV1Block):
        """Flatten blocks whose *key* is a condition, e.g. ``>=dx90_20b { ... }``.

        A ``[$WIN32]`` suffix is KV1 grammar and the parser has already applied it.
        This is the other VMT convention: a fallback block named after the condition,
        whose contents override the surrounding parameters when it holds and are
        discarded when it does not.

        Only keys that are *known* condition symbols are treated this way. That is
        deliberate -- an unrecognised key is an ordinary sub-block such as ``proxies``,
        and evaluating it as a condition would resolve to false and silently delete it.
        """
        kept, promote = [], []
        for entry in node.entries:
            state = CONDITIONS.symbols.get(entry.key)
            if state is None:
                kept.append(entry)
            elif state and isinstance(entry.value, kv1.KV1Block):
                promote.append(entry.value)
            # a false condition, or one holding a scalar, is simply dropped
        if len(kept) != len(node.entries):
            node.entries[:] = kept
            # applied after the surviving entries so the fallback's values win
            for block in promote:
                for child_key, child_value in block.items():
                    node[child_key] = child_value

        for _, value in node.blocks():
            self._resolve_conditional_blocks(value)

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item) -> Union[kv1.KV1Block, str]:
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get_unvisited_params(self):
        unvisited_params = {}
        for k, v in self.data.items():
            if k not in self._usage_report:
                unvisited_params[k] = v
        return unvisited_params

    def get(self, name, default=None) -> Union[kv1.KV1Block, str]:
        value = self.data.get(name, default)
        if value == "":
            return default
        self._usage_report.add(name)
        return value

    def get_vector(self, name, default=(0, 0, 0)):
        raw_value = self.get(name, None)
        if raw_value is None:
            return default, None
        # Values are kept verbatim, and hand-authored ones are padded: TF2's
        # `models/player/sniper/sniper_lens` writes `"$envmaptint" " [1 .8 .4]"`.
        # The leading space has to go before the first character can classify the
        # syntax, or the vector is mistaken for a whitespace-separated list.
        raw_value = raw_value.strip()
        if not raw_value:
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
        try:
            return tuple(map(converter, values)), converter
        except ValueError:
            # Braces denote the 0-255 range, but some materials write those values
            # as floats -- e.g. Portal 2's `models/player/chell_nodraw` has
            # ``$color "{0.0 0.0 0.0}"``. Parse leniently while keeping `int` as the
            # reported type so callers still divide by 255.
            return tuple(float(value) for value in values), converter

    def get_string(self, name, default='invalid'):
        raw_value = self.get(name, default)
        return str(raw_value) if raw_value is not None else None

    @staticmethod
    def _first_number(raw_value: str):
        """Best-effort scalar extraction from a hand-authored VMT value.

        Shipped materials contain typos and vectors where a scalar is expected --
        e.g. ``$detailblendfactor .4```` (stray backtick, Portal 2
        ``nature/dirtfloor004d``) or ``$alpha ".5 .5 .5"``. Pull the first numeric
        token out instead of raising, so one bad key cannot fail the whole material.
        """
        match = re.search(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', raw_value)
        return float(match.group(0)) if match else None

    def get_int(self, name, default=0):
        raw_value = self.get(name, None)
        if raw_value is None:
            return default
        try:
            return int(float(raw_value)) if '.' in raw_value else int(raw_value)
        except (TypeError, ValueError):
            number = self._first_number(str(raw_value))
            if number is None:
                logger.warn(f'Cannot read {name!r} as int: {raw_value!r}')
                return default
            return int(number)

    def get_float(self, name, default=0.0):
        raw_value = self.get(name, None)
        if raw_value is None:
            return default
        try:
            return float(raw_value.replace("[", "").replace("]", ""))
        except (AttributeError, TypeError, ValueError):
            number = self._first_number(str(raw_value))
            if number is None:
                logger.warn(f'Cannot read {name!r} as float: {raw_value!r}')
                return default
            return number

    def get_transform_matrix(self, name, default=None):
        if default is None:
            default = {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1.0), 'rotate': (0.0, 0.0, 0.0),
                       'translate': (0.0, 0.0, 0.0)}
        raw_value = self.get(name)
        if raw_value is None:
            return None
        matrix = default.copy()
        tokens = raw_value.split()

        def take_numbers(count: int) -> list[float]:
            """Pop up to ``count`` leading numeric tokens.

            Hand-authored transforms are not always fully specified -- e.g.
            ``$bumptransform "center 0 0 scale 5.25 rotate 90 translate 0 0"`` gives
            ``scale`` a single uniform value. Popping a fixed number of tokens would
            swallow the next keyword, so stop as soon as a non-number appears and let
            the caller broadcast what it got.
            """
            values = []
            while len(values) < count and tokens:
                try:
                    values.append(float(tokens[0]))
                except ValueError:
                    break
                tokens.pop(0)
            return values

        while tokens:
            name = tokens.pop(0)
            if name == 'center':
                values = take_numbers(2)
                if values:
                    x = values[0]
                    y = values[1] if len(values) > 1 else x
                    matrix[name] = x, y, 0.0
            elif name == 'scale':
                values = take_numbers(2)
                if values:
                    x = values[0]
                    y = values[1] if len(values) > 1 else x
                    matrix[name] = x, y, 1.0
            elif name == 'rotate':
                values = take_numbers(1)
                if values:
                    matrix[name] = 0, 0, radians(values[0])
            elif name == 'translate':
                values = take_numbers(2)
                if values:
                    x = values[0]
                    y = values[1] if len(values) > 1 else x
                    matrix[name] = x, y, 0.0
            else:
                logger.warn(f'Unhandled transform component {name!r} in {raw_value!r}')

        return matrix
