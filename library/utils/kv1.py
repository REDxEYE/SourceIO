"""Valve KeyValues 1 (KV1) parser -- VMT, VDF, gameinfo, MDL keyvalues, entity lumps.

Tolerant like Source: damage is reported and skipped, never fatal. Token based and
newline agnostic, as ``KeyValues::RecursiveLoadFromBuffer``::

    pair  := key (value | '{' block '}') condition?

Behaviour that real content forces:
conditions resolve at parse time (unknown symbol false, unparseable kept);
``[1 1 1]`` is a value but ``[!$ps3]`` a condition; ``"$color2" .25 .25 .25`` folds to
one vector while ``$model 1 "$basetexture" "x"`` stays two pairs; ``game+mod`` expands
to one entry per part and duplicate keys survive in order. Escapes are off by default
(VMT paths are full of ``models\\props\\x``); pass ``escapes=True`` for Steam .vdf/.acf.
"""
from enum import Enum
from typing import Any, Iterator, Mapping, NamedTuple, Sequence, Union

from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('KV1')

__all__ = [
    'KV1Block', 'KV1Entry', 'KV1Diagnostic', 'Severity', 'ConditionContext',
    'DEFAULT_CONDITIONS', 'loads', 'load', 'load_bytes', 'loads_blocks',
    'dumps', 'dump', 'parse_vector', 'is_integer_syntax',
]

#: Symbols for ``[...]`` conditions. Deliberately describes a Windows PC even on Linux:
#: the flags pick a *content* branch, and asset import always wants the PC one.
DEFAULT_CONDITIONS: dict[str, bool] = {
    '$win32': True,
    '$windows': True,
    '$x360': False,
    '$ps3': False,
    '$osx': False,
    '$linux': False,
    '$posix': False,
    '$mobile': False,
    '$lowfill': False,
    '$gl': False,
    # DX level gates: take the high branch a modern GPU would get.
    '>=dx90': True,
    '>=dx90_20b': True,
    '<dx90_20b': False,
    '<dx90': False,
    '>=dx80': True,
    '<dx80': False,
    '>dx80': True,
    '>=dx70': True,
    '<dx70': False,
    'gpu>=1': True,
    'gpu>=2': True,
    'gpu>=3': True,
    'gpu>1': True,
    'gpu>2': True,
    'gpu<1': False,
    'gpu<2': False,
    'gpu<3': False,
    # Used by VMT `<condition>?<key>` params.
    '360': False,
    'ps3': False,
    'sonyps3': False,
    'gameconsole': False,
    'lowfill': False,
    'srgb': True,
    'srgb_pc': True,
}


class Severity(Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'


class KV1Diagnostic(NamedTuple):
    """One recovered problem. Never raised; collected and logged."""
    severity: Severity
    message: str
    line: int
    column: int
    source: str

    def __str__(self):
        return f'{self.source}:{self.line}:{self.column}: {self.severity.value}: {self.message}'


class ConditionContext:
    """Resolves ``[...]`` expressions. Supports ``!``, ``&&``, ``||`` and parentheses.

    An unknown symbol is false, matching Source with the convar unset.
    """

    def __init__(self, symbols: Mapping[str, bool] | None = None,
                 unknown_is_true: bool = False):
        self.symbols = dict(DEFAULT_CONDITIONS)
        if symbols:
            self.symbols.update({k.lower(): v for k, v in symbols.items()})
        self.unknown_is_true = unknown_is_true
        self.unknown_seen: set[str] = set()

    def lookup(self, symbol: str) -> bool:
        key = symbol.lower()
        if key in self.symbols:
            return self.symbols[key]
        self.unknown_seen.add(key)
        return self.unknown_is_true

    def is_fully_known(self, expression: str) -> bool:
        """Whether every symbol is modelled.

        Asked before treating something as a condition at all: a key like
        ``!gameconsole?$phong`` is only safe to rewrite when the prefix is recognised,
        or an ordinary key containing ``?`` would evaluate false and be deleted.
        """
        try:
            tokens = _tokenize_condition(expression)
        except _ConditionSyntaxError:
            return False
        if not tokens:
            return False
        return all(token in ('!', '(', ')', '||', '&&') or token.lower() in self.symbols
                   for token in tokens)

    def evaluate(self, expression: str) -> tuple[bool, str | None]:
        """``(result, error)``; ``error`` is set when the expression was unparseable."""
        try:
            tokens = _tokenize_condition(expression)
            if not tokens:
                return True, 'empty condition'
            value, index = self._or_expr(tokens, 0)
            if index != len(tokens):
                return True, f'trailing tokens in condition {expression!r}'
            return value, None
        except _ConditionSyntaxError as ex:
            # Keep the key: not understanding a condition is no reason to drop data.
            return True, str(ex)

    def _or_expr(self, tokens: Sequence[str], index: int) -> tuple[bool, int]:
        value, index = self._and_expr(tokens, index)
        while index < len(tokens) and tokens[index] == '||':
            rhs, index = self._and_expr(tokens, index + 1)
            value = value or rhs
        return value, index

    def _and_expr(self, tokens: Sequence[str], index: int) -> tuple[bool, int]:
        value, index = self._unary(tokens, index)
        while index < len(tokens) and tokens[index] == '&&':
            rhs, index = self._unary(tokens, index + 1)
            value = value and rhs
        return value, index

    def _unary(self, tokens: Sequence[str], index: int) -> tuple[bool, int]:
        if index >= len(tokens):
            raise _ConditionSyntaxError('unexpected end of condition')
        token = tokens[index]
        if token == '!':
            value, index = self._unary(tokens, index + 1)
            return (not value), index
        if token == '(':
            value, index = self._or_expr(tokens, index + 1)
            if index >= len(tokens) or tokens[index] != ')':
                raise _ConditionSyntaxError('missing ")" in condition')
            return value, index + 1
        if token in ('||', '&&', ')'):
            raise _ConditionSyntaxError(f'unexpected {token!r} in condition')
        return self.lookup(token), index + 1


class _ConditionSyntaxError(Exception):
    pass


def _tokenize_condition(expression: str) -> list[str]:
    """Split into symbols and operators. ``>=dx90``/``gpu>=1`` are single symbols, so
    comparison characters glue to their word instead of becoming operators."""
    tokens: list[str] = []
    index, length = 0, len(expression)
    while index < length:
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        if expression.startswith('||', index) or expression.startswith('&&', index):
            tokens.append(expression[index:index + 2])
            index += 2
            continue
        if char in '()':
            tokens.append(char)
            index += 1
            continue
        if char == '!':
            # `!=` belongs to a symbol; a lone `!` is negation.
            if expression.startswith('!=', index):
                start = index
                while index < length and not expression[index].isspace() \
                        and expression[index] not in '()|&':
                    index += 1
                tokens.append(expression[start:index])
                continue
            tokens.append('!')
            index += 1
            continue
        start = index
        while index < length and not expression[index].isspace() \
                and expression[index] not in '()!' \
                and not expression.startswith('||', index) \
                and not expression.startswith('&&', index):
            index += 1
        if index == start:  # never spin
            index += 1
        tokens.append(expression[start:index])
    return tokens


# --- tokens ---

class _Tok(Enum):
    STRING = 'string'
    OPEN = '{'
    CLOSE = '}'
    CONDITION = 'condition'
    EOF = 'eof'


class _Token(NamedTuple):
    kind: _Tok
    value: str
    line: int
    column: int
    quoted: bool
    #: First token on its line; bounds value folding.
    starts_line: bool


_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}

#: Newlines a quoted string may cross before it is judged unterminated. Values legally
#: contain newlines, but an unmatched quote would otherwise run to the next quote --
#: possibly megabytes into a corrupt lump. One newline is all real content needs.
MAX_STRING_LINES = 4


#: Typographic quotes sometimes wrap a key, e.g. ``“$selfillum” 1``, from editing in a
#: word processor. Stripped from keys only: localisation VDFs use them inside text.
_SMART_QUOTES = ('“”', '‘’', '„“', '«»')


def _normalise_key(key: str) -> str:
    """Strip padding a lookup could never match. Keys are identifiers, so trailing
    space is always a typo -- unlike values, which stay verbatim."""
    key = key.strip()
    for open_quote, close_quote in _SMART_QUOTES:
        if len(key) > 1 and key[0] == open_quote and key[-1] == close_quote:
            key = key[1:-1].strip()
            break
    return key


def _is_numeric(text: str) -> bool:
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def _all_numeric(text: str) -> bool:
    """Every part numeric, i.e. an unquoted vector rather than a condition."""
    parts = text.replace(',', ' ').split()
    return bool(parts) and all(_is_numeric(part) for part in parts)


class _Lexer:
    """Turns KV1 text into tokens, recording rather than raising on damage."""

    def __init__(self, text: str, source: str, escapes: bool,
                 diagnostics: list[KV1Diagnostic]):
        self.text = text
        self.source = source
        self.escapes = escapes
        self.diagnostics = diagnostics
        self.index = 0
        self.line = 1
        self.line_start = 0
        self._fresh_line = True

    def _diag(self, severity: Severity, message: str, column: int | None = None):
        column = self.index - self.line_start + 1 if column is None else column
        record = KV1Diagnostic(severity, message, self.line, column, self.source)
        self.diagnostics.append(record)
        if severity is Severity.ERROR:
            logger.warn(str(record))
        else:
            logger.debug(str(record))

    def _newline(self):
        self.line += 1
        self.index += 1
        self.line_start = self.index
        self._fresh_line = True

    def _skip_to_eol(self):
        found = self.text.find('\n', self.index)
        self.index = len(self.text) if found < 0 else found

    def tokens(self) -> Iterator[_Token]:
        text = self.text
        length = len(text)
        while self.index < length:
            char = text[self.index]

            if char == '\n':
                self._newline()
                continue
            if char in ' \t\r\v\f':
                self.index += 1
                continue

            # `//` everywhere, plus `\\` which some authoring tools emit.
            if char == '/' and text.startswith('//', self.index):
                self._skip_to_eol()
                continue
            if char == '\\' and text.startswith('\\\\', self.index):
                self._skip_to_eol()
                continue
            if char == '/' and text.startswith('/*', self.index):
                end = text.find('*/', self.index + 2)
                if end < 0:
                    self._diag(Severity.ERROR, 'unterminated /* comment')
                    self.index = length
                else:
                    self.line += text.count('\n', self.index, end)
                    self.index = end + 2
                continue

            starts_line, self._fresh_line = self._fresh_line, False
            line, column = self.line, self.index - self.line_start + 1

            if char == '{':
                self.index += 1
                yield _Token(_Tok.OPEN, '{', line, column, False, starts_line)
                continue
            if char == '}':
                self.index += 1
                yield _Token(_Tok.CLOSE, '}', line, column, False, starts_line)
                continue
            if char in '"\'':
                # Valve only delimits with `"`, but `'1'` does turn up meaning `1`,
                # and nothing starts a token with an apostrophe, so honouring `'` is
                # free.
                token = self._read_quoted(char, line, column, starts_line)
                if token is not None:
                    yield token
                continue
            if char == '[':
                yield from self._read_bracket(line, column, starts_line)
                continue
            if char == ']':
                self._diag(Severity.ERROR, "stray ']'")
                self.index += 1
                continue
            yield self._read_bare(line, column, starts_line)

        yield _Token(_Tok.EOF, '', self.line, self.index - self.line_start + 1,
                     False, self._fresh_line)

    def _salvaged(self, content: str, line: int, column: int, starts_line: bool,
                  ) -> _Token | None:
        """Text rescued from an unterminated string, or None when empty.

        An empty token would become a *key* and pair with whatever follows, so
        ``$key `value"`` would eat the next pair. A deliberate ``""`` key terminates
        properly and is kept.
        """
        if not content.strip():
            return None
        return _Token(_Tok.STRING, content.strip(), line, column, True, starts_line)

    def _read_quoted(self, quote: str, line: int, column: int,
                     starts_line: bool) -> _Token | None:
        """Read a quoted string, which may legitimately contain newlines.

A value like ``"vo<LF>pc"`` is real, and stopping at the newline
        would salvage only ``vo`` and shift every remaining pair in the block. Bounded
        by :data:`MAX_STRING_LINES` so an unmatched quote in corrupt input cannot
        swallow the file.
        """
        self.index += 1
        if not self.escapes:
            end = self.text.find(quote, self.index)
            if end >= 0:
                crossed = self.text.count('\n', self.index, end)
                # Same-line close is never ambiguous; crossing a newline needs the
                # _closes_a_value check.
                if crossed == 0:
                    value = self.text[self.index:end]
                    self.index = end + 1
                    return _Token(_Tok.STRING, value, line, column, True, starts_line)
                if crossed <= MAX_STRING_LINES and self._closes_a_value(end):
                    value = self.text[self.index:end]
                    self._diag(Severity.INFO,
                               f'string spans {crossed} newline(s): {value[:40]!r}',
                               column)
                    self.line += crossed
                    self.line_start = self.text.rfind('\n', self.index, end) + 1
                    self.index = end + 1
                    return _Token(_Tok.STRING, value, line, column, True, starts_line)
            return self._unterminated(line, column, starts_line, end)

        chunks: list[str] = []
        crossed = 0
        while self.index < len(self.text):
            char = self.text[self.index]
            if char == '\\' and self.index + 1 < len(self.text):
                nxt = self.text[self.index + 1]
                chunks.append(_ESCAPES.get(nxt, '\\' + nxt))
                self.index += 2
                continue
            if char == quote:
                self.index += 1
                self.line += crossed
                return _Token(_Tok.STRING, ''.join(chunks), line, column, True, starts_line)
            if char == '\n':
                crossed += 1
                if crossed > MAX_STRING_LINES:
                    self._diag(Severity.ERROR,
                               f'unterminated string {"".join(chunks)[:40]!r}', column)
                    return self._salvaged(''.join(chunks), line, column, starts_line)
                self.line_start = self.index + 1
            chunks.append(char)
            self.index += 1
        self._diag(Severity.ERROR, 'unterminated string at end of file', column)
        return self._salvaged(''.join(chunks), line, column, starts_line)

    #: What may legally follow the closing quote of a *value*.
    _AFTER_CLOSE = ' \t\r\n{}[]"\'/'

    def _closes_a_value(self, end: int) -> bool:
        """Whether the quote at ``end`` closes a value rather than opening a key.

        A candidate on a later line is ambiguous, and the next character decides it::

            "sounds" "vo<LF>pc"<LF>"rendermode"    -> newline, so it closes
            "$a" "unclosed<LF>"$b" "2"             -> '$', so it opens

        Without this the second form absorbs the following pair.
        """
        nxt = self.text[end + 1] if end + 1 < len(self.text) else ''
        return nxt == '' or nxt in self._AFTER_CLOSE

    def _unterminated(self, line: int, column: int, starts_line: bool,
                      end: int) -> _Token | None:
        """Recover an unmatched quote by taking the rest of the line."""
        newline = self.text.find('\n', self.index)
        stop = len(self.text) if newline < 0 else newline
        value = self.text[self.index:stop]
        self.index = stop
        reason = ('no closing quote' if end < 0
                  else 'next quote looks like it opens a key, not closes this value')
        self._diag(Severity.ERROR,
                   f'unterminated string {value[:40]!r} ({reason})', column)
        return self._salvaged(value, line, column, starts_line)

    def _read_bracket(self, line: int, column: int, starts_line: bool) -> Iterator[_Token]:
        """``[...]`` is a condition unless numeric, then a value. An unterminated ``[``
        is closed at the line end and classified the same way rather than discarded:
        ``$color2 [1 1 1`` still means a vector, ``value [$win32`` still a condition."""
        end = self.text.find(']', self.index)
        newline = self.text.find('\n', self.index)
        unterminated = end < 0 or (0 <= newline < end)
        if unterminated:
            stop = len(self.text) if newline < 0 else newline
            content = self.text[self.index + 1:stop]
            self.index = stop
            self._diag(Severity.ERROR, f'unterminated "[" ({content[:40]!r}), '
                                       f'closed at end of line', column)
        else:
            content = self.text[self.index + 1:end]
            self.index = end + 1
        if _all_numeric(content):
            # Kept bracketed so downstream vector parsing sees the original syntax.
            yield _Token(_Tok.STRING, f'[{content}]', line, column, False, starts_line)
        elif content.strip():
            yield _Token(_Tok.CONDITION, content.strip(), line, column, False, starts_line)
        elif not unterminated:
            yield _Token(_Tok.CONDITION, '', line, column, False, starts_line)

    def _read_bare(self, line: int, column: int, starts_line: bool) -> _Token:
        text = self.text
        start = self.index
        length = len(text)
        while self.index < length:
            char = text[self.index]
            if char in ' \t\r\n\v\f{}"\'[]':
                break
            if char == '/' and text.startswith('//', self.index):
                break
            self.index += 1
        return _Token(_Tok.STRING, text[start:self.index], line, column, False, starts_line)


# --- model ---

class KV1Entry(NamedTuple):
    """One key/value pair, with the condition it was gated on (if any)."""
    key: str
    value: Union[str, 'KV1Block']
    condition: str | None = None
    line: int = 0


class KV1Block:
    """An ordered multimap. Duplicates and order are preserved -- gameinfo's
    ``SearchPaths`` needs repeated ``Game`` keys, and order sets mount precedence.
    Keys compare lowercased, as Source does.

    Deliberately *not* a :class:`collections.abc.Mapping`: ``keys()`` yields duplicates,
    so ``dict(block)`` would silently lose entries. Use :meth:`to_dict`.
    """

    __slots__ = ('entries', 'diagnostics', 'includes')

    def __init__(self, entries: list[KV1Entry] | None = None,
                 diagnostics: list[KV1Diagnostic] | None = None,
                 includes: list[str] | None = None):
        self.entries: list[KV1Entry] = entries if entries is not None else []
        #: Populated on the root block only.
        self.diagnostics: list[KV1Diagnostic] = diagnostics if diagnostics is not None else []
        #: ``#base`` / ``#include`` targets, in order of appearance.
        self.includes: list[str] = includes if includes is not None else []

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[str]:
        return iter(entry.key for entry in self.entries)

    def __contains__(self, key) -> bool:
        key = str(key).lower()
        return any(entry.key == key for entry in self.entries)

    def __getitem__(self, key) -> Union[str, 'KV1Block']:
        key = str(key).lower()
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        raise KeyError(key)

    def get(self, key, default=None):
        """First entry for ``key``, like ``KeyValues::FindKey``."""
        key = str(key).lower()
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        return default

    def get_all(self, key) -> list:
        """Every entry for ``key``, in file order."""
        key = str(key).lower()
        return [entry.value for entry in self.entries if entry.key == key]

    #: Name used by the parser this replaces.
    get_multiple = get_all

    def items(self) -> Iterator[tuple[str, Union[str, 'KV1Block']]]:
        """Every pair including duplicates -- not a de-duplicated mapping view."""
        for entry in self.entries:
            yield entry.key, entry.value

    def keys(self) -> Iterator[str]:
        return iter(entry.key for entry in self.entries)

    def values(self) -> Iterator[Union[str, 'KV1Block']]:
        return iter(entry.value for entry in self.entries)

    def blocks(self) -> Iterator[tuple[str, 'KV1Block']]:
        for entry in self.entries:
            if isinstance(entry.value, KV1Block):
                yield entry.key, entry.value

    def __setitem__(self, key, value):
        key = str(key).lower()
        for index, entry in enumerate(self.entries):
            if entry.key == key:
                self.entries[index] = entry._replace(value=value)
                return
        self.entries.append(KV1Entry(key, value))

    def __delitem__(self, key):
        key = str(key).lower()
        remaining = [entry for entry in self.entries if entry.key != key]
        if len(remaining) == len(self.entries):
            raise KeyError(key)
        self.entries[:] = remaining

    def append(self, key: str, value, condition: str | None = None, line: int = 0):
        """Add an entry without replacing an existing one of the same key."""
        self.entries.append(KV1Entry(str(key).lower(), value, condition, line))

    def __repr__(self):
        return f'KV1Block({[(e.key, e.value) for e in self.entries]!r})'

    def top(self) -> tuple[str, Union[str, 'KV1Block']]:
        """The single root pair, e.g. ``("lightmappedgeneric", <block>)`` for a VMT."""
        if not self.entries:
            return '', KV1Block()
        if len(self.entries) > 1:
            logger.debug(f'{len(self.entries)} root nodes, using the first '
                         f'({self.entries[0].key!r})')
        entry = self.entries[0]
        return entry.key, entry.value

    def merge(self, other: 'KV1Block'):
        """Overlay ``other``, recursing into blocks. Used by VMT ``patch``."""
        for entry in other.entries:
            mine = self.get(entry.key)
            if isinstance(entry.value, KV1Block) and isinstance(mine, KV1Block):
                mine.merge(entry.value)
            else:
                self[entry.key] = entry.value

    def to_dict(self) -> dict[str, Any]:
        """Plain dict; duplicate keys collapse into a list."""
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, KV1Block):
                value = value.to_dict()
            if key in out:
                if not isinstance(out[key], list):
                    out[key] = [out[key]]
                out[key].append(value)
            else:
                out[key] = value
        return out

    def get_string(self, key, default: str | None = None) -> str | None:
        value = self.get(key)
        if value is None or isinstance(value, KV1Block):
            return default
        return value

    def get_int(self, key, default: int = 0) -> int:
        value = self.get_float(key, None)
        return default if value is None else int(value)

    def get_float(self, key, default: float | None = 0.0) -> float | None:
        text = self.get_string(key)
        if text is None:
            return default
        numbers = parse_vector(text)
        if not numbers:
            logger.debug(f'Cannot read {key!r} as a number: {text!r}')
            return default
        return numbers[0]

    def get_bool(self, key, default: bool = False) -> bool:
        text = self.get_string(key)
        if text is None:
            return default
        lowered = text.strip().lower()
        if lowered in ('1', 'true', 'yes', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'off', ''):
            return False
        numbers = parse_vector(text)
        return bool(numbers[0]) if numbers else default

    def get_vector(self, key, default: tuple[float, ...] | None = None
                   ) -> tuple[float, ...] | None:
        text = self.get_string(key)
        if text is None:
            return default
        numbers = parse_vector(text)
        return tuple(numbers) if numbers else default


def parse_vector(text: str) -> list[float]:
    """Every number in a KV1 scalar or vector: ``[1 1 1]``, ``{255 0 0}``, ``.5 .5 .5``
    and typos like ``.4```` or ``42]``. Brace syntax means the 0-255 range, but that is
    the caller's business -- see :func:`is_integer_syntax`.
    """
    cleaned = []
    for char in text:
        cleaned.append(char if (char.isdigit() or char in '+-.eE') else ' ')
    numbers = []
    for part in ''.join(cleaned).split():
        try:
            numbers.append(float(part))
        except ValueError:
            continue
    return numbers


def is_integer_syntax(text: str) -> bool:
    """``{255 0 0}`` means 0-255, ``[1 0 0]`` means 0-1."""
    return text.lstrip().startswith('{')


# --- parser ---

_DIRECTIVES = ('#base', '#include')

#: Key given to a keyless ``{ ... }`` block. Entity lumps and MDL keyvalues are bare
#: sequences of these, so they are kept rather than dropped; `loads_blocks` unwraps.
_ANON_KEY = '<anonymous>'


class _Parser:
    def __init__(self, lexer: _Lexer, conditions: ConditionContext,
                 diagnostics: list[KV1Diagnostic], keep_disabled: bool,
                 max_depth: int = 128):
        self.tokens = lexer.tokens()
        self.lexer = lexer
        self.conditions = conditions
        self.diagnostics = diagnostics
        self.keep_disabled = keep_disabled
        self.max_depth = max_depth
        self._peeked: _Token | None = None
        self.includes: list[str] = []

    def peek(self) -> _Token:
        if self._peeked is None:
            self._peeked = next(self.tokens)
        return self._peeked

    def next(self) -> _Token:
        if self._peeked is not None:
            token, self._peeked = self._peeked, None
            return token
        return next(self.tokens)

    def _diag(self, severity: Severity, message: str, token: _Token):
        record = KV1Diagnostic(severity, message, token.line, token.column,
                               self.lexer.source)
        self.diagnostics.append(record)
        if severity is Severity.ERROR:
            logger.warn(str(record))
        else:
            logger.debug(str(record))

    def parse_block(self, depth: int, terminated: bool) -> KV1Block:
        """Parse entries until ``}`` (when ``terminated``) or EOF."""
        block = KV1Block()
        while True:
            token = self.peek()

            if token.kind is _Tok.EOF:
                if terminated:
                    self._diag(Severity.ERROR, 'missing "}" -- block closed at '
                                               'end of file', token)
                return block
            if token.kind is _Tok.CLOSE:
                self.next()
                if not terminated:
                    self._diag(Severity.ERROR, 'stray "}"', token)
                    continue
                return block
            if token.kind is _Tok.CONDITION:
                self.next()
                self._diag(Severity.ERROR, f'condition [{token.value}] with nothing '
                                           f'to apply to', token)
                continue
            if token.kind is _Tok.OPEN:
                # Keyless block: entity lumps and MDL keyvalues are built this way, so
                # keep it under a reserved key rather than discarding its contents.
                self.next()
                self._diag(Severity.INFO, 'block without a key', token)
                block.append(_ANON_KEY, self.parse_block(depth + 1, True),
                             None, token.line)
                continue

            self.next()
            key = _normalise_key(token.value)
            if not key:
                # `"" "tonemap,,,0,-1"` -- Hammer emits nameless outputs. Dropping the
                # key would make the value look like the next key and lose both.
                self._diag(Severity.INFO, 'empty key kept', token)
            if key.lower() in _DIRECTIVES:
                self._read_directive(key, token)
                continue
            self._parse_pair(block, key, token, depth)

    def _read_directive(self, key: str, token: _Token):
        nxt = self.peek()
        if nxt.kind is _Tok.STRING:
            self.next()
            self.includes.append(nxt.value)
            self._diag(Severity.INFO, f'{key} {nxt.value!r} recorded but not resolved',
                       token)
        else:
            self._diag(Severity.ERROR, f'{key} without a target', token)

    def _parse_pair(self, block: KV1Block, key: str, key_token: _Token, depth: int):
        nxt = self.peek()

        # `key [cond] { ... }` / `key [cond] value`
        condition = None
        if nxt.kind is _Tok.CONDITION:
            self.next()
            condition = nxt.value
            nxt = self.peek()

        if nxt.kind is _Tok.OPEN:
            self.next()
            if depth >= self.max_depth:
                self._diag(Severity.ERROR, f'nesting deeper than {self.max_depth}, '
                                           f'skipping block', nxt)
                self.parse_block(depth + 1, True)
                return
            value: Union[str, KV1Block] = self.parse_block(depth + 1, True)
            trailing = self.peek()
            if trailing.kind is _Tok.CONDITION:
                self.next()
                condition = trailing.value if condition is None else condition
        elif nxt.kind is _Tok.STRING:
            self.next()
            value = self._fold_value(nxt)
            trailing = self.peek()
            if trailing.kind is _Tok.CONDITION:
                self.next()
                condition = trailing.value if condition is None else condition
        else:
            self._diag(Severity.ERROR, f'key {key!r} has no value', key_token)
            return

        self._emit(block, key, value, condition, key_token)

    def _fold_value(self, first: _Token) -> str:
        """Absorb trailing bare numbers into one value.

        ``"$color2" .25 .25 .25`` is one vector, but
        ``$model 1 "$basetexture" "tools/toolsblack"`` is two pairs. Only *bare* numbers
        on the *same line* after a numeric value fold, which separates them: the second
        pair's key is quoted.
        """
        if first.quoted or not _is_numeric(first.value):
            return first.value
        parts = [first.value]
        while True:
            nxt = self.peek()
            if (nxt.kind is not _Tok.STRING or nxt.quoted or nxt.starts_line
                    or not _is_numeric(nxt.value)):
                break
            self.next()
            parts.append(nxt.value)
        if len(parts) == 1:
            return parts[0]
        folded = ' '.join(parts)
        self._diag(Severity.INFO, f'folded {len(parts)} bare numbers into one value '
                                  f'({folded!r})', first)
        return folded

    def _emit(self, block: KV1Block, key: str, value, condition: str | None,
              token: _Token):
        if condition is not None:
            result, error = self.conditions.evaluate(condition)
            if error:
                self._diag(Severity.ERROR,
                           f'condition [{condition}] kept because it could not be '
                           f'evaluated: {error}', token)
            elif not result:
                if not self.keep_disabled:
                    self._diag(Severity.INFO,
                               f'{key!r} skipped by condition [{condition}]', token)
                    return

        # `game+mod` registers one value under several keys.
        if '+' in key:
            parts = [part for part in key.split('+')]
            if len(parts) > 1 and all(part and not part.isspace() for part in parts):
                for part in parts:
                    block.append(part, value, condition, token.line)
                return
        block.append(key, value, condition, token.line)


# --- frontend ---

def loads(text: str, name: str = '<memory>', *, escapes: bool = False,
          conditions: ConditionContext | Mapping[str, bool] | None = None,
          keep_disabled: bool = False) -> KV1Block:
    """Parse KV1 text. Always returns a block; problems land in ``.diagnostics``."""
    if not isinstance(conditions, ConditionContext):
        conditions = ConditionContext(conditions)
    diagnostics: list[KV1Diagnostic] = []
    lexer = _Lexer(text, name, escapes, diagnostics)
    parser = _Parser(lexer, conditions, diagnostics, keep_disabled)
    try:
        block = parser.parse_block(0, False)
    except Exception as ex:  # noqa: BLE001 - tolerance is the whole point
        logger.exception(f'Unrecoverable error parsing {name}', ex)
        diagnostics.append(KV1Diagnostic(Severity.ERROR, f'aborted: {ex}',
                                         lexer.line, 1, name))
        block = KV1Block()
    block.diagnostics = diagnostics
    block.includes = parser.includes
    if conditions.unknown_seen:
        logger.debug(f'{name}: unknown condition symbols treated as false: '
                     f'{sorted(conditions.unknown_seen)}')
    return block


def loads_blocks(text: str, name: str = '<memory>', **kwargs) -> list[KV1Block]:
    """Parse a bare sequence of ``{ ... }`` blocks (BSP entity lumps, MDL keyvalues).

    Keyed top-level entries are tolerated and grouped into a leading block.
    """
    root = loads(text, name, **kwargs)
    blocks: list[KV1Block] = []
    loose = KV1Block()
    for entry in root.entries:
        if entry.key == _ANON_KEY and isinstance(entry.value, KV1Block):
            blocks.append(entry.value)
        elif isinstance(entry.value, KV1Block):
            blocks.append(entry.value)
        else:
            loose.entries.append(entry)
    if loose.entries:
        blocks.insert(0, loose)
    for block in blocks:
        block.diagnostics = root.diagnostics
    return blocks


def load_bytes(data: bytes, name: str = '<memory>', **kwargs) -> KV1Block:
    """Decode then parse. Encoding is never allowed to be what stops a read."""
    return loads(decode_text(data, name), name, **kwargs)


def decode_text(data: bytes, name: str = '<memory>') -> str:
    for encoding in ('utf-8-sig', 'utf-8', 'cp1252'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    logger.debug(f'{name}: not valid UTF-8 or CP1252, decoding as latin-1')
    return data.decode('latin1', 'replace')


def load(path: TinyPath | str, **kwargs) -> KV1Block:
    """Parse a file from disk."""
    path = TinyPath(path)
    with open(path, 'rb') as handle:
        return load_bytes(handle.read(), str(path), **kwargs)


# --- writing ---

_ESCAPE_OUT = {'\\': '\\\\', '"': '\\"', '\n': '\\n', '\t': '\\t', '\r': '\\r'}


def _quote(text: str, escapes: bool, what: str) -> str:
    """Quote a key or value for output."""
    if escapes:
        return '"' + ''.join(_ESCAPE_OUT.get(c, c) for c in text) + '"'
    if '"' in text:
        # Without escapes there is no way to represent it, and Valve has the same
        # limitation. Warn rather than emit something that will not read back.
        logger.warn(f'{what} {text[:40]!r} contains a quote and needs escapes=True '
                    f'to round-trip')
    return f'"{text}"'


def dumps(data: 'KV1Block | Sequence[KV1Block]', *, escapes: bool = False,
          indent: str = '\t') -> str:
    """Serialise back to KV1 text. Accepts a block or a sequence of them.

    A sequence is written as bare ``{ ... }`` blocks, the form entity lumps use, so it
    pairs with :func:`loads_blocks`. Keys and values are always quoted, duplicates are
    each emitted, and a recorded condition is written back as a ``[...]`` suffix.

    ``loads(dumps(block))`` reproduces ``block``. The exception is a block parsed with
    ``keep_disabled=True``: its false conditions are written out and then resolved away
    on the next read.
    """
    lines: list[str] = []
    if isinstance(data, KV1Block):
        lines.extend(f'#base {_quote(target, escapes, "include")}'
                     for target in data.includes)
        _dump_entries(data, lines, 0, escapes, indent)
    else:
        for block in data:
            _dump_block_body(block, lines, 0, escapes, indent, key=None)
    return '\n'.join(lines) + '\n' if lines else ''


def dump(data: 'KV1Block | Sequence[KV1Block]', path: TinyPath | str, *,
         escapes: bool = False, indent: str = '\t', encoding: str = 'utf-8') -> None:
    """Serialise to a file."""
    text = dumps(data, escapes=escapes, indent=indent)
    with open(TinyPath(path), 'w', encoding=encoding, newline='\n') as handle:
        handle.write(text)


def _dump_entries(block: KV1Block, lines: list[str], depth: int, escapes: bool,
                  indent: str):
    pad = indent * depth
    for entry in block.entries:
        suffix = f' [{entry.condition}]' if entry.condition else ''
        if isinstance(entry.value, KV1Block):
            key = None if entry.key == _ANON_KEY else entry.key
            _dump_block_body(entry.value, lines, depth, escapes, indent, key, suffix)
        else:
            lines.append(f'{pad}{_quote(entry.key, escapes, "key")} '
                         f'{_quote(str(entry.value), escapes, "value")}{suffix}')


def _dump_block_body(block: KV1Block, lines: list[str], depth: int, escapes: bool,
                     indent: str, key: str | None, suffix: str = ''):
    pad = indent * depth
    if key is not None:
        lines.append(f'{pad}{_quote(key, escapes, "key")}{suffix}')
    elif suffix:
        lines.append(f'{pad}{suffix.lstrip()}')
    lines.append(f'{pad}{{')
    _dump_entries(block, lines, depth + 1, escapes, indent)
    lines.append(f'{pad}}}')
