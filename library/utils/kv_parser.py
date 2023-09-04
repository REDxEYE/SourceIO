import warnings
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Mapping, Tuple, Union

__all__ = ["KVLexerException", "KVParserException", "ValveKeyValueParser", "KeyValuePair", "KVDataProxy"]

KeyValuePair = Tuple[str, Union[str, 'KeyValuePair', List['KeyValuePair']]]


class KVDataProxy(Mapping):
    known_conditions = {'$X360': False,
                        '$WIN32': True,
                        '$WINDOWS': True,
                        '$OSX': False,
                        '$LINUX': False,
                        '$POSIX': False,
                        '>=dx90_20b': True,
                        '<dx90_20b': False
                        }

    def __init__(self, data: List[KeyValuePair]):
        self.data = data

    def __contains__(self, item):
        return self.get(item) is not None

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator:
        return iter(a[0] for a in self.data)

    def __getitem__(self, item):
        value = self.get(item)
        if value is not None:
            return self._wrap_value(value)
        else:
            raise KeyError(f'Key {item!r} not found')

    def __delitem__(self, name: str) -> None:
        for item in self.data:
            if item[0] == name:
                self.data.remove(item)
                return

    def items(self):
        for key, value in self.data:
            yield key, self._wrap_value(value)

    def get(self, name, default=None) -> "KVDataProxy":
        name = name.lower()
        for key, value in self.data:
            if key == name:
                if isinstance(value, tuple) and len(value) == 2:
                    value, cond = value
                    if cond.startswith('!'):
                        cond = not self.known_conditions[cond[1:]]
                    else:
                        cond = self.known_conditions[cond]
                    if cond:
                        return self._wrap_value(value)
                else:
                    return self._wrap_value(value)
        return default

    def get_multiple(self, name) -> List["KVDataProxy"]:
        data = []
        name = name.lower()
        for key, value in self.data:
            if key == name:
                if isinstance(value, tuple) and len(value) == 2:
                    value, cond = value
                    if cond.startswith('!'):
                        cond = not self.known_conditions[cond[1:]]
                    else:
                        cond = self.known_conditions[cond]
                    if cond:
                        data.append(self._wrap_value(value))
                else:
                    data.append(self._wrap_value(value))
        return data

    def top(self) -> Tuple[str, Union[str, 'KVDataProxy']]:
        # assert len(self.data) == 1
        if len(self.data) > 1:
            print("More than one root node:")
            print(self.data[0])
            print(self.data[1])
        key, value = self.data[0]
        return key, self._wrap_value(value)

    def __setitem__(self, key, value):
        key = key.lower()
        for i, (name, _) in enumerate(self.data):
            if name == key:
                self.data[i] = name, self._wrap_value(value)
                return
        self.data.append((key, self._wrap_value(value)))

    def merge(self, other: 'KVDataProxy'):
        for o_item, o_value in other.items():
            if isinstance(o_value, KVDataProxy):
                if o_item not in self:
                    self[o_item] = o_value
                else:
                    self[o_item].merge(o_value)
            else:
                self[o_item] = o_value

    @staticmethod
    def _wrap_value(value):
        if isinstance(value, list):
            return KVDataProxy(value)
        return value

    def to_dict(self):
        items = {}
        for k, v in self.items():
            if isinstance(v, KVDataProxy):
                v = v.to_dict()
            items[k] = v
        return items


class KVLexerException(Exception):
    pass


class KVParserException(Exception):
    pass


class VKVToken(Enum):
    STRING = "String literal"
    NUMERIC = "Numeric literal"
    IDENTIFIER = "Identifier literal"
    COMMENT = "Comment literal"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    LBRACE = "{"
    RBRACE = "}"
    EXPRESSION = "Expression"
    NEWLINE = "\n"
    EOF = "End of file"


class ValveKeyValueLexer:

    def __init__(self, buffer: str, buffer_name: str = '<memory>'):
        self.buffer = buffer.replace('\r\n', '\n')
        self.buffer_name = buffer_name
        self._offset = 0
        self._line = 1
        self._column = 1

    @property
    def symbol(self):
        if self._offset < len(self.buffer):
            return self.buffer[self._offset]
        else:
            return ""

    @property
    def next_symbol(self):
        if self._offset + 1 < len(self.buffer):
            return self.buffer[self._offset + 1]
        else:
            return ""

    @property
    def leftover(self):
        return self.buffer[self._offset:]

    @property
    def line(self):
        return self._line

    @property
    def column(self):
        return self._column

    def advance(self):
        symbol = self.symbol
        if symbol:
            if symbol == '\r' and self.next_symbol == '\n':
                self._offset += 1

            if symbol == '\n' or symbol == '\r':
                self._line += 1
                self._column = 1
            else:
                self._column += 1
            self._offset += 1
        return symbol

    def skip(self, count=1):
        for _ in range(count):
            self.advance()

    @staticmethod
    def _is_valid_symbol(symbol):
        return (symbol.isprintable() or symbol in '\t\x7f\x1b') and symbol not in '$%{}[]"\'\n\r'

    def _is_valid_quoted_symbol(self, symbol):
        return self._is_valid_symbol(symbol) or symbol in '$%.,\'\\/<>=![]{}?'

    def _is_escaped_symbol(self):
        return self.next_symbol in '\'"\\'

    def read_simple_string(self, terminators='\n'):
        string_buffer = ""
        while True:
            symbol = self.symbol
            if symbol == "\\" and self.next_symbol == "x":
                self.advance()
            # if symbol in '"\'':
            #     self.advance()
            #     break
            if not self._is_valid_symbol(symbol) or symbol in terminators:
                break
            string_buffer += self.advance()
        return string_buffer.strip().rstrip()

    def read_quoted_string(self):
        terminator = self.advance()
        string_buffer = ""

        while True:
            symbol = self.symbol
            if symbol == "\\" and self.next_symbol == "x":
                self.advance()
                self.advance()
                self.advance()
            # if symbol == '\\' and self.next_symbol in '\'"ntr':
            #     self.advance()
            if not self._is_valid_quoted_symbol(symbol) or symbol in terminator + '\n':
                break

            string_buffer += self.advance()
        if self.symbol == terminator:
            self.advance()
        else:
            warnings.warn(f'Expected {terminator!r}, but got {self.symbol !r} at {self._line}:{self._column}')

        return string_buffer.strip().rstrip()

    def lex(self):
        while self._offset < len(self.buffer):
            symbol = self.symbol
            if symbol == '\n':
                if self.next_symbol == '\n':
                    while self.next_symbol == '\n':
                        self.advance()  # skip multiple new lines
                yield VKVToken.NEWLINE, self.advance()
            elif symbol.isspace():
                self.advance()
                continue
            elif symbol == '/' and self.next_symbol == '/':
                self.advance(), self.advance()
                comment = ''
                while self.symbol != '\n' and self:
                    comment += self.advance()
                # yield VKVToken.COMMENT, comment
            elif symbol == '{':
                yield VKVToken.LBRACE, self.advance()
            elif symbol == '}':
                yield VKVToken.RBRACE, self.advance()
            elif self._is_valid_symbol(symbol):
                string = self.read_simple_string(terminators=' \t\n')
                if string:
                    yield VKVToken.STRING, string
            elif symbol in '\'"':
                if self.next_symbol in '\'"':
                    self.advance(), self.advance()
                    yield VKVToken.STRING, ""
                    continue
                string = self.read_quoted_string()
                if string:
                    yield VKVToken.STRING, string
            elif symbol == '$':
                self.advance()
                string = self.read_simple_string(terminators=' \t\n')
                if string:
                    yield VKVToken.STRING, "$" + string
            elif self.symbol in "<>=":
                expr = ""
                while True:
                    if self.symbol not in "<>=":
                        break
                    expr += self.symbol
                if expr:
                    yield VKVToken.EXPRESSION, expr
            elif symbol == '%':
                self.advance()
                string = self.read_simple_string(terminators=' \t\n')
                if string:
                    yield VKVToken.STRING, "%" + string
            elif symbol == '[':
                yield VKVToken.LBRACKET, self.advance()
            elif symbol == ']':
                yield VKVToken.RBRACKET, self.advance()
            elif symbol.isprintable():
                warnings.warn(f'Unknown symbol {self.advance()!r} at {self._line}:{self._column}')
                continue
            else:
                raise KVLexerException(
                    f'Unknown symbol {symbol!r} in {self.buffer_name!r} at {self._line}:{self._column}')
        yield VKVToken.EOF, None

    def __bool__(self):
        return self._offset < len(self.buffer)


class ValveKeyValueParser:
    def __init__(self, path: Union[Path, str] = None, buffer_and_name: Tuple[str, str] = None, self_recover=False,
                 array_of_blocks=False):
        if path is not None:
            self._path = Path(path)
            with self._path.open() as f:
                self._lexer = ValveKeyValueLexer(f.read(), str(self._path))
        elif buffer_and_name is not None:
            self._lexer = ValveKeyValueLexer(*buffer_and_name)
            self._path = buffer_and_name[1]
        self._tokens = self._lexer.lex()
        self._last_peek = None
        self._self_recover = self_recover
        self._array_of_blocks = array_of_blocks

        self._tree = []

    @property
    def tree(self):
        if self._array_of_blocks:
            return [KVDataProxy(s) for s in self._tree]
        return KVDataProxy(self._tree)

    def peek(self):
        if self._last_peek is None:
            self._last_peek = next(self._tokens)
        return self._last_peek

    def advance(self):
        if self._last_peek is not None:
            ret = self._last_peek
            self._last_peek = None
            return ret
        return next(self._tokens)

    def expect(self, token_type):
        token, value = self.peek()
        if token == token_type:
            self.advance()
            return value
        else:
            if self._self_recover:
                warnings.warn(f"Trying to recover from unexpected token {token}:{value!r}, expected {token_type} "
                              f"in {self._path!r} at {self._lexer.line}:{self._lexer.column}")
                while not self.match(VKVToken.NEWLINE):
                    if self.match(VKVToken.EOF):
                        break
                    self.advance()
                pass
            else:
                raise KVParserException(f"Unexpected token {token}:{value!r}, expected {token_type}"
                                        f"in {self._path!r} at {self._lexer.line}:{self._lexer.column}")

    def match(self, token_type, consume=False):
        token, value = self.peek()
        if token == token_type:
            if consume:
                self.advance()
            return True
        return False

    def _skip_newlines(self):
        while self.match(VKVToken.NEWLINE):
            self.advance()

    def _parse_expression(self):
        expr = []
        while not self.match(VKVToken.RBRACKET, consume=False):
            expr.append(self.advance()[1])
        self.expect(VKVToken.RBRACKET)
        return expr

    def parse(self):
        node_stack = [self._tree]
        while self._lexer:
            self._skip_newlines()
            if self.match(VKVToken.STRING) or self.match(VKVToken.EXPRESSION):
                key = self.advance()[1]
                self._skip_newlines()
                if self.match(VKVToken.LBRACE, True):
                    new_tree_node = []
                    node_stack[-1].append((key.lower(), new_tree_node))
                    node_stack.append(new_tree_node)
                elif self.match(VKVToken.STRING):
                    value = self.advance()
                    if self.match(VKVToken.LBRACKET, True):
                        condition = self._parse_expression()
                        node_stack[-1].append((key.lower(), (value[1], condition)))
                    else:
                        node_stack[-1].append((key.lower(), value[1]))
                    self.expect(VKVToken.NEWLINE)
            elif self._array_of_blocks and self.match(VKVToken.LBRACE, True):
                new_tree_node = []
                node_stack[-1].append(new_tree_node)
                node_stack.append(new_tree_node)
            elif self.match(VKVToken.RBRACE, True):
                node_stack.pop(-1)
            elif self.match(VKVToken.EOF):
                break
            else:
                token, value = self.peek()
                raise KVParserException(
                    f"Unexpected token {token}:\"{value}\" in {self._path} at {self._lexer.line}:{self._lexer.column}")
