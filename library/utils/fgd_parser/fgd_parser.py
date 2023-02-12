from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

from ...shared.content_providers.content_manager import ContentManager
from ...utils.fgd_parser.fgd_classes import FGDEntity


class FGDLexerException(Exception):
    pass


class FGDParserException(Exception):
    pass


class FGDToken(Enum):
    STRING = "String literal"
    NUMERIC = "Numeric literal"
    IDENTIFIER = "Identifier literal"
    KEYWORD = "Keyword literal"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    LBRACE = "{"
    RBRACE = "}"
    EQUALS = "="
    COLON = ":"
    PLUS = "+"
    MINUS = "-"
    COMMA = ","
    DOT = "."
    FSLASH = "/"
    BSLASH = "\\"
    EOF = "End of file"


class FGDLexer:

    def __init__(self, buffer: str, buffer_name: str = '<memory>'):
        self.buffer = buffer
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

    def lex(self):
        while self._offset < len(self.buffer):
            if self.symbol == '"':
                self.advance()
                string_buffer = ""
                while True:
                    if self.symbol == '"':
                        self.advance()
                        break
                    string_buffer += self.advance()
                yield FGDToken.STRING, string_buffer
            elif self.symbol.isspace():
                self.advance()
            elif self.symbol.isdigit() or (self.symbol == '-' and self.next_symbol.isdigit()):
                num_buffer = ""
                while self.symbol.isdigit() or (self.symbol == '-' and self.next_symbol.isdigit()):
                    num_buffer += self.advance()
                yield FGDToken.NUMERIC, int(num_buffer)
            elif self.symbol.isidentifier() or self.symbol == '@':
                string_buffer = self.advance()
                while True:
                    if self.symbol.isspace() or not (self.symbol.isidentifier() or self.symbol.isdigit()):
                        break
                    string_buffer += self.advance()
                if string_buffer.startswith('@'):
                    yield FGDToken.KEYWORD, string_buffer
                else:
                    yield FGDToken.IDENTIFIER, string_buffer
            elif self.symbol == '/' and self.next_symbol == '/':
                while True:
                    if self.symbol in '\n\r':
                        break
                    self.advance()
            elif self.symbol == '/' and self.next_symbol == '*':
                self.skip(2)
                while True:
                    if not self.symbol:
                        raise FGDLexerException(f"Unexpected EOF in {self.buffer_name}")
                    if self.symbol == '*' and self.next_symbol == '/':
                        self.skip(2)
                        break
                    self.advance()
            elif self.symbol == '{':
                yield FGDToken.LBRACE, self.advance()
            elif self.symbol == '}':
                yield FGDToken.RBRACE, self.advance()
            elif self.symbol == '(':
                yield FGDToken.LPAREN, self.advance()
            elif self.symbol == ')':
                yield FGDToken.RPAREN, self.advance()
            elif self.symbol == '[':
                yield FGDToken.LBRACKET, self.advance()
            elif self.symbol == ']':
                yield FGDToken.RBRACKET, self.advance()
            elif self.symbol == '=':
                yield FGDToken.EQUALS, self.advance()
            elif self.symbol == ':':
                yield FGDToken.COLON, self.advance()
            elif self.symbol == '-':
                yield FGDToken.MINUS, self.advance()
            elif self.symbol == '+':
                yield FGDToken.PLUS, self.advance()
            elif self.symbol == ',':
                yield FGDToken.COMMA, self.advance()
            elif self.symbol == '.':
                yield FGDToken.DOT, self.advance()
            elif self.symbol == '/':
                yield FGDToken.FSLASH, self.advance()
            elif self.symbol == '\\':
                yield FGDToken.BSLASH, self.advance()
            else:
                raise FGDLexerException(
                    f'Unknown symbol "{self.symbol}" in "{self.buffer_name}" at {self._line}:{self._column}')
        yield FGDToken.EOF, None

    def __bool__(self):
        return self._offset < len(self.buffer)


class FGDParser:
    def __init__(self, path: Union[Path, str] = None, buffer_and_name: Tuple[str, str] = None):
        if path is not None:
            self._path = Path(path)
            with self._path.open() as f:
                self._lexer = FGDLexer(f.read(), str(self._path))
        elif buffer_and_name is not None:
            self._lexer = FGDLexer(*buffer_and_name)
            self._path = buffer_and_name[1]
        self._tokens = self._lexer.lex()
        self._last_peek = None

        self.classes: List[FGDEntity] = []
        self.excludes = []
        self.pragmas = {}
        self.includes = []
        self.entity_groups = []
        self.vis_groups = {}

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
            raise FGDParserException(f"Unexpected token {token_type}, got {token}:\"{value}\" "
                                     f"in {self._path} at {self._lexer.line}:{self._lexer.column}")

    def match(self, token_type, consume=False):
        token, value = self.peek()
        if token == token_type:
            if consume:
                self.advance()
            return True
        return False

    def parse(self):
        while self._lexer:
            if self.match(FGDToken.KEYWORD):
                _, value = self.advance()
                if value == '@mapsize':
                    self._parse_mapsize()
                elif value.lower() == "@include":
                    self._parse_include()
                elif value.lower() == '@exclude':
                    self.excludes.append(self.expect(FGDToken.IDENTIFIER))
                elif value.lower() == '@entitygroup':
                    self._parse_entity_group()
                elif value.lower() == '@materialexclusion':
                    self._parse_material_exclusion()
                elif value.lower() == '@autovisgroup':
                    self._parse_autovis_group()
                elif value.startswith('@') and value.lower().endswith("class"):
                    self._parse_baseclass(value[1:])
            elif self.match(FGDToken.EOF):
                break
            else:
                token, value = self.peek()
                raise FGDParserException(
                    f"Unexpected token {token}:\"{value}\" in {self._path} at {self._lexer.line}:{self._lexer.column}")

    def _parse_include(self):
        include = self.expect(FGDToken.STRING)
        file = ContentManager().find_file(include)
        if file is not None:
            parsed_include = FGDParser(buffer_and_name=(file.read().decode("ascii"), include))
            parsed_include.parse()
            self.classes.extend(parsed_include.classes)
            self.pragmas.update(parsed_include.pragmas)
            self.excludes.extend(parsed_include.excludes)
            self.entity_groups.extend(parsed_include.entity_groups)
            self.includes.append(include)

    def _parse_mapsize(self):
        self.expect(FGDToken.LPAREN)
        max_x = self.expect(FGDToken.NUMERIC)
        self.expect(FGDToken.COMMA)
        max_y = self.expect(FGDToken.NUMERIC)
        self.expect(FGDToken.RPAREN)
        self.pragmas['mapsize'] = (max_x, max_y)

    def _parse_entity_group(self):
        group = {'name': self.expect(FGDToken.STRING), 'meta': {}}
        if self.match(FGDToken.LBRACE):
            self.advance()
            while not self.match(FGDToken.RBRACE):
                key = self.expect(FGDToken.IDENTIFIER)
                self.expect(FGDToken.EQUALS)
                value = self.expect(FGDToken.IDENTIFIER)
                group['meta'][key] = value
            self.advance()
        self.entity_groups.append(group)

    def _find_parent_class(self, class_name):
        for cls in self.classes:
            if cls.name == class_name:
                return cls
        return None

    def _parse_baseclass(self, class_type):

        definitions = []
        if self.match(FGDToken.IDENTIFIER):
            while not self.match(FGDToken.EQUALS):
                meta_prop_type = self.expect(FGDToken.IDENTIFIER)
                if meta_prop_type == 'base':
                    definitions.append((meta_prop_type, self._parse_bases()))

                elif meta_prop_type == 'color':
                    self.expect(FGDToken.LPAREN)
                    r = self.expect(FGDToken.NUMERIC)
                    g = self.expect(FGDToken.NUMERIC)
                    b = self.expect(FGDToken.NUMERIC)
                    self.expect(FGDToken.RPAREN)
                    definitions.append((meta_prop_type, (r, g, b)))
                elif meta_prop_type == 'metadata':
                    meta = {}
                    self.expect(FGDToken.LBRACE)
                    while not self.match(FGDToken.RBRACE):
                        key = self.expect(FGDToken.IDENTIFIER)
                        self.expect(FGDToken.EQUALS)
                        value = self.expect(FGDToken.STRING)
                        meta[key] = value
                    self.expect(FGDToken.RBRACE)
                    definitions.append((meta_prop_type, meta))
                else:
                    if self.match(FGDToken.LPAREN):
                        self.expect(FGDToken.LPAREN)
                        meta = []
                        while not self.match(FGDToken.RPAREN):
                            meta.append(self.advance()[1])
                            if self.match(FGDToken.COMMA):
                                self.advance()
                        self.expect(FGDToken.RPAREN)
                        definitions.append((meta_prop_type, meta))
                    else:
                        definitions.append((meta_prop_type, True))

        self.expect(FGDToken.EQUALS)
        class_name = self.expect(FGDToken.IDENTIFIER)

        doc = None
        if self.match(FGDToken.COLON, True):
            if self.match(FGDToken.STRING):
                doc = self._parse_joined_string()

        self.expect(FGDToken.LBRACKET)
        io = []
        props = []
        while self.match(FGDToken.IDENTIFIER):
            token, ident = self.peek()
            if token == FGDToken.IDENTIFIER and ident in ['input', 'output']:
                self._parse_class_io(io)
            else:
                self._parse_class_param(props)

        self.expect(FGDToken.RBRACKET)
        if class_type == 'OverrideClass':
            class_obj = self._find_parent_class(class_name)
            if class_obj is None:
                class_obj = FGDEntity(class_type, class_name, definitions, doc, props, io)
            else:
                class_obj.override(definitions, doc, props, io)
        else:
            class_obj = FGDEntity(class_type, class_name, definitions, doc, props, io)
        self.classes.append(class_obj)

    def _parse_fully_qualified_identifier(self):
        p1 = self.expect(FGDToken.IDENTIFIER)
        while self.match(FGDToken.DOT, True):
            p1 += '.' + self.expect(FGDToken.IDENTIFIER)
        return p1

    def _parse_complex_type(self):
        p1 = self.expect(FGDToken.IDENTIFIER)
        while self.match(FGDToken.COLON, True):
            p1 += ':' + self.expect(FGDToken.IDENTIFIER)
        return p1

    def _parse_joined_string(self):
        p1 = self.expect(FGDToken.STRING)
        while self.match(FGDToken.PLUS, True):
            if self.match(FGDToken.STRING):
                p1 += self.expect(FGDToken.STRING)
            else:
                break
        return p1

    def _parse_bases(self):
        bases = []
        self.expect(FGDToken.LPAREN)
        while True:
            bases.append(self.expect(FGDToken.IDENTIFIER))
            if not self.match(FGDToken.COMMA, True):
                break
        self.expect(FGDToken.RPAREN)
        return bases

    def _parse_class_io(self, storage: list):
        io_type = self.expect(FGDToken.IDENTIFIER)
        name = self.expect(FGDToken.IDENTIFIER)
        self.expect(FGDToken.LPAREN)
        args = []
        while not self.match(FGDToken.RPAREN):
            args.append(self.expect(FGDToken.IDENTIFIER))
        self.expect(FGDToken.RPAREN)
        if self.match(FGDToken.COLON):
            self.advance()
            doc_str = self._parse_joined_string() if self.match(FGDToken.STRING) else None
        else:
            doc_str = None
        storage.append({'name': name, 'type': io_type, 'args': args, 'doc': doc_str})

    def _parse_class_param_meta(self):
        meta = {}
        while True:
            meta_name = self.expect(FGDToken.IDENTIFIER)
            if self.match(FGDToken.EQUALS, True):
                value = self.expect(FGDToken.STRING)
            else:
                value = True
            meta[meta_name] = value
            if not self.match(FGDToken.COMMA, True):
                break
        self.expect(FGDToken.RBRACKET)
        return meta

    def _parse_class_param(self, storage):
        prop = {'meta': {}}
        name = self._parse_fully_qualified_identifier()
        self.expect(FGDToken.LPAREN)
        param_type = self._parse_complex_type()
        self.expect(FGDToken.RPAREN)
        if self.match(FGDToken.IDENTIFIER) and self.peek()[1] in ['report', 'readonly']:
            prop['meta'][self.expect(FGDToken.IDENTIFIER)] = True
        if self.match(FGDToken.LBRACKET, True):
            prop['meta'].update(self._parse_class_param_meta())

        data = []
        if self.match(FGDToken.COLON):
            while True:
                if not (self.match(FGDToken.STRING) or self.match(FGDToken.NUMERIC) or self.match(FGDToken.COLON)):
                    break
                self.expect(FGDToken.COLON)
                value = None  # No value, just 2 ":" symbols
                if not self.match(FGDToken.COLON):  # We have value
                    if self.match(FGDToken.STRING):  # String can be split by + signs, so we need to account for it
                        value = self._parse_joined_string()
                    elif self.match(FGDToken.NUMERIC):  # any other token
                        value = self.expect(FGDToken.NUMERIC)
                data.append(value)
            if len(data) == 3:
                prop['display_name'], prop['default'], prop['doc'] = data
            elif len(data) == 2:
                prop['display_name'], prop['default'] = data
            elif len(data) == 1:
                prop['display_name'] = data[0]
            else:
                print(data)

        if self.match(FGDToken.EQUALS) and "choices" in param_type.lower():
            # parse choices
            self.advance()
            self.expect(FGDToken.LBRACKET)
            choices = {}
            while not self.match(FGDToken.RBRACKET):
                choice_name = self.expect(FGDToken.STRING) if self.match(FGDToken.STRING) else self.expect(
                    FGDToken.NUMERIC)
                self.expect(FGDToken.COLON)
                value = self.expect(FGDToken.STRING)
                choices[choice_name] = value
            self.expect(FGDToken.RBRACKET)
            prop['choices'] = choices
        elif self.match(FGDToken.EQUALS) and 'flags' in param_type.lower():
            # parse flags
            self.advance()
            self.expect(FGDToken.LBRACKET)
            flags = {}
            while not self.match(FGDToken.RBRACKET):
                mask = self.expect(FGDToken.NUMERIC)
                self.expect(FGDToken.COLON)
                flag_name = self.expect(FGDToken.STRING)
                self.expect(FGDToken.COLON)
                default = self.expect(FGDToken.NUMERIC)

                flags[flag_name] = (mask, default)
            self.expect(FGDToken.RBRACKET)
            prop['flags'] = flags
        elif self.match(FGDToken.EQUALS) and 'tag_list' in param_type.lower():
            # parse flags
            self.advance()
            self.expect(FGDToken.LBRACKET)
            flags = {}
            while not self.match(FGDToken.RBRACKET):
                mask = self.expect(FGDToken.STRING)
                self.expect(FGDToken.COLON)
                tag_name = self.expect(FGDToken.STRING)
                self.expect(FGDToken.COLON)
                default = self.expect(FGDToken.NUMERIC)

                flags[tag_name] = (mask, default)
            self.expect(FGDToken.RBRACKET)
            prop['tag_list'] = flags

        prop['name'] = name
        prop['type'] = param_type
        storage.append(prop)

    def _parse_material_exclusion(self):
        self.expect(FGDToken.LBRACKET)
        m = self.pragmas['material_exclusion'] = []
        while not self.match(FGDToken.RBRACKET, True):
            mat = self.expect(FGDToken.STRING)
            m.append(mat)

    def _parse_autovis_group(self):
        self.expect(FGDToken.EQUALS)
        name = self.expect(FGDToken.STRING)
        self.expect(FGDToken.LBRACKET)
        vis_group = self.vis_groups[name] = {}
        while not self.match(FGDToken.RBRACKET, True):
            sub_name = self.expect(FGDToken.STRING)
            vis_list = vis_group[sub_name] = []
            self.expect(FGDToken.LBRACKET)
            while not self.match(FGDToken.RBRACKET, True):
                ent_name = self.expect(FGDToken.STRING)
                vis_list.append(ent_name)


if __name__ == '__main__':
    test_file = Path(r"F:\SteamLibrary\steamapps\common\Half-Life Alyx\game\hlvr\hlvr.fgd")
    # test_file = Path(r"H:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\bin\swarm.fgd")
    # test_file = Path(r"H:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\bin\base.fgd")
    ContentManager().scan_for_content(test_file)
    parser = FGDParser(test_file)
    parser.parse()
    for cls in parser.classes:
        print(cls.parser_code())
    pass
