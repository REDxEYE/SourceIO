import io
import typing
import uuid
from pathlib import Path

from SourceIO.library.utils import TinyPath


class Lexer:
    def __init__(self, stream: typing.TextIO, filename: str):
        self.stream = stream
        self.name = filename
        self.pos = (0, 1, 1)
        self.cur = (0, 1, 1)
        self.val = None
        self.hdr = False
        self._advance()

    def next(self):
        while True:
            val = self.val
            pos = self.cur

            if val.isspace():
                self._advance()
                continue

            if val == '<':
                if self._advance() != '!' or self._advance() != '-' or self._advance() != '-':
                    self._report(pos, 'Truncated header opening quote')

                self._advance()
                self.hdr = True

                return pos, 'lhdr', None

            if val == '-' and self.hdr:
                if self._advance() != '-' or self._advance() != '>':
                    self._report(pos, 'Truncated header closing quote')

                self._advance()
                self.hdr = False

                return pos, 'rhdr', None

            if val == '{' and self.hdr:
                buf = ''

                val = self._advance()
                pos = self.pos

                while val != '}':
                    if not (val.isalnum() or val == '-'):
                        self._report(pos, 'Non-hex character in UUID identifier')

                    buf += val
                    val = self._advance()

                self._advance()

                return pos, 'uuid', uuid.UUID(buf)

            if val.isalpha() or val == '_':
                buf = ''

                while val.isalpha() or val.isdigit() or val == '_':
                    buf += val
                    val = self._advance()

                return pos, 'symbol', buf

            if val.isdigit() or val == '.' or val == '-':
                num = 0
                mag = 0
                sig = 1
                dot = False

                while val.isdigit() or val == '.' or val == '-':
                    if val == '.':
                        val = self._advance()
                        dot = True
                        continue

                    if val == '-':
                        sig = -1
                        val = self._advance()
                        continue

                    num = num * 10 + int(val)

                    if dot:
                        mag -= 1

                    val = self._advance()

                return pos, 'number', num * 10 ** mag * sig

            if val == '"':
                buf = ''
                val = self._advance()

                def read(match_newline=False):
                    nonlocal val, buf, pos

                    while val != '"':
                        if val is None or (match_newline and val == '\n'):
                            self._report(pos, 'Truncated string closing quote')

                        buf += val
                        val = self._advance()
                        pos = self.pos

                read(match_newline=True)

                if self._advance() == '"' and len(buf) == 0:
                    val = self._advance()

                    read(match_newline=False)

                    if self._advance() != '"' or self._advance() != '"':
                        self._report(pos, 'Truncated multi-line closing quote')

                    self._advance()

                return pos, 'string', buf

            if val == '{':
                self._advance()
                return pos, 'lbrace', None

            if val == '}':
                self._advance()
                return pos, 'rbrace', None

            if val == '[':
                self._advance()
                return pos, 'lbracket', None

            if val == ']':
                self._advance()
                return pos, 'rbracket', None

            if val == '=':
                self._advance()
                return pos, 'assign', None

            if val == ',':
                self._advance()
                return pos, 'comma', None

            if val == ':':
                self._advance()
                return pos, 'colon', None

            if val == '':
                return pos, 'eof', None

            self._report(pos, f'Invalid character: \'{val}\'')

    def _advance(self):
        pos_old, row_old, col_old = self.pos
        pos_new, row_new, col_new = self.pos
        val = self.stream.read(1)

        if val != '':
            pos_new += 1
            col_new += 1

        if val == '\n':
            row_new += 1
            col_new = 1

        self.pos = pos_new, row_new, col_new
        self.cur = pos_old, row_old, col_old
        self.val = val

        return val

    def _report(self, pos, msg: str):
        raise ValueError(f'{self.name}:{pos[1]}:{pos[2]}: {msg}')


class Parser(Lexer):
    def __init__(self, stream: typing.TextIO, filename: str):
        super().__init__(stream, filename)
        self.tok = None
        self.next()

    def parse_file(self):
        return self._parse_header(), self._parse()

    def _parse(self):
        pos, tag, value = self.tok

        if tag == 'lbrace':
            self.next()
            return self._parse_dict()

        if tag == 'lbracket':
            self.next()
            return self._parse_list()

        if tag == 'string':
            self.next()
            return value

        if tag == 'number':
            self.next()
            return value

        if tag == 'symbol':
            if value == 'true':
                self.next()
                return True

            if value == 'false':
                self.next()
                return False

        self._report(pos, f'Unexpected token \'{tag}\'')

    def _parse_header(self):
        self._expect('lhdr')
        name = self._expect('symbol')[2]

        assert self._expect('symbol')[2] == 'encoding'
        self._expect('colon')
        encoding_name = self._expect('symbol')[2]
        self._expect('colon')
        assert self._expect('symbol')[2] == 'version'
        encoding_version = self._expect('uuid')[2]

        assert self._expect('symbol')[2] == 'format'
        self._expect('colon')
        format_name = self._expect('symbol')[2]
        self._expect('colon')
        assert self._expect('symbol')[2] == 'version'
        format_version = self._expect('uuid')[2]

        self._expect('rhdr')

        return name, (encoding_name, encoding_version), (format_name, format_version)

    def _parse_dict(self):
        items = {}

        while not self._match('rbrace'):
            name = self._expect('symbol')[2]
            self._expect('assign')
            data = self._parse()

            items[name] = data

        return items

    def _parse_list(self):
        items = []

        while not self._match('rbracket'):
            items.append(self._parse())
            self._match('comma')

        return items

    def _match(self, tag: str, consume: bool = True):
        tok = self.tok

        if tok[1] != tag:
            return None

        if consume:
            self.next()

        return tok

    def _expect(self, tag: str, consume: bool = True):
        tok = self._match(tag, consume)

        if not tok:
            self._report(self.tok[0], f'Expected token of type \'{tag}\', found \'{self.tok[1]}\'')

        return tok

    def next(self):
        self.tok = super().next()


class Writer:
    def __init__(self, stream: typing.TextIO):
        self.stream = stream

    def write_header(self, name, enc, fmt):
        self.print(f'<!-- {name} encoding:{enc[0]}:version{{{enc[1]}}} format:{fmt[0]}:version{{{fmt[1]}}} -->', 0)

    def write(self, value, indentation: int, append_newline: bool):
        if isinstance(value, dict):
            self.write_dict(value, indentation, append_newline)
        elif isinstance(value, list):
            self.write_list(value, indentation, append_newline)
        elif isinstance(value, (TinyPath, Path)):
            self.write_string(value.as_posix().replace('\\', '/'), indentation, append_newline)
        elif isinstance(value, str):
            self.write_string(value, indentation, append_newline)
        elif isinstance(value, bool):
            self.write_bool(value, indentation, append_newline)
        elif isinstance(value, (int, float)):
            self.write_number(value, indentation, append_newline)
        else:
            raise TypeError(f'Invalid type: {value.__class__}')

    def write_dict(self, items: dict, indentation: int, append_newline: bool):
        self.print('{', indentation, True)

        for key, value in items.items():
            if isinstance(value, (dict, list)):
                self.print(f'{key} = ', indentation + 1, True)
                self.write(value, indentation + 1, True)
            else:
                self.print(f'{key} = ', indentation + 1, False)
                self.write(value, 0, True)

        self.print('}', indentation, append_newline)

    def write_list(self, items: list, indentation: int, append_newline: bool):
        self.print('[', indentation, True)

        for index, value in enumerate(items):
            self.write(value, indentation + 1, False)

            if index < len(items) - 1:
                self.print(',', 0, False)

            self.print('', 0, True)

        self.print(']', indentation, append_newline)

    def write_string(self, value: str, indentation: int, append_newline: bool):
        value = f'"{value}"' if value.find('\n') < 0 else f'"""{value}"""'
        self.print(value, indentation, append_newline)

    def write_number(self, value: int, indentation: int, append_newline: bool):
        value = str(value if isinstance(value, int) else round(value, 6))
        self.print(value, indentation, append_newline)

    def write_bool(self, value: bool, indentation: int, append_newline: bool):
        self.print('true' if value else 'false', indentation, append_newline)

    def print(self, value, indent: int, append_newline: bool = True):
        self.stream.write('\t' * indent + value)

        if append_newline:
            self.stream.write('\n')


class KeyValues:
    @staticmethod
    def read_file(filename: str):
        return KeyValues.read_data(open(filename, 'r', encoding='latin', errors='replace'), filename)

    @staticmethod
    def read_data(stream: typing.TextIO, filename: str = '<input>'):
        return Parser(stream, filename).parse_file()

    @staticmethod
    def dump(name: str, enc: tuple, fmt: tuple, data, out: typing.TextIO):
        writer = Writer(out)
        writer.write_header(name, enc, fmt)
        writer.write(data, 0, False)
        return writer

    @staticmethod
    def dump_str(name: str, enc: tuple, fmt: tuple, data):
        buf = io.StringIO()
        KeyValues.dump(name, enc, fmt, data, buf)
        buf.seek(0)
        return buf.read()
