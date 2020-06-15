import uuid


class Lexer:
    def __init__(self, path: str):
        self.path = path
        self.buf = open(path, 'r', encoding='utf-8', newline='\n')
        self.pos = 0
        self.row = 1
        self.col = 1
        self.cur = None
        self.hdr = False
        self.advance()

    def next(self):
        while True:
            val = self.cur[0]
            pos = self.cur[1:]

            if val.isspace():
                self.advance()
                continue

            if val == '<':
                if self.advance() != '!' or self.advance() != '-' or self.advance() != '-':
                    self.report(pos, 'Truncated header opening quote')

                self.advance()
                self.hdr = True

                return pos, 'lhdr', None

            if val == '-' and self.hdr:
                if self.advance() != '-' or self.advance() != '>':
                    self.report(pos, 'Truncated header closing quote')

                self.advance()
                self.hdr = False

                return pos, 'rhdr', None

            if val == '{' and self.hdr:
                buf = ''

                val = self.advance()
                pos = self.cur[1:]

                while val != '}':
                    if not (val.isalnum() or val == '-'):
                        self.report(pos, 'Non-hex character in UUID identifier')

                    buf += val
                    val = self.advance()

                self.advance()

                return pos, 'uuid', uuid.UUID(buf)

            if val.isalpha() or val == '_':
                buf = ''

                while val.isalpha() or val.isdigit() or val == '_':
                    buf += val
                    val = self.advance()

                return pos, 'symbol', buf

            if val.isdigit() or val == '.' or val == '-':
                num = 0
                mag = 0
                sig = 1
                dot = False

                while val.isdigit() or val == '.' or val == '-':
                    if val == '.':
                        val = self.advance()
                        dot = True
                        continue

                    if val == '-':
                        sig = -1
                        val = self.advance()
                        continue

                    num = num * 10 + int(val)

                    if dot:
                        mag -= 1

                    val = self.advance()

                return pos, 'number', num * 10 ** mag * sig

            if val == '"':
                buf = ''
                val = self.advance()

                def read(match_newline=False):
                    nonlocal val, buf, pos

                    while val != '"':
                        if val is None or (match_newline and val == '\n'):
                            self.report(pos, 'Truncated string closing quote')

                        buf += val
                        val = self.advance()
                        pos = self.cur[1:]

                read(match_newline=True)

                if self.advance() == '"' and len(buf) == 0:
                    val = self.advance()

                    read(match_newline=False)

                    if self.advance() != '"' or self.advance() != '"':
                        self.report(pos, 'Truncated multi-line closing quote')

                    self.advance()

                return pos, 'string', buf

            if val == '{':
                self.advance()
                return pos, 'lbrace', None

            if val == '}':
                self.advance()
                return pos, 'rbrace', None

            if val == '[':
                self.advance()
                return pos, 'lbracket', None

            if val == ']':
                self.advance()
                return pos, 'rbracket', None

            if val == '=':
                self.advance()
                return pos, 'assign', None

            if val == ',':
                self.advance()
                return pos, 'comma', None

            if val == ':':
                self.advance()
                return pos, 'colon', None

            if val == '':
                return pos, 'eof', None

            self.report(pos, f'Invalid character: \'{val}\'')

    def advance(self):
        pos = self.pos
        row = self.row
        col = self.col
        val = self.buf.read(1)

        if val != '':
            self.pos += 1
            self.col += 1

        if val == '\n':
            self.row += 1
            self.col = 1

        self.cur = val, pos, row, col

        return val

    def report(self, pos, msg: str):
        raise ValueError(f'{self.path}:{pos[1]}:{pos[2]}: {msg}')


class Parser(Lexer):
    def __init__(self, path: str):
        super().__init__(path)
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

        self.report(pos, f'Unexpected token \'{tag}\'')

    def _parse_header(self):
        self.expect('lhdr')
        name = self.expect('symbol')[2]

        assert self.expect('symbol')[2] == 'encoding'
        self.expect('colon')
        encoding_name = self.expect('symbol')[2]
        self.expect('colon')
        assert self.expect('symbol')[2] == 'version'
        encoding_version = self.expect('uuid')[2]

        assert self.expect('symbol')[2] == 'format'
        self.expect('colon')
        format_name = self.expect('symbol')[2]
        self.expect('colon')
        assert self.expect('symbol')[2] == 'version'
        format_version = self.expect('uuid')[2]

        self.expect('rhdr')

        return name, (encoding_name, encoding_version), (format_name, format_version)

    def _parse_dict(self):
        items = {}

        while not self.match('rbrace'):
            name = self.expect('symbol')[2]
            self.expect('assign')
            data = self._parse()

            items[name] = data

        return items

    def _parse_list(self):
        items = []

        while not self.match('rbracket'):
            items.append(self._parse())
            self.match('comma')

        return items

    def match(self, tag: str, consume: bool = True):
        tok = self.tok

        if tok[1] != tag:
            return None

        if consume:
            self.next()

        return tok

    def expect(self, tag: str, consume: bool = True):
        tok = self.match(tag, consume)

        if not tok:
            self.report(self.tok[0], f'Expected token of type \'{tag}\', found \'{self.tok[1]}\'')

        return tok

    def next(self):
        self.tok = super().next()


class Writer:
    def __init__(self, header, data, output):
        self.output = output
        self.dump_header(*header)
        self.dump(data, 0, True)

    def dump_header(self, name, e, f):
        self.print(f'<!-- {name} encoding:{e[0]}:version{{{e[1]}}} format:{f[0]}:version{{{f[1]}}} -->', 0, True)

    def dump(self, value, indent: int, newline: bool):
        if isinstance(value, dict):
            self.print('{', indent, True)
            for k, v in value.items():
                if isinstance(v, dict) or isinstance(v, list):
                    self.print(f'{k} = ', indent + 1, True)
                    self.dump(v, indent + 1, True)
                else:
                    self.print(f'{k} = ', indent + 1, False)
                    self.dump(v, 0, True)
            self.print('}', indent, newline)
        elif isinstance(value, list):
            self.print('[', indent, newline)
            for index, v in enumerate(value):
                self.dump(v, indent + 1, False)
                if index < len(value) - 1:
                    self.print(',', 0, False)
                self.print('', 0, True)
            self.print(']', indent, newline)
        elif isinstance(value, str):
            if value.find('\n') < 0:
                self.print(f'"{value}"', indent, newline)
            else:
                self.print(f'"""{value}"""', indent, newline)
        elif isinstance(value, bool):
            self.print(str(value).lower(), indent, newline)
        elif isinstance(value, int):
            self.print(str(value), indent, newline)
        elif isinstance(value, float):
            self.print(str(round(value, 6)), indent, newline)
        else:
            raise TypeError(f'Invalid type: {value.__class__}')

    def print(self, value, indent, newline):
        self.output.write('    ' * indent + value)

        if newline:
            self.output.write('\n')


class KeyValues:
    @staticmethod
    def read(path: str):
        return Parser(path).parse_file()

    @staticmethod
    def dump(header, data, out):
        return Writer(header, data, out)


def main():
    # path = r'C:/Users/ShadelessFox/Downloads/Telegram Desktop/medic.vmdl'
    # data = KeyValues.read(path)
    #
    # print(data)

    import sys

    path = r'C:\Users\ShadelessFox\Downloads\Telegram Desktop\charizardfemale.vmdl'

    KeyValues.dump(*KeyValues.read(path), sys.stdout)


if __name__ == '__main__':
    main()