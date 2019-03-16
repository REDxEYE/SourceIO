import re
import shlex
from pathlib import Path


def fix_slashes(path, path_sep='/'):
    try:
        return path.replace('\\', path_sep).replace('/', path_sep)
    except TypeError:
        return path


def encode_quotes(string):
    """
    Return a string with single and double quotes escaped, keyvalues style
    """
    return string.replace('"', '\\"').replace("'", "\\'")


def decode_quotes(string):
    """
    Return a string with escaped single and double quotes without escape characters.
    """
    return string.replace('\\"', '"').replace("\\'", "'")


def stripcomments(lines):
    """
    Strips all C++ style block and line comments from a list of lines using RegEx
    """

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    out_lines = []
    for line in lines:
        line = re.sub(pattern, replacer, line).strip()
        out_lines.append(line)

    return out_lines


def parse_line(line):
    """
    Line parser that extracts key value pairs from a line and returns a list of the tokens with escaped quotes.
    """
    # Fix any trailing slashes that are escaping quotes
    if line.endswith('\\"'):
        line = line.rsplit('\\"', 1)
        line = '\\\\"'.join(line)
    elif line.endswith("\\'"):
        line = line.rsplit("\\'", 1)
        line = "\\\\'".join(line)

    lex = shlex.shlex(line, posix=True)
    lex.escapedquotes = '\"\''
    lex.whitespace = ' \n\t='
    lex.wordchars += '|.:/\\+*%$'  # Do not split on these chars
    # Escape all quotes in result
    tokens = [encode_quotes(token) for token in lex]

    if len(tokens) == 1:
        # Key with no value gets empty string as value
        tokens.append('')
    elif len(tokens) > 2:
        # Multiple value tokens, invalid
        raise TypeError
    vals = tokens[1].split(" ")
    if vals and len(vals) > 1:
        a = [val.isnumeric() for val in vals]
        if all(a):
            tokens[1] = list(map(int, vals))
    return tokens


def get_mod_path(path: Path):
    """

    :rtype: Path
    """
    org = path
    if 'models' in path.parts or 'materials' in path.parts:
        while len(path.parts) > 1:
            path = path.parent
            if path.parts[-1] == 'models' and path.parts[-2] == 'materials':
                return path.parent.parent
            if path.parts[-1] == 'models':
                return path.parent
            if path.parts[-1] == 'materials':
                return path.parent
            if len(path.parts) == 1:
                print(org)
                return None
    return path
