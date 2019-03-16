#         VALVE FILESYSTEM INTEGRATION
import os
import re
import shlex
import sys
from pathlib import Path

# Platform
from SourceIO.vtf.ValveFileSystem.path import ValvePath

WIN_32_SUFFIX = 'win32'
WIN_64_SUFFIX = 'win64'

# The mail server used to send mail
MAIL_SERVER = 'exchange'
DEFAULT_AUTHOR = 'python@valvesoftware.com'

# Make sure there is a HOME var...
try:
    os.environ['HOME'] = os.environ['USERPROFILE']
except KeyError:
    os.environ['HOME'] = str(Path('%HOMEDRIVE%/%HOMEPATH%'))

_MOD = None


def mod():
    """
    returns the mod name of the current project
    """
    global _MOD
    try:
        _MOD = Path(os.environ['VPROJECT']).name
        return _MOD
    except KeyError:
        raise KeyError('%VPROJECT% not defined')


_GAME = None


def game():
    """
    returns a ValvePath instance representing the %VGAME% path - path construction this way is super easy:
    somePropPath = game() / mod() / 'models/props/some_prop.dmx'
    """
    global _GAME
    try:
        _GAME = Path(os.environ['VPROJECT']) / '..'
        return _GAME
    except KeyError:
        raise KeyError('%VPROJECT% not defined.')
    except Exception:
        raise Exception('%VPROJECT% is defined with an invalid path.')


_CONTENT = None


def content():
    """
    returns a ValvePath instance representing the %VCONTENT% path - path construction this way is super easy:
    somePropPath = content() / 'ep3/models/characters/alyx/maya/alyx_model.ma'
    """
    global _CONTENT

    try:
        return Path(os.environ['VCONTENT'])
    except KeyError:
        try:
            _CONTENT = Path(os.environ['VPROJECT']) / '../../content'
            return _CONTENT
        except KeyError:
            KeyError('%VPROJECT% not defined')


_PROJECT = None


def project():
    """
    returns a ValvePath instance representing the %VPROJECT% path - path construction this way is super easy:
    somePropPath = project() / 'models/props/some_prop.mdl'
    """
    global _PROJECT
    try:
        _PROJECT = Path(os.environ['VPROJECT'])
        return _PROJECT
    except KeyError:
        raise KeyError('%VPROJECT% not defined')


_TOOLS = None


def tools(engine='Source 2'):
    """
    returns the location of our tools.
    """
    global _TOOLS

    if engine == 'Source':
        if _TOOLS is None:
            try:
                _TOOLS = Path(os.environ['VTOOLS'])
            except KeyError:
                try:
                    _TOOLS = Path(os.environ['VGAME']) / '/../../tools'

                except KeyError:
                    try:
                        _TOOLS = Path(os.environ['VPROJECT']) / '/../../../tools'
                    except KeyError:
                        raise KeyError('%VGAME% or %VPROJECT% not defined - cannot determine tools path')
    else:
        if _TOOLS is None:
            try:
                _TOOLS = Path(os.environ['VTOOLS'])
            except KeyError:
                try:
                    _TOOLS = Path(os.environ['VGAME']) / '/sdktools'
                except KeyError:
                    try:
                        _TOOLS = Path(os.environ['VPROJECT']) / '../sdktools'
                    except KeyError:
                        raise KeyError('%VGAME% or %VPROJECT% not defined - cannot determine tools path')

    return _TOOLS


_PLATFORM = WIN_32_SUFFIX


def platform():
    """
    Returns the platform of the current environment, defaults to win32
    """
    global _PLATFORM

    try:
        _PLATFORM = os.environ['VPLATFORM']
    except KeyError:
        try:
            # next try to determine platform by looking for win64 bin directory in the path
            bin64_dir = r'{0}\bin\{1}'.format(os.environ['VGAME'], WIN_64_SUFFIX)
            if bin64_dir in os.environ['PATH']:
                _PLATFORM = WIN_64_SUFFIX
        except (KeyError, Exception):
            pass
    return _PLATFORM


def addon():
    """
    Returns the addon of the current environment or None if no addon is set
    """
    try:
        s_addon = os.environ['VADDON']
        return s_addon
    except KeyError:
        return None


def iter_content_directories():
    for m in gameInfo.get_search_mods():
        yield content() / m


def iter_game_directories():
    for m in gameInfo.get_search_mods():
        yield game() / m


def get_addon_from_full_path(full_path):
    """
    Returns the name of the addon determined by examining the specified path
    Returns None if the specified file is not under an addon for the current
    game()/content() tree, calls getModAndAddonTupleFromFullPath() and returns
    2nd component
    """
    full_path = ValvePath(full_path)
    return full_path.addon_name


def set_addon_from_full_path(full_path):
    """
    Sets the addon from the specified path
    """
    full_path = ValvePath(full_path)
    set_addon(full_path.addon_name)


def resolve_valve_path(valve_path, base_path=content()):
    """
    A "Valve ValvePath" is one that is relative to a mod - in either the game or content tree.

    Ie: if you have a project with the content dir: d:/content and the game dir: d:/game
    a "Valve ValvePath" looks like this:  models/awesomeProps/coolThing.vmdl

    To resolve this path one must look under each mod the current project inherits from.
    So a project "test" which inherits from "another" would result in the following searches:
        d:/content/test/models/awesomeProps/coolThing.vmdl
        d:/content/another/models/awesomeProps/coolThing.vmdl

    Similarly for the game tree.

    If the path cannot be resolved to a real file, None is returned.
    """
    for mod_name in gameInfo.get_search_mods():
        p = base_path / mod_name / valve_path
        if p.exists:
            return p


def full_path_to_relative_path(path, base_path=content()):
    """
    Converts a full path to a relative path based on the current gameinfo and the
    specified base directory.  Directory defaults to content, call with game() to
    search the game directory.  If the path cannot be converted to a relative path
    it's returned exactly as it was passed in, otherwise the relative path is returned
    """

    full_path = os.path.normpath(str(path))
    base_path = os.path.normpath(str(base_path))

    for mod_name in gameInfo.get_search_mods():
        mod_name = str(mod_name)
        tmp_dir = os.path.normpath(os.path.join(base_path, mod_name))
        try:
            tmp_rel_path = os.path.relpath(full_path, tmp_dir)
        except ValueError:
            continue
        if not tmp_rel_path.startswith('..'):
            return tmp_rel_path

    # Not found, return what was passed
    return path


def relative_path_to_full_path(path, base_path=content(), exist=True, include_addons=None):
    """
    Converts a relative path to a full path based on the current gameinfo and
    specified base directory.  If exist is True then the file must already exist
    and if it cannot be found, nothing is returned.  if exist is False any file
    that matches on the search path will be preferentially returned but if not
    a path based on the current mod will be returned
    """

    rel_path = os.path.normpath(str(path))
    base_path = os.path.normpath(str(base_path))

    for mod_name in gameInfo.get_search_mods(include_addons=include_addons):
        mod_name = str(mod_name)
        tmp_dir = os.path.normpath(os.path.join(base_path, mod_name))
        tmp_full_path = os.path.join(tmp_dir, rel_path)
        if os.path.exists(tmp_full_path):
            return tmp_full_path

    # if we don't care if it exists, return basePath/mod/relPath
    if not exist:
        return os.path.normpath(os.path.join(os.path.join(base_path, mod()), rel_path))

    # Not found, must exist, return None
    return None


def fix_slashes(path, path_sep='/'):
    return path.replace('\\', path_sep).replace('/', path_sep)


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


def set_mod(new_mod):
    """
    sets the current mod to something else.  makes sure to update VPROJECT, VMOD and re-parses global gameInfo for
    the new mod so that calls to gamePath and contentPath return correctly
    """
    global gameInfo
    os.environ['VMOD'] = str(new_mod)
    os.environ['VPROJECT'] = (game() / new_mod).asNative()
    gameInfo = GameInfoFile()


def set_addon(new_addon):
    """
    sets the current addon to something else.  Mod needs to be set ahead of time
    Pass None or the empty string to unset the current addon
    """
    if new_addon:
        os.environ['VADDON'] = str(new_addon)
    elif 'VADDON' in os.environ:
        del os.environ['VADDON']


def as_relative(filepath):
    """
    """
    return str(ValvePath(filepath).as_relative())


def content_mod_relative_path(filepath):
    """
    Returns a path instance that is relative to the mod if the path is under the content tree
    """
    return ValvePath(filepath).as_content_mod_relative()


def addon_relative_content_path(filepath):
    """
    Returns a path instance that is relative to the addon if the path is under the content tree
    """
    # Make sure the path starts with content before stripping it away
    return filepath.asAddonRelativeContentPath()


def content_mod_relative_path_fuzzy(filepath):
    """
    returns a path instance that is relative to the mod if the path is under the content tree
    if an automatic match cannot be found, look for the content and mod strings using gameinfo file.
    """
    return filepath.asContentModRelativePathFuzzy()


def project_relative_path(filepath):
    """
    returns a path instance that is relative to vproject().
    this method is provided purely for symmetry - its pretty trivial
    """
    return filepath - project()


def make_source_absolute_path(filepath):
    """
    Returns a ValvePath instance as a "source" relative filepath.
    If the filepath doesn't exist under the project tree, the original filepath is returned.
    """
    return filepath.asModRelative()


def make_source1_texture_path(filepath):
    """
    returns the path as if it were a source1 texture/material path - ie the path is relative to the
    materials, or materialsrc directory.  if the materials or materialsrc directory can't be found
    in the path, the original path is returned
    """
    if not isinstance(filepath, ValvePath):
        filepath = ValvePath(filepath)

    try:
        idx = filepath.index('materials')
    except ValueError:
        try:
            idx = filepath.index('materialsrc')
        except ValueError:
            return filepath

    return filepath[idx + 1:]


_VALIDATE_LOCATION_IMPORT_HOOK = None


def enable_valid_dependency_check():
    """
    sets up an import hook that ensures all imported modules live under the game directory
    """
    global _VALIDATE_LOCATION_IMPORT_HOOK

    valid_paths = [game()]

    class Importer(object):
        @staticmethod
        def find_module(fullname, path=None):
            last_name = fullname.rsplit('.', 1)[-1]
            script_name = last_name + '.py'

            if path is None:
                path = []

            # not sure if there is a better way of asking python where it would look for a script/module - but
            # I think this encapsulates the logic...
            # at least under 2.6.  I think > 3 works differently?
            for d in (path + sys.path):
                py_filepath = ValvePath(d) / script_name
                py_module_path = ValvePath(d) / last_name / '__init__.py'
                if py_filepath.exists or py_module_path.exists:
                    for validPath in valid_paths:
                        if py_filepath.is_under(validPath):
                            return None

                    print("### importing a script outside of game!", py_filepath)
                    return None

            return None

    _VALIDATE_LOCATION_IMPORT_HOOK = Importer()
    sys.meta_path.append(_VALIDATE_LOCATION_IMPORT_HOOK)


def disable_valid_dependency_check():
    """
    disables the location validation import hook
    """
    global _VALIDATE_LOCATION_IMPORT_HOOK

    if _VALIDATE_LOCATION_IMPORT_HOOK is None:
        return

    sys.meta_path.remove(_VALIDATE_LOCATION_IMPORT_HOOK)

    _VALIDATE_LOCATION_IMPORT_HOOK = None


try:
    enable_hook = os.environ['ENSURE_PYTHON_CONTAINED']
    if enable_hook:
        enable_valid_dependency_check()
except KeyError:
    pass


def remove_line_comments(lines):
    """
    removes all line comments from a list of lines
    """
    new_lines = []
    for line in lines:
        comment_start = line.find('//')
        if comment_start != -1:
            line = line[:comment_start]
            if not line:
                continue
            # strip trailing whitespace and tabs
            line = line.rstrip(' \t')
        new_lines.append(line)

    return new_lines


def remove_block_comments(lines):
    """
    removes all block comments from a list of lines
    """
    new_lines = []
    end = len(lines)
    n = 0
    while n < end:
        block_comment_start = lines[n].find('/*')
        new_lines.append(lines[n])
        cont_flag = 0
        if block_comment_start != -1:
            new_lines[-1] = lines[n][:block_comment_start]
            while n < end:
                block_comment_end = lines[n].find('*/')
                if block_comment_end != -1:
                    new_lines[-1] += lines[n][block_comment_end + 2:]
                    n += 1
                    cont_flag = 1
                    break

                n += 1

        if cont_flag:
            continue

        n += 1
    return new_lines


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


class Chunk(object):
    """
    a chunk creates a reasonably convenient way to hold and access key value pairs, as well as a way to access
    a chunk's parent.  the value attribute can contain either a string or a list containing other Chunk instances
    """

    def __init__(self, key, value=None, parent=None, append=False, quote_compound_keys=True):
        self.key = key
        self.value = value
        self.parent = parent
        self.quote_compound_keys = quote_compound_keys
        if append:
            parent.append(self)

    def __getitem__(self, item):
        return self.value[item]

    def __getattr__(self, attr):
        if self.hasLen:
            for val in self.value:
                if val.key == attr:
                    return val

        raise AttributeError("has no attribute called %s" % attr)

    def __len__(self):
        if self.hasLen:
            return len(self.value)
        return None

    @property
    def has_len(self):
        return isinstance(self.value, list)

    def __iter__(self):
        if self.hasLen:
            return iter(self.value)
        raise TypeError("non-compound value is not iterable")

    def __repr__(self, depth=0):
        str_lines = []

        compound_line = '{0}{1}\n'
        if self.quote_compound_keys:
            compound_line = '{0}"{1}"\n'

        if isinstance(self.value, list) and not isinstance(self.value[0], int):
            str_lines.append(compound_line.format('\t' * depth, self.key))
            str_lines.append('\t' * depth + '{\n')
            for val in self.value:
                str_lines.append(val.__repr__(depth + 1))
            str_lines.append('\t' * depth + '}\n')
        else:
            v = self.value

            str_lines.append('%s"%s" "%s"\n' % ('\t' * depth, self.key, v))

        return ''.join(str_lines)

    __str__ = __repr__

    def __hash__(self):
        return id(self)

    def iter_children(self):
        """
        """
        if self.hasLen:
            for chunk in self:
                if chunk.has_len:
                    for subChunk in chunk.iter_children():
                        yield subChunk
                else:
                    yield chunk

    def as_dict(self, parent_dict):
        if isinstance(self.value, list):
            parent_dict[self.key] = sub_dict = {}
            for c in self.value:
                c.as_dict(sub_dict)
        else:
            parent_dict[self.key] = self.value

    def append(self, new):
        """
        Append a chunk to the end of the list.
        """
        if not isinstance(self.value, list):
            self.value = []

        self.value.append(new)

        # set the parent of the new Chunk to this instance
        new.parent = self

    def insert(self, index, new):
        """
        Insert a new chunk at a particular index.
        """
        if not isinstance(self.value, list):
            self.value = []

        self.value.insert(index, new)

        # Set the parent of the new Chunk to this instance
        new.parent = self

    def remove(self, chunk):
        """
        Remove given chunk from this chunk.
        """
        for c in self.value:
            if c == chunk:
                self.value.remove(c)
                return

    def remove_by_key(self, key):
        """
        Remove any chunks with the given key from this chunk. Does not recursively search all children.
        """
        for c in self.value:
            if c.key == key:
                self.value.remove(c)

    def find_key(self, key):
        """
        recursively searches this chunk and its children and returns a list of chunks with the given key
        """
        matches = []
        if self.key == key:
            matches.append(self)
        if self.hasLen:
            for val in self.value:
                matches.extend(val.find_key(key))

        return matches

    def find_value(self, value):
        """
        recursively searches this chunk and its children and returns a list of chunks with the given value
        """
        matches = []
        if self.hasLen:
            for val in self.value:
                matches.extend(val.find_value(value))
        elif self.value == value:
            matches.append(self)

        return matches

    def find_key_value(self, key, value, recursive=True):
        """
        recursively searches this chunk and its children and returns a list of chunks with the given key AND value
        """
        key_lower = key.lower()

        matches = []
        if self.hasLen:
            for val in self.value:
                if val.key.lower() == key_lower and val.value == value:
                    matches.append(val)

                if recursive:
                    matches.extend(val.find_key_value(key, value))

        return matches

    def test_on_values(self, value_test):
        matches = []
        if self.hasLen:
            for val in self.value:
                matches.extend(val.test_on_values(value_test))
        elif value_test(self.value):
            matches.append(self)

        return matches

    def list_attr(self):
        # lists all the "attributes" - an attribute is just as a named key.
        # NOTE: only Chunks with length have attributes
        attrs = []
        for attr in self:
            attrs.append(attr.key)

        return attrs

    def has_attr(self, attr):
        attrs = self.list_attr()
        return attr in attrs

    def get_file_object(self):
        """
        walks up the chunk hierarchy to find the top chunk
        """
        parent = self.parent
        last_parent = parent
        safety = 1000
        while parent is not None and safety:
            last_parent = parent
            parent = parent.parent
            safety -= 1

        return last_parent

    def duplicate(self, skip_null_chunks=False):
        """
        makes a deep copy of this chunk
        """
        chunk_type = type(self)

        def copy_chunk(chunk):
            chunk_copy = chunk_type(chunk.key)

            # recurse if nessecary
            if chunk.has_len:
                chunk_copy.value = []

                for childChunk in chunk.__iter__(skip_null_chunks):
                    child_chunk_copy = copy_chunk(childChunk)
                    chunk_copy.append(child_chunk_copy)
            else:
                chunk_copy.value = chunk.value

            return chunk_copy

        return copy_chunk(self)
    #
    # def delete(self):
    #     """
    #     deletes this chunk
    #     """
    #     parent_chunk = self.parent
    #     if parent_chunk:
    #
    #         # the chunk will definitely be in the _rawValue list so remove it from there
    #         parent_chunk._rawValue.remove(self)
    #
    #         # the chunk MAY also be in the value list, so try to remove it from there, but if its not,
    #         # ignore the exception.  NULL chunks don't get put into this list
    #         try:
    #             parent_chunk.value.remove(self)
    #         except ValueError:
    #             pass


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
    toks = [encode_quotes(token) for token in lex]

    if len(toks) == 1:
        # Key with no value gets empty string as value
        toks.append('')
    elif len(toks) > 2:
        # Multiple value tokens, invalid
        raise TypeError
    vals = toks[1].split(" ")
    if vals and len(vals) > 1:
        a = [val.isnumeric() for val in vals]
        if all(a):
            toks[1] = list(map(int, vals))
    return toks


class KeyValueFile(object):
    """
    A class for working with KeyValues2 format files.
    self.data contains a list which holds all the top level Chunk objects
    """

    def __init__(self, filepath=None, line_parser=parse_line, chunk_class=Chunk, read_callback=None,
                 supports_comments=True,
                 initial_data=None, string_buffer=None):
        """
        line_parser needs to return key,value
        """
        self._filepath = Path(filepath)
        self.data = self.value = []
        if initial_data is not None:
            self.data.append(initial_data)
        self.key = None
        self.parent = None
        self.line_parser = line_parser
        self.chunk_class = chunk_class
        self.callback = read_callback
        self.supports_comments = supports_comments

        def null_callback(*args):
            return args

        self.null_callback = null_callback

        # if no callback is defined, create a dummy one
        if self.callback is None:
            self.callback = null_callback

        # if no line parser was given, then use a default one
        if self.line_parser is None:
            def simple_line_parse(line):
                toks = line.split()
                if len(toks) == 1:
                    return toks[0], []
                else:
                    return toks[0], toks[1]

            self.line_parser = simple_line_parse

        # if a filepath exists, then read it
        if filepath and (os.path.exists(filepath)):
            self.read()
        if string_buffer:
            self.parse_lines(string_buffer)

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, new_filepath):
        """
        this wrapper is here so to ensure the _filepath attribute is a ValvePath instance
        """
        self._filepath = Path(new_filepath)

    def read(self, filepath=None):
        """
        reads the actual file, and passes the data read over to the parse_lines method
        """
        if filepath is None:
            filepath = self.filepath
        else:
            filepath = ValvePath(filepath)

        self.parse_lines(filepath.read())

    def parse_lines(self, lines):
        """
        this method does the actual parsing/data creation.  deals with comments, passing off data to the line_parser,
        firing off the read callback, all that juicy stuff...
        """
        lines = [l.strip() for l in lines]

        # remove comments
        if self.supports_comments:
            lines = stripcomments(lines)
        # lines = remove_line_comments(lines)
        # lines = remove_block_comments(lines)

        num_lines = len(lines)

        # hold a list representation of the current spot in the hierarchy
        parent_list = [self]
        parent_list_end = self
        callback = self.callback
        line_parser = self.line_parser
        for n, line in enumerate(lines):
            # run the callback - if there are any problems, replace the callback with the null_callback
            # noinspection PyBroadException
            try:
                callback(n, num_lines)
            except Exception:
                callback = self.null_callback

            if line == '':
                pass
            elif line == '{':
                cur_parent = parent_list[-1][-1]
                parent_list.append(cur_parent)
                parent_list_end = cur_parent
            elif line == '}':
                parent_list.pop()
                parent_list_end = parent_list[-1]
            else:
                try:
                    key, value = line_parser(line)
                except (TypeError, ValueError):
                    print(line)
                    raise TypeError(
                        'Malformed keyvalue found in file near line {0} in {1}. Check for misplaced quotes'.format(
                            n + 1, self.filepath))
                parent_list_end.append(self.chunk_class(key, value, parent_list_end))
            n += 1

    def __getitem__(self, *args):
        """
        provides an index based way of accessing file data - self[0,1,2] accesses the third child of
        the second child of the first root element in self
        """
        args = args[0]
        if not isinstance(args, tuple):
            data = self.data[args]
        else:
            data = self.data[args[0]]
            if len(args) > 1:
                for arg in args[1:]:
                    data = data[arg]

        return data

    def __len__(self):
        """
        lists the number of root elements in the file
        """
        return len(self.data)

    def __repr__(self):
        """
        this string representation of the file is almost identical to the formatting of a vmf file written
        directly out of hammer
        """
        str_list = []
        for chunk in self.data:
            a = str(chunk)
            str_list.append(a)
        return ''.join(str_list)

    __str__ = __repr__
    serialize = __repr__

    def unserialize(self, the_string):
        """
        """
        the_string_lines = the_string.split('\n')
        self.parse_lines(the_string_lines)

    @property
    def has_len(self):
        try:
            _ = self.data[0]
            return True
        except IndexError:
            return False

    def as_dict(self):
        """
        returns a dictionary representing the key value file - this isn't always possible as it is valid for
        a keyValueFile to have mutiple keys with the same key name within the same level - which obviously
        isn't possible with a dictionary - so beware!
        """
        as_dict = {}
        for chunk in self.data:
            chunk.as_dict(as_dict)

        return as_dict

    def append(self, chunk):
        """
        appends data to the root level of this file - provided to make the vmf file object appear
        more like a chunk object
        """
        self.data.append(chunk)

    def find_key(self, key):
        """
        returns a list of all chunks that contain the exact key given
        :rtype: List[Chunk]
        """
        matches = []
        for item in self.data:
            matches.extend(item.find_key(key))

        return matches

    def has_key(self, key):
        """
        returns true if the exact named key exists
        """
        for item in self.data:
            if item.has_attr(key):
                return True
        return False

    def find_value(self, value):
        """
        returns a list of all chunks that contain the exact value given
        """
        matches = []
        for item in self.data:
            matches.extend(item.find_value(value))

        return matches

    def find_key_value(self, key, value):
        """
        returns a list of all chunks that have the exact key and value given
        """
        matches = []
        for item in self.data:
            matches.extend(item.find_key_value(key, value))

        return matches

    def get_root_chunk(self):
        """
        Return the base chunk for the file.
        """
        try:
            return self.value[0]
        except IndexError:
            return None

    root_chunk = property(get_root_chunk, doc="The base chunk for the file.")

    def test_on_values(self, value_test):
        """
        returns a list of chunks that return true to the method given - the method should take as its
        first argument the value of the chunk it is testing against.  can be useful for finding values
        containing substrings, or all compound chunks etc...
        """
        matches = []
        for item in self.data:
            matches.extend(item.test_on_values(value_test))

        return matches

    def write(self, filepath=None):
        """
        writes the instance back to disk - optionally to a different location from that which it was
        loaded.  NOTE: deals with perforce should the file be managed by p4
        """
        if filepath is None:
            filepath = self.filepath
        else:
            filepath = ValvePath(filepath)

        filepath.write(str(self))

    def reset_cache(self):
        pass


class GameInfoFile(KeyValueFile):
    """
    Provides an interface to gameinfo relevant operations - querying search paths, game root, game title etc...
    Passing startEmpty=True creates an empty object. Otherwise the current VPROJECT will be used to fetch the gameinfo.
    The parselines method can be passed a list of strings to fill an empty GameInfoFile object.
    """

    def __init__(self, filepath=None, chunk_class=Chunk, read_callback=None, modname=None, start_empty=False):

        # noinspection PyBroadException
        try:
            project()
        except Exception:
            return

        self.modname = modname

        if (filepath is None) and (not start_empty):
            filepath = Path(os.path.join(str(project()), 'gameinfo.gi'))
            # look for a gameinfo.txt instead, pick a gameinfo.gi as default if it doesn't exist.
            if not filepath.exists:
                filepath = Path(os.path.join(str(project()), 'gameinfo.txt'))
            if not filepath.exists:
                filepath = Path(os.path.join(str(project()), 'gameinfo.gi'))

        if filepath:
            # Get filename and mod name from path
            self.filename = os.path.split(str(filepath))[1]
            if not self.modname:
                self.modname = os.path.split(os.path.dirname(str(filepath)))[1]
        else:
            self.filename = None

        KeyValueFile.__init__(self, filepath, parse_line, chunk_class, read_callback, True)

    def __getattr__(self, attr):
        try:
            return getattr(self[0], attr)
        except IndexError:
            raise AttributeError("attribute '%s' not found" % attr)

    def get_search_paths(self):
        return [str(ValvePath.join('%VPROJECT%/../', modEntry)) for modEntry in self.get_search_mods()]

    def get_search_mods(self, include_addons=None):
        """
        Get a list of mod names listed in the SearchPaths

        includeAddons, depending if a global addon is set via the set_addon() function,
        and depending if includeAddons is explicitly set versus left as the default the
        following behaviors occur, example assumes AddonRoot of 'foo_addons' and addon of 'bar'


                       | includeAddons     | includeAddons     | includeAddons     |
                       | Default(None)     | True              | False             |
        ---------------+-------------------+-------------------+-------------------+
         addon() None  | No addon          | AddonRoot         | No addon          |
                       |                   | prepended         |                   |
                       |                   |                   |                   |
         e.g.          |                   | 'foo_addons'      |                   |
        ---------------+-------------------+-------------------+-------------------+
         addon() 'bar' | AddonRoot/addon() | AddonRoot/addon() | No addon          |
                       | prepended         | prepended         |                   |
                       |                   |                   |                   |
         e.g.          | 'foo_addons/bar'  | 'foo_addons/bar'  |                   |
        ---------------+-------------------+-------------------+-------------------+

        """
        # always has the base mod in it...
        searchMods = [self.modname]
        # See if a global addon is set
        sAddon = addon()
        # if the user has explicitly set includeAddons to True or the user has set a global
        # addon and hasn't explicitly set includeAddons to False then include addons
        bIncludeAddons = include_addons or ((sAddon) and (include_addons is None))

        gi = '|gameinfo_path|'
        sp = '|all_source_engine_paths|'
        for chunk in self.FileSystem.SearchPaths:
            bAddon = ('addonroot' in chunk.key.lower())
            if (bAddon) and (not bIncludeAddons):
                continue
            pos = chunk.value.find(gi)
            if pos != -1: continue

            sPath = chunk.value.replace(sp, '')
            if not sPath: continue
            if bAddon:
                sAddon = addon()
                if sAddon:
                    sPath = sPath + '/' + sAddon
            if sPath not in searchMods:
                if bAddon:
                    searchMods.insert(0, sPath)
                else:
                    searchMods.append(sPath)
        return searchMods

    def getAddonRoots(self):
        """
        Return a list of addon root names in the SearchPaths
        """
        addonNames = []
        for chunk in self.FileSystem.SearchPaths:
            if 'addonroot' in chunk.key.lower():
                if chunk.value not in addonNames:
                    addonNames.append(chunk.value)
        return addonNames

    def getTitle(self):
        try:
            return self[0].title.value
        except AttributeError:
            try:
                return self[0].game.value
            except:
                return None

    title = property(getTitle)

    def getEngine(self):
        try:
            return self[0].ToolsEnvironment.Engine.value
        except AttributeError:
            try:
                return self[0].engine.value
            except:
                return None

    engine = property(getEngine)

    def getToolsDir(self):
        try:
            return self[0].ToolsEnvironment.ToolsDir.value
        except AttributeError:
            try:
                return self[0].ToolsDir.value
            except:
                return None

    toolsDir = property(getToolsDir)

    def __get_useVPLATFORM(self):
        try:
            return self[0].ToolsEnvironment.UseVPLATFORM.value
        except AttributeError:
            try:
                return self[0].UseVPLATFORM.value
            except:
                return None

    useVPLATFORM = property(__get_useVPLATFORM)

    def __getPythonVersion(self):
        try:
            return self[0].ToolsEnvironment.PythonVersion.value
        except AttributeError:
            try:
                return self[0].PythonVersion.value
            except:
                return None

    pythonVersion = property(__getPythonVersion)

    def __get_PythonHomeDisable(self):
        try:
            return self[0].ToolsEnvironment.PythonHomeDisable.value
        except AttributeError:
            try:
                return self[0].PythonHomeDisable.value
            except:
                raise

    pythonHomeDisable = property(__get_PythonHomeDisable)

    def __get_PythonDir(self):
        try:
            return self[0].ToolsEnvironment.PythonDir.value
        except AttributeError:
            try:
                return self[0].PythonDir.value
            except:
                return None

    pythonDir = property(__get_PythonDir)

    def writeDefaultFile(self):
        """
        Creates a default GameInfo file with basic structure
        """
        self.filepath.write('''"GameInfo"\n{\n\tgame "tmp"\n\tFileSystem\n\t{\n\t\tSearchPaths\n'+
		'\t\t{\n\t\t\tGame |gameinfo_path|.\n\t\t}\n\t}\n}''')

    def simpleValidate(self):
        """
        Checks to see if the file has some basic keyvalues
        """
        try:
            getattr(self[0], 'game')
            getattr(self[0], 'SearchPaths')
            return True
        except AttributeError:
            raise GameInfoException('Not a valid gameinfo file.')

        # Read the current gameinfo


gameInfo = GameInfoFile()


class GameInfoException(Exception):
    def __init__(self, message, errno=None):
        Exception.__init__(self, message)
        self.errno = None
        self.strerror = message


def getAddonBasePaths(asContent=False):
    """
    Returns a list of Paths for the addon base directories for the current mod (eg. 'dota_addons')
    Returns content-based Paths instead of game-based Paths if asContent is set to True.
    """
    if asContent:
        basePath = content()
    else:
        basePath = game()

    return [(basePath + addon) for addon in gameInfo.getAddonRoots()]


def getAddonPaths(asContent=False):
    """
    Returns a list of addons Paths for the current mod.
    Returns content-based Paths instead of game-based Paths if asContent is set to True.
    """
    addonPaths = []
    for base in getAddonBasePaths(asContent):
        addonPaths.extend(base.dirs())
    return addonPaths


def getAddonNames():
    """
    Returns a list of addon names for the current mod.
    """
    return [d.addon_name for d in getAddonPaths()]


def lsGamePath(path, recursive=False):
    """
    lists all files under a given 'valve path' - ie a game or content relative path.  this method needs to iterate
    over all known search mods as defined in a project's gameInfo script
    """
    path = Path(path)
    files = []

    for modPath in [Path(os.path.join(base, path.asNative())) for base in gameInfo.get_search_paths()]:
        files += list(modPath.files(recursive=recursive))

    return files


def lsContentPath(path, recursive=False):
    """
    similar to lsGamePath except that it lists files under the content tree, not the game tree
    """
    path = Path(path)
    c = content()
    files = []

    for modPath in [c / mod / path for mod in gameInfo.get_search_mods()]:
        files += list(modPath.files(recursive=recursive))

    return files


def contentPath(modRelativeContentPath):
    """allows you do specify a path using mod relative syntax instead of a fullpath
    example:
       assuming vproject is set to d:/main/game/tf_movies
       contentPath( 'models/player/soldier/parts/maya/soldier_reference.ma' )

       returns: d:/main/content/tf/models/player/soldier/parts/maya/soldier_reference.ma

    NOTE: None is returned in the file can't be found in the mod hierarchy
    """
    return Path(modRelativeContentPath).expandAsContent(gameInfo)


def gamePath(modRelativeContentPath):
    """allows you do specify a path using mod relative syntax instead of a fullpath
    example:
       assuming vproject is set to d:/main/game/tf
       gamePath( 'models/player/soldier.mdl' )

       returns: d:/main/game/tf/models/player/soldier.mdl

    NOTE: None is returned in the file can't be found in the mod hierarchy
    """
    return Path(modRelativeContentPath).expandAsGame(gameInfo)


def textureAsGameTexture(texturePath):
    """
    returns a resolved game texture filepath given some sort of texture path
    """
    if not isinstance(texturePath, Path):
        texturePath = Path(texturePath)

    if texturePath.isAbs():
        if texturePath.isUnder(content()):
            relPath = texturePath - content()
        else:
            relPath = texturePath - game()
        relPath = relPath[2:]
    else:
        relPath = texturePath
        if relPath.startswith('materials') or relPath.startswith('materialsrc'):
            relPath = relPath[1:]

    relPath = Path(Path('materials') + relPath).setExtension('vtf')
    relPath = relPath.expandAsGame(gameInfo)

    return relPath


def textureAsContentTexture(texturePath):
    """
    returns a resolved content texture filepath for some sort of texture path.  it looks for psd
    first, and then for tga.  if neither are found None is returned
    """
    if not isinstance(texturePath, Path):
        texturePath = Path(texturePath)

    xtns = ['psd', 'tga']
    if texturePath.isAbs():
        if texturePath.isUnder(content()):
            relPath = texturePath - content()
        else:
            relPath = texturePath - game()
        relPath = relPath[2:]
    else:
        relPath = texturePath
        if relPath.startswith('materials') or relPath.startswith('materialsrc'):
            relPath = relPath[1:]

    contentPath = Path('materialsrc') / relPath
    for xtn in xtns:
        tmp = contentPath.expandAsContent(gameInfo, xtn)
        if tmp is None: continue

        return tmp

    return None


def resolveMaterialPath(materialPath):
    """
    returns a resolved material path given some sort of material path
    """
    if not isinstance(materialPath, Path):
        materialPath = Path(materialPath)

    if materialPath.isAbs():
        if materialPath.isUnder(content()):
            relPath = materialPath - content()
        else:
            relPath = materialPath - game()
        relPath = relPath[2:]
    else:
        relPath = materialPath
        if relPath.startswith('materials') or relPath.startswith('materialsrc'):
            relPath = relPath[1:]

    relPath = ('materials' + relPath).setExtension('vmt')
    relPath = relPath.expandAsGame(gameInfo)

    return relPath


# end


