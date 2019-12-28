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


class Chunk:
    """
    a chunk creates a reasonably convenient way to hold and access key value pairs, as well as a way to access
    a chunk's parent.  the value attribute can contain either a string or a list containing other Chunk instances
    """

    def __init__(self, key, value=None, parent=None,
                 append=False, quote_compound_keys=True):
        self.key = key
        self.value = value
        self.parent = parent
        self.quote_compound_keys = quote_compound_keys
        if append:
            parent.append(self)

    def __getitem__(self, item):
        return self.value[item]

    def __getattr__(self, attr):
        if self.has_len:
            for val in self.value:
                if val.key == attr:
                    return val

        raise AttributeError("has no attribute called %s" % attr)

    def __len__(self):
        if self.has_len:
            return len(self.value)
        return None

    @property
    def has_len(self):
        return isinstance(self.value, list)

    def __iter__(self):
        if self.has_len:
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
        if self.has_len:
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
        if self.has_len:
            for val in self.value:
                matches.extend(val.find_key(key))

        return matches

    def find_value(self, value):
        """
        recursively searches this chunk and its children and returns a list of chunks with the given value
        """
        matches = []
        if self.has_len:
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
        if self.has_len:
            for val in self.value:
                if val.key.lower() == key_lower and val.value == value:
                    matches.append(val)

                if recursive:
                    matches.extend(val.find_key_value(key, value))

        return matches

    def test_on_values(self, value_test):
        matches = []
        if self.has_len:
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

    def delete(self):
        """
        deletes this chunk
        """
        parent_chunk = self.parent
        if parent_chunk:

            # the chunk will definitely be in the _rawValue list so remove it from there
            # parent_chunk._rawValue.remove(self)

            # the chunk MAY also be in the value list, so try to remove it from there,
            # but if its not, ignore the exception.
            # NULL chunks don't get put into this list
            try:
                parent_chunk.value.remove(self)
            except ValueError:
                pass


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
        if isinstance(filepath, Path):
            self._filepath = filepath
        else:
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

        def null_callback(*_):
            pass

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
        if filepath and self.filepath.exists():
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
        if not isinstance(new_filepath, Path):
            self._filepath = Path(new_filepath)
        else:
            self._filepath = new_filepath

    def read(self, filepath=None):
        """
        reads the actual file, and passes the data read over to the parse_lines method
        """
        if filepath is None:
            filepath = self.filepath
        else:
            filepath = Path(filepath)

        self.parse_lines(filepath.open('r').readlines())

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
                parent_list_end.append(
                    self.chunk_class(
                        key, value, parent_list_end))
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
        the_string_lines = the_string.split('\n')
        self.parse_lines(the_string_lines)

    @property
    def has_len(self):
        try:
            _ = self.data[0]
            return True
        except IndexError:
            return False

    @property
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

    @property
    def root_chunk(self):
        """
        Return the base chunk for the file.
        """
        try:
            return self.value[0]
        except IndexError:
            return None

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
            filepath = Path(filepath)

        filepath.open('w').write(str(self))

    def reset_cache(self):
        pass


class GameInfoFile(KeyValueFile):
    """
    Provides an interface to gameinfo relevant operations - querying search paths, game root, game title etc...
    Passing startEmpty=True creates an empty object. Otherwise the current VPROJECT will be used to fetch the gameinfo.
    The parselines method can be passed a list of strings to fill an empty GameInfoFile object.
    """

    def __init__(self, filepath, chunk_class=Chunk,
                 read_callback=None, modname=None):
        self.modname = modname

        self.filepath = Path(filepath)

        self.filename = self.filepath.stem
        if not self.modname:
            self.modname = self.filepath.parent.name
        else:
            self.filename = None
        self.project = self.filepath.parent.parent
        self.path_cache = []
        super().__init__(filepath, parse_line, chunk_class, read_callback, True)

    def __getattr__(self, attr):
        try:
            return getattr(self[0], attr)
        except IndexError:
            raise AttributeError("attribute '%s' not found" % attr)

    def get_search_paths_recursive(self, visited_mods=None):
        if visited_mods is None:
            visited_mods = []
        if self.modname in visited_mods:
            return
        paths = self.get_search_paths()
        for path in self.get_search_paths():
            if path.stem == '*':
                path = path.parent
            gi_path = path / 'gameinfo.txt'
            if gi_path.exists():
                gi = GameInfoFile(gi_path)
                visited_mods.append(self.modname)
                new_paths = gi.get_search_paths_recursive(
                    list(set(visited_mods)))
                del gi
                if new_paths:
                    for p in new_paths:
                        if p not in paths:
                            paths.append(p)

        return list(paths)

    def get_search_paths(self):

        return [Path(self.project / modEntry)
                for modEntry in self.get_search_mods()]

    def get_search_mods(self, include_addons=None):
        """
        Get a list of mod names listed in the SearchPaths

        include_addons, depending if a global addon is set via the set_addon() function,
        and depending if include_addons is explicitly set versus left as the default the
        following behaviors occur, example assumes AddonRoot of 'foo_addons' and addon of 'bar'


                       | include_addons     | include_addons     | include_addons     |
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
        search_mods = [self.modname]
        # See if a global addon is set
        # if the user has explicitly set include_addons to True or the user has set a global
        # addon and hasn't explicitly set include_addons to False then include
        # addons

        gi = '|gameinfo_path|'
        sp = '|all_source_engine_paths|'
        for chunk in self.FileSystem.SearchPaths:
            addon = ('addonroot' in chunk.key.lower())
            if addon and not include_addons:
                continue
            pos = chunk.value.find(gi)
            if pos != -1:
                continue

            s_path = chunk.value.replace(sp, '')
            if not s_path:
                continue
            if s_path not in search_mods:
                if addon:
                    search_mods.insert(0, s_path)
                else:
                    search_mods.append(s_path)
        return search_mods

    def get_addon_roots(self):
        """
        Return a list of addon root names in the SearchPaths
        """
        addon_names = []
        for chunk in self.FileSystem.SearchPaths:
            if 'addonroot' in chunk.key.lower():
                if chunk.value not in addon_names:
                    addon_names.append(chunk.value)
        return addon_names

    def find_file(self, filepath: str, additional_dir=None,
                  extention=None, use_recursive=False):
        if use_recursive:
            if self.path_cache:
                paths = self.path_cache
            else:
                paths = self.get_search_paths_recursive()
                self.path_cache = paths

        else:
            paths = self.get_search_paths()
        for mod_path in paths:
            if mod_path.stem == '*':
                mod_path = mod_path.parent
            if additional_dir:
                new_filepath = mod_path / additional_dir / filepath
            else:
                new_filepath = mod_path / filepath
            if extention:
                new_filepath = new_filepath.with_suffix(extention)
            if new_filepath.exists():
                return new_filepath
        else:
            return None

    def find_texture(self, filepath, use_recursive=False):
        return self.find_file(filepath, 'materials',
                              extention='.vtf', use_recursive=use_recursive)

    def find_material(self, filepath, use_recursive=False):
        return self.find_file(filepath, 'materials',
                              extention='.vmt', use_recursive=use_recursive)

    @property
    def title(self):
        try:
            return self[0].title.value
        except AttributeError:
            try:
                return self[0].game.value
            except (KeyError, ValueError):
                return None

    @property
    def engine(self):
        try:
            return self[0].ToolsEnvironment.Engine.value
        except AttributeError:
            try:
                return self[0].engine.value
            except (KeyError, ValueError):
                return None

    def tool_dirs(self):
        try:
            return self[0].ToolsEnvironment.ToolsDir.value
        except AttributeError:
            try:
                return self[0].ToolsDir.value
            except (KeyError, ValueError):
                return None

    def write_default_file(self):
        """
        Creates a default GameInfo file with basic structure
        """
        self.filepath.open().write('''"GameInfo"\n{\n\tgame "tmp"\n\tFileSystem\n\t{\n\t\tSearchPaths\n'
'\t\t{\n\t\t\tGame |gameinfo_path|.\n\t\t}\n\t}\n}''')

    def simple_validate(self):
        """
        Checks to see if the file has some basic keyvalues
        """
        try:
            getattr(self[0], 'game')
            getattr(self[0], 'SearchPaths')
            return True
        except AttributeError:
            raise GameInfoException('Not a valid gameinfo file.')


class GameInfoException(Exception):
    def __init__(self, message, errno=None):
        Exception.__init__(self, message)
        self.errno = errno
        self.strerror = message


class MaterialPathResolver:
    """
    "Best Effort" material resolver for use when
    no Source games are installed, or files are
    not installed in a Source mod directory
    """

    def __init__(self, filepath):
        self._filepath = Path(filepath)

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, new_filepath):
        """
        this wrapper is here so to ensure the _filepath attribute is a ValvePath instance
        """
        if not isinstance(new_filepath, Path):
            self._filepath = Path(new_filepath)
        else:
            self._filepath = new_filepath

    def find_texture(self, filepath, use_recursive=False):
        return self.find_file(filepath, 'materials',
                              extention='.vtf', use_recursive=use_recursive)

    def find_material(self, filepath, use_recursive=False):
        return self.find_file(filepath, 'materials',
                              extention='.vmt', use_recursive=use_recursive)

    def find_file(self, filepath: str, additional_dir=None,
                  extention=None, use_recursive=False):
        filepath = Path(filepath)
        if use_recursive:
            while len(filepath.parts)>1:
                if additional_dir:
                    new_filepath = self.filepath / Path(additional_dir) / Path(filepath)
                else:
                    new_filepath = self.filepath / Path(filepath)

                if extention:
                    new_filepath = new_filepath.with_suffix(extention)

                if new_filepath.exists():
                    return new_filepath
                filepath = filepath.parent

        return None
