from __future__ import with_statement

import datetime
import os
import random
import re
import shutil
import stat
import sys
from pathlib import Path

from . import valve

# try to import the windows api - this may fail if we're not running on windows
try:
    import win32con, win32api
except ImportError:
    win32api = None
    win32con = None
    pass

# set the pickle protocol to use
PICKLE_PROTOCOL = 2

# set some variables for separators
NICE_SEPARATOR = '/'
NASTY_SEPARATOR = '\\'
NATIVE_SEPARATOR = (NICE_SEPARATOR, NASTY_SEPARATOR)[os.name == 'nt']
PATH_SEPARATOR = '/'  # (NICE_SEPARATOR, NASTY_SEPARATOR)[ os.name == 'nt' ]
OTHER_SEPARATOR = '\\'  # (NASTY_SEPARATOR, NICE_SEPARATOR)[ os.name == 'nt' ]
UNC_PREFIX = PATH_SEPARATOR * 2


def clean_path(path_string):
    """
    will clean out all nasty crap that gets into pathnames from various sources.
    maya will often put double, sometimes triple slashes, different slash types etc
    """
    path = str(path_string).strip().replace(OTHER_SEPARATOR, PATH_SEPARATOR)
    isUNC = path.startswith(UNC_PREFIX)
    while path.find(UNC_PREFIX) != -1:
        path = path.replace(UNC_PREFIX, PATH_SEPARATOR)

    if isUNC:
        path = PATH_SEPARATOR + path

    return path


ENV_REGEX = re.compile("%[^%]+%")
findall = re.findall


def real_path(path):
    if win32api:
        return win32api.GetLongPathName(win32api.GetShortPathName(str(path)))
    else:
        return path


class PathError(Exception):
    """
    Exception to handle errors in path values.
    """

    def __init__(self, msg, errno=None):
        # Exception.__init__( self, message )
        self.errno = errno
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

    def __repr__(self):
        return str(self.msg)


def resolve_and_split(path, env_dict=None, raise_on_missing=False):
    """
    recursively expands all environment variables and '..' tokens in a pathname
    """
    if env_dict is None:
        env_dict = os.environ

    path = str(path)

    # first resolve any env variables
    missing_vars = set()
    if '%' in path:  # performing this check is faster than doing the regex
        matches = findall(ENV_REGEX, path)
        while matches:
            for match in matches:
                try:
                    path = path.replace(match, env_dict[match[1:-1]])
                except KeyError:
                    if raise_on_missing:
                        raise PathError(
                            'Attempted to resolve a ValvePath using an environment variable that does not exist.', 1)
                    missing_vars.add(match)

            matches = set(findall(ENV_REGEX, path))

            # remove any variables that have been found to be missing...
            for missing in missing_vars:
                matches.remove(missing)

    # now resolve any subpath navigation
    if OTHER_SEPARATOR in path:  # believe it or not, checking this first is faster
        path = path.replace(OTHER_SEPARATOR, PATH_SEPARATOR)

    # is the path a UNC path?
    isUNC = path[:2] == UNC_PREFIX
    if isUNC:
        path = path[2:]

    # remove duplicate separators
    duplicate_separator = UNC_PREFIX
    while duplicate_separator in path:
        path = path.replace(duplicate_separator, PATH_SEPARATOR)

    path_toks = path.split(PATH_SEPARATOR)
    paths_to_use = []
    paths_to_use_append = paths_to_use.append
    for n, tok in enumerate(path_toks):
        # resolve a .. unless the previous token is a missing envar
        if tok == "..":
            if n > 0 and (path_toks[n - 1] in missing_vars):
                raise PathError(
                    'Attempted to resolve a ValvePath with ".." into the directory of environment variable "{0}" that does '
                    'not exist.'.format(
                        path_toks[n - 1]), 2)
            try:
                paths_to_use.pop()
            except IndexError:
                if raise_on_missing:
                    raise

                paths_to_use = path_toks[n:]
                break
        else:
            paths_to_use_append(tok)

    # finally convert it back into a path string and pop out the last token if its empty
    path = PATH_SEPARATOR.join(paths_to_use)
    try:
        if not paths_to_use[-1]:
            paths_to_use.pop()
    except IndexError:
        raise PathError('Attempted to resolve a ValvePath with "{0}", which is not a valid path string.'.format(path))

    path = real_path(path)

    # if its a UNC path, stick the UNC prefix
    if isUNC:
        return UNC_PREFIX + path, paths_to_use, True

    return path, paths_to_use, isUNC


def resolve(path, env_dict=None, raise_on_missing=False):
    return resolve_and_split(path, env_dict, raise_on_missing)[0]


def generate_random_path_name():
    now = datetime.datetime.now()
    rnd = '%06d' % (abs(random.gauss(0.5, 0.5) * 10 ** 6))
    return '%TEMP%' + PATH_SEPARATOR + 'TMP_FILE_%s%s%s%s%s%s%s%s' % (
        now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, rnd)


resolvePath = resolve

sz_BYTES = 0
sz_KILOBYTES = 1
sz_MEGABYTES = 2
sz_GIGABYTES = 3


class ValvePath(Path):
    __CASE_MATTERS = os.name != 'nt'

    @classmethod
    def set_case_matter(cls, state):
        cls.__CASE_MATTERS = state

    @classmethod
    def does_case_matter(cls):
        return cls.__CASE_MATTERS

    @classmethod
    def join(cls, *toks, **kw):
        return ValvePath('/'.join(toks), **kw)

    def __new__(cls, path='', case_matters=None, env_dict=None):
        """
        if case doesn't matter for the path instance you're creating, setting case_matters
        to False will do things like caseless equality testing, caseless hash generation
        """

        # early out if we've been given a ValvePath instance - paths are immutable so there is no reason not to just 
        # return what was passed in
        if type(path) == cls:
            return path

        # set to an empty string if we've been init'd with None
        if path is None:
            path = ''

        resolved_path, path_tokens, is_unc = resolve_and_split(path, env_dict)
        new = ValvePath(resolved_path)
        new.isUNC = is_unc
        new.hasTrailing = resolved_path.endswith(PATH_SEPARATOR)
        new._splits = tuple(path_tokens)
        new._passed = path

        # case sensitivity, if not specified, defaults to system behaviour
        if case_matters is not None:
            new.__CASE_MATTERS = case_matters

        return new

    @classmethod
    def temp(cls):
        """
        returns a temporary filepath - the file should be unique (i think) but certainly the file is guaranteed
        to not exist
        """
        random_path_name = ValvePath(generate_random_path_name())
        while random_path_name.exists:
            random_path_name = ValvePath(generate_random_path_name())

        return random_path_name

    def __add__(self, other):
        return self.__class__('%s%s%s' % (self, PATH_SEPARATOR, other), self.__CASE_MATTERS)

    # the / or + operator both concatenate path tokens
    __div__ = __add__
    __truediv__ = __add__

    def __radd__(self, other):
        return self.__class__(other, self.__CASE_MATTERS) + self

    __rdiv__ = __radd__

    def __getitem__(self, item):
        return self._splits[item]

    def __getslice__(self, a, b):
        is_unc = self.isUNC
        if a:
            is_unc = False

        return self._toks_to_path(self._splits[a:b], is_unc, self.hasTrailing)

    def __len__(self):

        return len(self._splits)

    def __contains__(self, item):
        if not self.__CASE_MATTERS:
            return item.lower() in [s.lower() for s in self._splits]

        return item in list(self._splits)

    def __hash__(self):
        """
        the hash for two paths that are identical should match - the most reliable way to do this
        is to use a tuple from self.split to generate the hash from
        """
        if not self.__CASE_MATTERS:
            return hash(tuple([s.lower() for s in self._splits]))

        return hash(tuple(self._splits))

    def _toks_to_path(self, toks, is_unc=False, has_trailing=False):
        """
        given a bunch of path tokens, deals with prepending and appending path
        separators for unc paths and paths with trailing separators
        """
        toks = list(toks)
        if is_unc:
            toks = ['', ''] + toks

        if has_trailing:
            toks.append('')

        return self.__class__(PATH_SEPARATOR.join(toks), self.__CASE_MATTERS)

    def resolve(self, env_dict=None, raise_on_missing=False):
        """
        will re-resolve the path given a new env_dict
        """
        if env_dict is None:
            return self
        else:
            return ValvePath(self._passed, self.__CASE_MATTERS, env_dict)

    def unresolved(self):
        """
        returns the un-resolved path - this is the exact string that the path was instantiated with
        """
        return self._passed

    def is_equal(self, other):
        """
        compares two paths after all variables have been resolved, and case sensitivity has been
        taken into account - the idea being that two paths are only equal if they refer to the
        same ValveFileSystem object.  NOTE: this doesn't take into account any sort of linking on *nix
        systems...
        """
        if isinstance(other, ValvePath):
            # Convert Paths to strings
            other_str = str(other.as_file())
        elif other:
            # Convert non-empty strings to Paths
            other_str = ValvePath(other, self.__CASE_MATTERS)
        else:
            # Leave empty strings and convert non-strings
            other_str = str(other)

        self_str = str(self.as_file())

        if not self.__CASE_MATTERS:
            self_str = self_str.lower()
            other_str = other_str.lower()

        return self_str == other_str

    __eq__ = is_equal

    def __ne__(self, other):
        return not self.is_equal(other)

    @property
    def as_str(self):
        return str(self)

    @classmethod
    def getcwd(cls):
        """
        returns the current working directory as a path object
        """
        return ValvePath(os.getcwd())

    @classmethod
    def setcwd(cls, path):
        """
        simply sets the current working directory - NOTE: this is a class method so it can be called
        without first constructing a path object
        """
        new_path = ValvePath(path)
        try:
            os.chdir(new_path)
        except WindowsError:
            return None

        return new_path

    putcwd = setcwd

    def get_stat(self):
        try:
            return os.stat(self.as_str)
        except:
            # return a null stat_result object
            return os.stat_result([0 for n in range(os.stat_result.n_sequence_fields)])

    stat = property(get_stat)

    def get_modified_date(self):
        """
        Return the last modified date in seconds.
        """
        return self.stat[8]

    modified_date = property(get_modified_date, doc="Return the last modified date in seconds.")

    def is_abs(self):
        try:
            return os.path.isabs(str(self))
        except:
            return False

    def abs(self):
        """
        returns the absolute path as is reported by os.path.abspath
        """
        return self.__class__(os.path.abspath(str(self)))

    def split(self, sep=None, maxsplit=None):
        """
        Returns the splits tuple - ie. the path tokens
        The additional arguments only included for class compatibility.
        """
        assert (sep is None) or (sep == '\\') or (
                sep == '/'), "ValvePath objects can only be split by path separators, ie. '/'."
        return list(self._splits)

    def as_dir(self):
        """
        makes sure there is a trailing / on the end of a path
        """
        if self.hasTrailing:
            return self

        return self.__class__('%s%s' % (self._passed, PATH_SEPARATOR), self.__CASE_MATTERS)

    def as_file(self):
        """
        makes sure there is no trailing path separators
        """
        if not self.hasTrailing:
            return self

        # Don't remove single slash Paths
        if len(self) == 1:
            return self

        return self.__class__(str(self)[:-1], self.__CASE_MATTERS)

    def is_dir(self):
        """
        bool indicating whether the path object points to an existing directory or not.  NOTE: a
        path object can still represent a file that refers to a file not yet in existence and this
        method will return False
        """
        return os.path.isdir(self)

    def is_file(self):
        """
        see isdir notes
        """
        return os.path.isfile(self)

    def get_readable(self):
        """
        returns whether the current instance's file is readable or not.  if the file
        doesn't exist False is returned
        """
        try:
            s = os.stat(str(self))
            return s.st_mode & stat.S_IREAD
        except:
            # i think this only happens if the file doesn't exist
            return False

    def is_readable(self):
        return bool(self.get_readable())

    def set_writable(self, state=True):
        """
        sets the writeable flag (ie: !readonly)
        """
        try:
            set_to = stat.S_IREAD
            if state:
                set_to = stat.S_IWRITE

            os.chmod(str(self), set_to)
        except:
            pass

    def get_writable(self):
        """
        returns whether the current instance's file is writeable or not.  if the file
        doesn't exist True is returned
        """
        try:
            s = os.stat(str(self))
            return s.st_mode & stat.S_IWRITE
        except:
            # i think this only happens if the file doesn't exist - so return true
            return True

    def is_writeable(self):
        return bool(self.get_writable())

    def has_extension(self, extension):
        """
        returns whether the extension is of a certain value or not
        """
        ext = self.suffix
        if not self.__CASE_MATTERS:
            ext = ext.lower()
            extension = extension.lower()

        return ext == extension

    def replace(self, search, replacement='', case_matters=None):
        """
        A simple search replace method - works on path tokens.
        If caseMatters is None, then the system default case sensitivity is used.
        If the string is not found, the original ValvePath is returned.
        """
        try:
            idx = self.find(search, case_matters)
        except ValueError:
            # If no match is found, return original ValvePath
            return self
        if idx == -1:
            # string not found, return the original object
            return self
        elif search in ('\\', '/'):
            # Use base class method if the search string is a path sep and return a str. If we don't
            # return a str, the path replacement would have no effect as the ValvePath.__str__
            # representation always presents a path with forward slashes.
            return ValvePath(str(self).replace(search, replacement))

        toks = list(self.split())
        toks[idx] = replacement

        return self._toks_to_path(toks, self.isUNC, self.hasTrailing)

    def find(self, search, case_matters=None):
        """
        Returns the index of the given path token.
        Returns -1 if the token is not found.
        """

        if case_matters is None:
            # in this case assume system case sensitivity - ie sensitive only on *nix platforms
            case_matters = self.__CASE_MATTERS

        if not case_matters:
            toks = [p.lower() for p in self.parts]
            search = search.lower()
        else:
            toks = self.parts

        try:
            idx = toks.index(search)
        except ValueError:
            return -1

        return idx

    index = find

    def match_case(self):
        """
        If running under an env where file case doesn't matter, this method will return a ValvePath instance
        whose case matches the file on disk.  It assumes the file exists
        """
        if self.does_case_matter():
            return self

        for f in self.parent.files():
            if f == self:
                return f

    def get_size(self, units=sz_MEGABYTES):
        """
        returns the size of the file in mega-bytes
        """
        div = float(1024 ** units)
        return os.path.getsize(self) / div

    def create(self):
        """
        if the directory doesn't exist - create it
        """
        if not self.exists:
            os.makedirs(str(self))

    def _delete(self):
        """
        WindowsError is raised if the file cannot be deleted
        """
        if self.is_file():
            try:
                os.remove(str(self))
            except WindowsError as e:
                win32api.SetFileAttributes(self, win32con.FILE_ATTRIBUTE_NORMAL)
                os.remove(str(self))
        elif self.is_dir():
            for f in self.files(recursive=True):
                f.delete()

            win32api.SetFileAttributes(self, win32con.FILE_ATTRIBUTE_NORMAL)
            shutil.rmtree(str(self.as_dir()), True)

    def delete(self):
        """
        Delete the file. For P4 operations, return the result.
        """
        return self._delete()

    remove = delete

    def _rename(self, new_name, name_is_leaf=False):
        """
        it is assumed newPath is a fullpath to the new dir OR file.  if nameIsLeaf is True then
        newName is taken to be a filename, not a filepath.  the fullpath to the renamed file is
        returned
        """
        new_path = ValvePath(new_name)
        if name_is_leaf:
            new_path = self.parent / new_name

        if self.is_file():
            if new_path != self:
                if new_path.exists:
                    new_path.delete()

            # Now perform the rename
            os.rename(str(self), new_path)
        elif self.is_dir():
            raise NotImplementedError('dir renaming not implemented yet...')

        return new_path

    def rename(self, new_name, name_is_leaf=False):
        """
        it is assumed newPath is a fullpath to the new dir OR file.  if nameIsLeaf is True then
        newName is taken to be a filename, not a filepath.  the instance is modified in place.
        if the file is in perforce, then a p4 rename (integrate/delete) is performed
        """
        if self.is_dir():
            raise NotImplementedError('dir renaming not implemented yet...')

        return self._rename(new_name, name_is_leaf)

    move = rename

    def _copy(self, target, name_is_leaf=False):
        """
        same as rename - except for copying.  returns the new target name
        """
        if self.is_file():
            target = ValvePath(target)
            if name_is_leaf:
                as_path = self.parent / target
                target = as_path

            if self == target:
                return target

            shutil.copy2(str(self), str(target))

            return target
        elif self.is_dir():
            raise NotImplementedError('dir copying not implemented yet...')

    # shutil.copytree( str(self), str(target) )
    def copy(self, target, name_is_leaf=False, ):
        """
        Same as rename - except for copying. Returns the new target name
        """
        if self.is_file():
            target = ValvePath(target)
            if name_is_leaf:
                target = self.parent / target

        return self._copy(target, name_is_leaf)

    def read(self, strip=True):
        """
        returns a list of lines contained in the file. NOTE: newlines are stripped from the end but whitespace
        at the head of each line is preserved unless strip=False
        """
        if self.exists and self.is_file():
            with self.open() as file_id:
                if strip:
                    lines = [line.rstrip() for line in file_id.readlines()]
                else:
                    lines = file_id.read()
                file_id.close()

            return lines

    def _write(self, contents_str):
        """
        writes a given string to the file defined by self
        """

        # make sure the directory to we're writing the file to exists
        self.parent.create()

        with self.open('w') as f:
            f.write(str(contents_str))

    def write(self, contents_str, ):
        """
        Wraps ValvePath.write:  if doP4 is true, the file will be either checked out of p4 before writing or
        add to perforce after writing if its not managed already.
        """
        assert isinstance(self, ValvePath)
        return self._write(contents_str)

    def __sub__(self, other):
        return self.relative_to(other)

    def __rsub__(self, other):
        return self.__class__(other, self.__CASE_MATTERS).relative_to(self)

    def inject(self, other, env_dict=None):
        """
        injects an env variable into the path - if the env variable doesn't
        resolve to tokens that exist in the path, a path string with the same
        value as self is returned...

        NOTE: a string is returned, not a ValvePath instance - as ValvePath instances are
        always resolved

        NOTE: this method is alias'd by __lshift__ and so can be accessed using the << operator:
        d:/main/content/mod/models/someModel.ma << '%VCONTENT%' results in %VCONTENT%/mod/models/someModel.ma
        """

        toks = toks_lower = self._splits
        other_toks = ValvePath(other, self.__CASE_MATTERS, env_dict=env_dict).split()
        new_toks = []
        n = 0
        if not self.__CASE_MATTERS:
            toks_lower = [t.lower() for t in toks]
            other_toks = [t.lower() for t in other_toks]

        while n < len(toks):
            tok, tok_lower = toks[n], toks_lower[n]
            if tok_lower == other_toks[0]:
                all_match = True
                for tok, otherTok in zip(toks_lower[n + 1:], other_toks[1:]):
                    if tok != otherTok:
                        all_match = False
                        break

                if all_match:
                    new_toks.append(other)
                    n += len(other_toks) - 1
                else:
                    new_toks.append(toks[n])
            else:
                new_toks.append(tok)
            n += 1

        return PATH_SEPARATOR.join(new_toks)

    __lshift__ = inject

    def find_nearest(self):
        """
        returns the longest path that exists on disk
        """
        path = self
        while not path.exists and len(path) > 1:
            path = path.up()

        if not path.exists:
            return None
        return path

    def as_native(self):
        """
        returns a string with system native path separators
        """
        return str(self).replace(PATH_SEPARATOR, NATIVE_SEPARATOR)

    def startswith(self, other):
        """
        returns whether the current instance begins with a given path fragment.  ie:
        ValvePath('d:/temp/someDir/').startswith('d:/temp') returns True
        """
        if not isinstance(other, type(self)):
            other = ValvePath(other, self.__CASE_MATTERS)

        other_toks = other.split()
        self_toks = self.split()
        if not self.__CASE_MATTERS:
            other_toks = [t.lower() for t in other_toks]
            self_toks = [t.lower() for t in self_toks]

        if len(other_toks) > len(self_toks):
            return False

        for tokOther, tokSelf in zip(other_toks, self_toks):
            if tokOther != tokSelf:
                return False

        return True

    is_under = startswith

    def endswith(self, other):
        """
        determines whether self ends with the given path - it can be a string
        """
        # copies of these objects NEED to be made, as the results from them are often cached - hence modification
        # to them would screw up the cache, causing really hard to track down bugs...  not sure what the best answer to
        # this is,
        # but this is clearly not it...  the caching decorator could always return copies of mutable objects, but that
        # sounds wasteful...  for now, this is a workaround
        other_toks = list(ValvePath(other).split())
        self_toks = list(self._splits)
        other_toks.reverse()
        self_toks.reverse()
        if not self.__CASE_MATTERS:
            other_toks = [t.lower() for t in other_toks]
            self_toks = [t.lower() for t in self_toks]

        for tok_other, tok_self in zip(other_toks, self_toks):
            if tok_other != tok_self:
                return False

        return True

    def _list_filesystem_items(self, itemtest, names_only=False, recursive=False):
        """
        does all the listing work - itemtest can generally only be one of os.path.isfile or
        os.path.isdir.  if anything else is passed in, the arg given is the full path as a
        string to the ValveFileSystem item
        """
        if not self.exists:
            return

        if recursive:
            walker = os.walk(str(self))
            for path, subs, files in walker:
                path = ValvePath(path, self.__CASE_MATTERS)

                for sub in subs:
                    p = path / sub
                    if itemtest(p):
                        if names_only:
                            p = p.name

                        yield p
                    else:
                        break  # if this doesn't match, none of the other subs will

                for item in files:
                    p = path / item
                    if itemtest(p):
                        if names_only:
                            p = p.name

                        yield p
                    else:
                        break  # if this doesn't match, none of the other items will
        else:
            for item in os.listdir(str(self)):
                p = self / item
                if itemtest(p):
                    if names_only:
                        p = p.name

                    yield p

    def dirs(self, names_only=False, recursive=False):
        """
        returns a generator that lists all sub-directories.  If namesOnly is True, then only directory
        names (relative to the current dir) are returned
        """
        return self._list_filesystem_items(os.path.isdir, names_only, recursive)

    def files(self, names_only=False, recursive=False):
        """
        returns a generator that lists all files under the path (assuming its a directory).  If namesOnly
        is True, then only directory names (relative to the current dir) are returned
        """
        return self._list_filesystem_items(os.path.isfile, names_only, recursive)

    ########### VALVE SPECIFIC PATH METHODS ###########

    def expand_as_game(self, gameinfo, extension=None, must_exist=True):
        """
        expands a given "mod" relative path to a real path under the game tree.  if an extension is not given, it is
        assumed the path already contains an extension.  ie:  models/player/scout.vmt would get expanded to:
        <gameRoot>/<mod found under>/models/player/scout.vmt - where the mod found under is the actual mod the file
        exists under on the user's system

        The mustExistFlag was added due to a bug with how a mod's pre-existing assets were being determined.
        If the files did not exist it would default to the parent mod, and then delete those files as part of a
        clean up step so in the pre-compiled check we set mustExist to be False.
        """
        the_path = self
        if extension is not None:
            the_path = self.with_suffix(extension)

        search_paths = gameinfo.get_search_paths()
        for path in search_paths:

            tmp = os.path.join(path, str(the_path))
            if not must_exist or os.path.exists(tmp):
                return tmp

        return None

    def expand_as_content(self, gameinfo, extension=None):
        """
        as for expand_as_game except for content rooted paths
        """
        the_path = self
        if extension is not None:
            the_path = self.with_suffix(extension)

        for mod in gameinfo.get_search_mods():
            under_mod = '%VCONTENT%' + (mod + the_path)
            if under_mod.exists:
                return under_mod

        return None

    def expand_as_game_addon(self, gameinfo, addon, extension=None, must_exist=False):
        """
        Given an addon name, expand a relative ValvePath to an absolute game ValvePath.
        E.g. passing 'pudge_battle' and a ValvePath of 'maps/pudge_battle.bsp', this would expand to:
        '<VGAME>/dota/dota_addons/pudge_battle/maps/pudge_battle.bsp'
        """
        addonPath = '%VGAME%' + self.expand_as_addon_base_path(addon)
        if extension is not None:
            addonPath = self.with_suffix(extension)
        if (not must_exist) or (os.path.exists(addonPath)):
            return addonPath

    def expand_as_content_addon(self, gameinfo, addon, extension=None, must_exist=False):
        """
        Given an addon name, expand a relative ValvePath to an abosolute game ValvePath.
        E.g. passing 'pudge_battle' and a ValvePath of 'maps/pudge_battle.bsp', this would expand to:
        '<VCONTENT>/dota/dota_addons/pudge_battle/maps/pudge_battle.bsp'
        """
        addonPath = '%VCONTENT%' + self.expand_as_addon_base_path(addon)
        if extension is not None:
            addonPath = self.with_suffix(extension)
        if (not must_exist) or (os.path.exists(addonPath)):
            return addonPath

    def expand_as_addon_base_path(self, addon):
        """
        Given an addon name, expand as an addon-relative ValvePath for the mod.
        E.g. passing 'pudge_battle' and a ValvePath of 'maps/pudge_battle.bsp', this would expand to:
        'dota_addons/pudge_battle/maps/pudge_battle.bsp'
        """
        return ValvePath('{0}_addons\\{1}\\').format(valve.mod(), addon) + self

    def belongs_to_content(self, game_info):
        for mod in game_info.get_search_mods():
            if self.is_under(valve.content() / mod):
                return True

        return False

    def belongs_to_game(self, game_info):
        for mod in game_info.get_search_mods():
            if self.is_under(valve.game() / mod):
                return True
        return False

    def belongs_to_mod(self):
        """
        Return True if ValvePath belongs under *content* of the current mod
        """
        if self.is_under(valve.content() / valve.mod()):
            return True
        return False

    def as_relative(self):
        """
        returns the path relative to either VCONTENT or VGAME.
        If the path isn't under either of these directories, None is returned.
        """
        c = valve.content()
        g = valve.game()
        if self.is_under(c):
            return self - c
        elif self.is_under(g):
            return self - g
        return None

    def as_relative_fuzzy(self):
        """
        Returns the path relative to either VCONTENT or VGAME if there is an exact match.
        If the path isn't under either of these directories, try looking for a pattern match using 'content' or 'game'
        in the path and matching any of the mods in gameinfo Search Paths.
        If still no matches, return None
        """
        # Try direct match first
        rel = self.as_relative()
        if rel is not None:
            return rel

        # No direct matches found, continue
        mods = valve.gameInfo.get_search_mods()
        toks = self.split()
        toks_lower = [s.lower() for s in toks]

        # Step up through tokens looking for a case-insensitive match
        for modName in mods:
            if modName in toks:
                if 'content' in toks_lower:
                    for i in range(len(toks) - 1, 1, -1):
                        # Look for set of tokens that are 'content' followed by the mod name
                        if (toks_lower[i] == modName) and (toks_lower[i - 1] == 'content'):
                            return ValvePath('\\'.join(toks[i:]))
                elif 'game' in toks_lower:
                    for i in range(len(toks) - 1, 1, -1):
                        # Look for set of tokens that are 'game' followed by the mod name
                        if (toks_lower[i] == modName) and (toks_lower[i - 1] == 'game'):
                            return ValvePath('\\'.join(toks[i:]))
        return None

    def as_mod_relative(self):
        """
        Returns a ValvePath instance that is relative to the mod.
        If the path doesn't match the game or content, the original filepath is returned.
        """
        rel_path = self.as_relative()
        if not rel_path: return self
        return rel_path[1:]

    def as_content_mod_relative(self):
        """
        Returns a path instance that is relative to the mod if the path is under the content tree.
        If the path doesn't match the content tree, the original filepath is returned.
        """
        # Make sure the path starts with content before stripping it away
        if self.startswith(valve.content()):
            return self.as_relative()[1:]
        else:
            return self

    def as_content_mod_relative_path_fuzzy(self):
        """
        Returns a path instance that is relative to the mod if the path is under the content tree.
        if an automatic match cannot be found, look for the content and mod strings using gameinfo file.
        """
        # Get content-relative path first
        rel_path = self.as_relative_fuzzy()

        if rel_path is not None:
            # Make sure the path starts with a mod in search mods before stripping it away
            for modName in valve.gameInfo.get_search_mods():
                if rel_path.startswith(modName):
                    return rel_path[1:]
            # Mod name must already be stripped
            return rel_path
        else:
            # Couldn't find a relative path, return original
            return self

    def as_game_mod_relative(self):
        """
        Returns a path instance that is relative to the mod if the path is under the game tree.
        If the path doesn't match the game tree, the original filepath is returned.
        """
        # Make sure the path starts with content before stripping it away
        if self.startswith(valve.game()):
            return self.as_relative()[1:]
        else:
            return self

    def as_addon_relative(self):
        """
        Takes an absolute ValvePath and returns a path instance that is relative to the addon, either
        in VCONTENT or VGAME. If the path isn't under either of these directories, None is returned.
        """
        rel_path = self.as_relative()
        if rel_path and rel_path.is_addon and (len(rel_path) > 2):
            return rel_path[2:]
        return None

    def as_addon_relative_fuzzy(self):
        """
        Returns the addon-relative path under either VCONTENT or VGAME if there is an exact match.
        If the path isn't under either of these directories, try looking for a pattern match using 'content' or 'game'
        in the addon path and matching any of the mods in gameinfo Search Paths.
        If still no matches, return None
        """
        # Try direct match first
        rel = self.as_addon_relative()
        if rel is not None:
            return rel

        # No direct matches found, continue
        addon_mod_names = ['{0}_addons'.format(m.lower()) for m in valve.gameInfo.get_search_mods()]
        toks = self.split()
        toks_lower = [s.lower() for s in toks]

        # Step up through tokens looking for a case-insensitive match
        for addonModName in addon_mod_names:
            if addonModName in toks:
                if 'content' in toks_lower:
                    for i in range(len(toks) - 1, 1, -1):
                        # Look for set of tokens that are 'content' followed by the mod name
                        if (toks_lower[i] == addonModName) and (toks_lower[i - 1] == 'content'):
                            addon_rel_path = ValvePath('\\'.join(toks[i:]))
                            # Return path with mod_addons and addon name removed
                            return addon_rel_path[2:]
                elif 'game' in toks_lower:
                    for i in range(len(toks) - 1, 1, -1):
                        # Look for set of tokens that are 'game' followed by the mod name
                        if (toks_lower[i] == addonModName) and (toks_lower[i - 1] == 'game'):
                            addon_rel_path = ValvePath('\\'.join(toks[i:]))
                            # Return path with mod_addons and addon name removed
                            return addon_rel_path[2:]
        return None

    def as_addon_relative_game_path(self):
        """
        Takes an absolute VGAME ValvePath and returns a path instance that is relative to the addon.
        If the path isn't under VGAME, None is returned.
        """
        if self.is_under(valve.game()):
            return self.as_addon_relative()
        return None

    def as_addon_relative_content_path(self):
        """
        Takes an absolute VCONTENT ValvePath and returns a path instance that is relative to the addon.
        If the path isn't under VCONTENT, None is returned.
        """
        if self.is_under(valve.content()):
            return self.as_addon_relative()
        return None

    def as_mdl(self, is_addon=False):
        """
        Returns as .mdl path relative to the models directory.
        Passing the is_addon argument as True will force it look for an addon directory
        to strip from the base of the path.
        """
        p = self.as_mod_relative()
        assert p != '', 'mdl cannot be determined from empty path!'

        if self.is_addon or is_addon:
            # Remove the addon base path if it is present
            if p.startswith(p.addon_base_dir):
                p = p[1:]
            # For addons, models dir is under the addon name.
            assert p[1] == 'models', 'path not under addon models directory!'
            mdl = p[2:].setExtension('mdl')
        else:
            assert p[0] == 'models', 'path not under models directory!'
            mdl = p[1:].setExtension('mdl')

        return mdl

    def as_vmdl(self, is_addon=False):
        """
        Returns as .vmdl path relative to the models directory.
        Passing the is_addon argument as True will force it look for an addon directory
        to strip from the base of the path.
        """
        p = self.as_mod_relative()
        assert p != '', 'vmdl cannot be determined from empty path!'
        if self.is_addon or is_addon:
            # Remove the addon base path if it is present
            if len(p) > 1 and p.addon_base_dir:
                p = p[1:]
            # For addons, models dir is under the addon name.
            assert p[1] == 'models', 'path not under addon models directory!'
            vmdl = p[2:].setExtension('vmdl')

        else:
            assert p[0] == 'models', 'path not under models directory!'
            vmdl = p[1:].setExtension('vmdl')

        return vmdl

    @property
    def addon_name(self):
        """
        Returns the addon name, if found, or None.
        Note that this is somewhat strict, as the _addons base directory must be directly under game or content,
        or in the case of a relative path, at the start of the path.
        """
        for base in valve.gameInfo.get_addon_roots():

            if (self.startswith(base)) and (len(self) > 1):
                # The path is already relative, starting with addons dir
                return self[1]

            # Get a relative path
            rel_path = self.as_relative()
            if rel_path and (rel_path.startswith(base)) and (len(rel_path) > 1):
                return rel_path[1]

        return None

    @property
    def is_addon(self):
        """
        Returns True is the path is an Addon path.
        Note that this is somewhat strict, as the <mod>_addons
        must be directly under game or content, or at the start of the path.
        """
        return self.addon_name is not None

    @property
    def addon_base_dir(self):
        """
        Returns the addons base directory name found in the ValvePath (e.g. 'dota_addons' ) or None.
        """
        for base in valve.getAddonBasePaths(asContent=self.is_under(valve.content())):
            if self.is_under(base):
                return base.name()
        return None


def find_in_pypath(filename):
    """
    given a filename or path fragment, will return the full path to the first matching file found in
    the sys.path variable
    """
    for p in map(ValvePath, sys.path):
        loc = p / filename
        if loc.exists:
            return loc

    return None


def find_in_path(filename):
    """
    given a filename or path fragment, will return the full path to the first matching file found in
    the PATH env variable
    """
    for p in map(ValvePath, os.environ['PATH'].split(';')):
        loc = p / filename
        if loc.exists:
            return loc

    return None

# end
