import os

from .path import ValvePath, PathError, P4File
from .valve import gameInfo, game
from .misc import removeDupes
import datetime

LOCALES = LOCAL, GLOBAL = 'local', 'global'
DEFAULT_XTN = 'preset'

# define where the base directories are for presets
kLOCAL_BASE_DIR = ValvePath('%HOME%/presets/')
try:
    kGLOBAL_BASE_DIR = game() / 'sdktools/presets'
except (PathError, KeyError):
    # If game can't be found, just guess from the script location
    kGLOBAL_BASE_DIR = ValvePath(__file__).up(6) / 'sdktools/presets'


class PresetException(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


def getPresetDirs(locale, tool):
    '''
    returns the base directory for a given tool's preset files
    '''
    global kLOCAL_BASE_DIR, kGLOBAL_BASE_DIR

    dirs = []
    mods = gameInfo.get_search_mods()
    if locale == LOCAL:
        for mod in mods:
            localDir = kLOCAL_BASE_DIR / mod / tool
            localDir.create()
            dirs.append(localDir)

        return dirs

    for mod in mods:
        globalDir = kGLOBAL_BASE_DIR / mod / tool
        globalDir.create()
        dirs.append(globalDir)

    return dirs


def presetPath(locale, tool, presetName, ext=DEFAULT_XTN):
    preset = getPresetDirs(locale, tool)[0] + scrubName(presetName, exceptions='./')
    preset = preset.set_extension(ext)

    return preset


def readPreset(locale, tool, presetName, ext=DEFAULT_XTN):
    '''
    reads in a preset file if it exists, returning its contents
    '''
    file = getPresetPath(presetName, tool, ext, locale)
    if file is not None:
        return file.read()
    return []


def savePreset(locale, tool, presetName, ext=DEFAULT_XTN, contentsStr=''):
    '''
    given a contents string, this convenience method will store it to a preset file
    '''
    preset = Preset(locale, tool, presetName, ext)
    preset.write(contentsStr, locale == GLOBAL)

    return preset


def unpicklePreset(locale, tool, presetName, ext=DEFAULT_XTN):
    '''
    same as readPreset except for pickled presets
    '''
    dirs = getPresetDirs(locale, tool)
    for dir in dirs:
        cur = dir / presetName
        cur.extension = ext
        if cur.exists: return cur.unpickle()
    raise IOError("file doesn't exist!")


def picklePreset(locale, tool, presetName, ext=DEFAULT_XTN, contentsObj=None):
    preset = presetPath(locale, tool, presetName, ext)
    preset.pickle(contentsObj, locale == GLOBAL)


def listPresets(locale, tool, ext=DEFAULT_XTN):
    '''
    lists the presets in a given local for a given tool
    '''
    files = []
    alreadyAdded = set()
    for d in getPresetDirs(locale, tool):
        if d.exists:
            for f in d.files():
                if f.name() in alreadyAdded: continue
                if f.has_extension(ext):
                    files.append(f)
                    alreadyAdded.add(f.name())

    # remove duplicates
    files = removeDupes(files)
    files = [Preset.FromFile(f) for f in files]

    return files


def listAllPresets(tool, ext=DEFAULT_XTN, localTakesPrecedence=False):
    '''
    lists all presets for a given tool and returns a dict with local and global keys.  the dict
    values are lists of ValvePath instances to the preset files, and are unique - so a preset in the
    global list will not appear in the local list by default.  if localTakesPrecedence is True,
    then this behaviour is reversed, and locals will trump global presets of the same name
    '''
    primaryLocale = GLOBAL
    secondaryLocale = LOCAL
    primary = listPresets(primaryLocale, tool, ext)
    secondary = listPresets(secondaryLocale, tool, ext)

    if localTakesPrecedence:
        primary, secondary = secondary, primary
        primaryLocale, secondaryLocale = secondaryLocale, primaryLocale

    # so teh localTakesPrecedence determines which locale "wins" when there are leaf name clashes
    # ie if there is a preset in both locales called "yesplease.preset", if localTakesPrecedence is
    # False, then the global one gets included, otherwise the local one is listed
    alreadyAdded = set()
    locales = {LOCAL: [], GLOBAL: []}
    for p in primary:
        locales[primaryLocale].append(p)
        alreadyAdded.add(p.name())

    for p in secondary:
        if p.name() in alreadyAdded: continue
        locales[secondaryLocale].append(p)

    return locales


def getPresetPath(presetName, tool, ext=DEFAULT_XTN, locale=GLOBAL):
    '''
    given a preset name, this method will return a path to that preset if it exists.  it respects the project's
    mod hierarchy, so it may return a path to a file not under the current mod's actual preset directory...
    '''
    searchPreset = '%s.%s' % (presetName, ext)
    dirs = getPresetDirs(locale, tool)
    for dir in dirs:
        presetPath = dir / searchPreset
        if presetPath.exists:
            return presetPath


def findPreset(presetName, tool, ext=DEFAULT_XTN, startLocale=LOCAL):
    '''
    looks through all locales and all search mods for a given preset name.  the startLocale simply dictates which
    locale is searched first - so if a preset exists under both locales, then then one found in the startLocale
    will get returned
    '''
    other = list(LOCALES).remove(startLocale)
    for loc in [startLocale, other]:
        p = getPresetPath(presetName, tool, ext, loc)
        if p is not None: return p


def dataFromPresetPath(path):
    '''
    returns a tuple containing the locale, tool, name, extension for a given ValvePath instance.  a PresetException
    is raised if the path given isn't an actual preset path
    '''
    locale, tool, name, ext = None, None, None, None
    pathCopy = ValvePath(path)
    if pathCopy.isUnder(kGLOBAL_BASE_DIR):
        locale = GLOBAL
        pathCopy -= kGLOBAL_BASE_DIR
    elif pathCopy.isUnder(kLOCAL_BASE_DIR):
        locale = LOCAL
        pathCopy -= kLOCAL_BASE_DIR
    else:
        raise PresetException("%s isn't under the local or the global preset dir" % path)

    tool = pathCopy[-2]
    ext = pathCopy.set_extension()
    name = pathCopy.name()

    return locale, tool, name, ext


def scrubName(theStr, replaceChar='_', exceptions=None):
    invalidChars = """`~!@#$%^&*()-+=[]\\{}|;':"/?><., """
    if exceptions:
        for char in exceptions:
            invalidChars = invalidChars.replace(char, '')

    for char in invalidChars:
        theStr = theStr.replace(char, '_')

    return theStr


# these are a bunch of variables used for keys in the export dict.  they're provided mainly for
# the sake of auto-completion...
kEXPORT_DICT_USER = 'user'
kEXPORT_DICT_MACHINE = 'machine'
kEXPORT_DICT_DATE = 'date'
kEXPORT_DICT_TIME = 'time'
kEXPORT_DICT_PROJECT = 'project'
kEXPORT_DICT_CONTENT = 'content'
kEXPORT_DICT_TOOL = 'tool_name'
kEXPORT_DICT_TOOL_VER = 'tool_version'
kEXPORT_DICT_SOURCE = 'scene'  # the source of the file - if any


def writeExportDict(toolName=None, toolVersion=None, **kwargs):
    '''
    returns a dictionary containing a bunch of common info to write when generating presets
    or other such export type data
    '''
    d = {}
    d[kEXPORT_DICT_USER] = os.environ['USERNAME']
    d[kEXPORT_DICT_MACHINE] = os.environ['COMPUTERNAME']
    now = datetime.datetime.now()
    d[kEXPORT_DICT_DATE], d[kEXPORT_DICT_TIME] = now.date(), now.time()
    d[kEXPORT_DICT_PROJECT] = os.environ['VPROJECT']
    d[kEXPORT_DICT_CONTENT] = os.environ['VCONTENT']
    d[kEXPORT_DICT_TOOL] = toolName
    d[kEXPORT_DICT_TOOL_VER] = toolVersion

    # add the data in kwargs to the export dict - NOTE: this is done using setdefault so its not possible
    # to clobber existing keys by specifying them as kwargs...
    for key, value in kwargs.items():
        d.setdefault(key, value)

    return d


class PresetManager(object):
    def __init__(self, tool, ext=DEFAULT_XTN):
        self.tool = tool
        self.extension = ext

    def getPresetDirs(self, locale=GLOBAL):
        '''
        returns the base directory for a given tool's preset files
        '''
        return getPresetDirs(locale, self.tool)

    def presetPath(self, name, locale=GLOBAL):
        return Preset(locale, self.tool, name, self.extension)

    def findPreset(self, name, startLocale=LOCAL):
        return Preset(*dataFromPresetPath(findPreset(name, self.tool, self.extension, startLocale)))

    def listPresets(self, locale=GLOBAL):
        return listPresets(locale, self.tool, self.extension)

    def listAllPresets(self, localTakesPrecedence=False):
        return listAllPresets(self.tool, self.extension, localTakesPrecedence)


class Preset(ValvePath):
    '''
    provides a convenient way to write/read and otherwise handle preset files
    '''

    def __new__(cls, locale, tool, name, ext=DEFAULT_XTN):
        '''
        locale should be one of either GLOBAL or LOCAL object references.  tool is the toolname
        used to refer to all presets of that kind, while ext is the file extension used to
        differentiate between multiple preset types a tool may have
        '''
        name = scrubName(name, exceptions='./')
        path = getPresetPath(name, tool, ext, locale)
        if path is None:
            path = presetPath(locale, tool, name, ext)

        return ValvePath.__new__(cls, path)

    def __init__(self, locale, tool, name, ext=DEFAULT_XTN):
        self.locale = locale
        self.tool = tool

    @staticmethod
    def FromFile(filepath):
        return Preset(*dataFromPresetPath(filepath))

    FromPreset = FromFile

    def up(self, levels=1):
        return ValvePath(self).up(levels)

    def other(self):
        '''
        returns the "other" locale - ie if teh current instance points to a GLOBAL preset, other()
        returns LOCAL
        '''
        if self.locale == GLOBAL:
            return LOCAL
        else:
            return GLOBAL

    def copy(self):
        '''
        copies the current instance from its current locale to the "other" locale. handles all
        perforce operations when copying a file from one locale to the other.  NOTE: the current
        instance is not affected by a copy operation - a new Preset instance is returned
        '''
        other = self.other()
        otherLoc = getPresetDirs(other, self.tool)[0]

        dest = otherLoc / self[-1]
        destP4 = None
        addToP4 = False

        # in this case, we want to make sure the file is open for edit, or added to p4...
        if other == GLOBAL:
            destP4 = P4File(dest)
            if destP4.managed():
                destP4.edit()
                print('opening %s for edit' % dest)
            else:
                addToP4 = True

        ValvePath.copy(self, dest)
        if addToP4:
            # now if we're adding to p4 - we need to know if the preset is a pickled preset - if it is, we need
            # to make sure we add it as a binary file, otherwise p4 assumes text, which screws up the file
            try:
                self.unpickle()
                destP4.add(type=P4File.BINARY)
            except Exception as e:
                # so it seems its not a binary file, so just do a normal add
                print('exception when trying to unpickle - assuming a text preset', e)
                destP4.add()
            print('opening %s for add' % dest)

        return Preset(self.other(), self.tool, self.name(), self.extension)

    def move(self):
        '''
        moves the preset from the current locale to the "other" locale.  all instance variables are
        updated to point to the new location for the preset
        '''
        newLocation = self.copy()

        # delete the file from disk - and handle p4 reversion if appropriate
        self.delete()

        # now point instance variables to the new locale
        self.locale = self.other()

        return self.FromFile(newLocation)

    def rename(self, newName):
        '''
        newName needs only be the new name for the preset - extension is optional.  All perforce
        transactions are taken care of.  all instance attributes are modified in place

        ie: a = Preset(GLOBAL, 'someTool', 'presetName')
        a.rename('the_new_name)
        '''
        if not newName.endswith(self.extension):
            newName = '%s.%s' % (newName, self.extension)

        return ValvePath.rename(self, newName, True)

    def getName(self):
        return ValvePath(self).set_extension()[-1]

# end
