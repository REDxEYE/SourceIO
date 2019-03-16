import inspect
import os


class GoodException(Exception):
    '''
    Good exceptions are just a general purpose way of breaking out of loops and whatnot. Basically anytime an exception is
    needed to control code flow and not indicate an actual problem using a GoodException makes it a little more obvious what
    the code is doing in the absence of comments
    '''
    pass


BreakException = GoodException


class Callback(object):
    '''
    Simple callable object for when you need to "bake" temporary args into a
    callback - useful mainly when creating callbacks for dynamicly generated UI items.
    '''

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args):
        return self.func(*self.args, **self.kwargs)


def removeDupes(iterable):
    '''
    '''
    unique = set()
    newIterable = iterable.__class__()
    for item in iterable:
        if item not in unique: newIterable.append(item)
        unique.add(item)

    return newIterable


def makeUniqueName(name, namesList):
    '''
    Increment the name with the (n) suffix until it doesn't match any names in namesList.
    Handles names with or without file extensions.
    Max 1000 iterations.
    '''
    basename, ext = os.path.splitext(name)
    n = 1
    while (True) and (n < 1000):
        testName = '{0} ({1}){2}'.format(basename, n, ext)
        if testName not in namesList:
            return testName
        n += 1
    assert ('Failed to create a unique name after 1000 iterations.')


def iterBy(iterable, count):
    '''
    returns an generator which will yield "chunks" of the iterable supplied of size "count".  eg:
    for chunk in iterBy( range( 7 ), 3 ): print chunk

    results in the following output:
    [0, 1, 2]
    [3, 4, 5]
    [6]
    '''
    cur = 0
    i = iter(iterable)
    while True:
        try:
            toYield = []
            for n in range(count): toYield.append(i.next())
            yield toYield
        except StopIteration:
            if toYield: yield toYield
            break


def findMostRecentDefinitionOf(variableName):
    '''
    '''
    try:
        fr = inspect.currentframe()
        frameInfos = inspect.getouterframes(fr, 0)

        # in this case, walk up the caller tree and find the first occurance of the variable named <variableName>
        for frameInfo in frameInfos:
            frame = frameInfo[0]
            var = None

            if var is None:
                try:
                    var = frame.f_locals[variableName]
                    return var
                except KeyError:
                    pass

                try:
                    var = frame.f_globals[variableName]
                    return var
                except KeyError:
                    pass

    # NOTE: this method should never ever throw an exception...
    except:
        pass


def getArgDefault(function, argName):
    '''
    returns the default value of the given named arg.  if the arg doesn't exist,
    or a NameError is raised.  if the given arg has no default an IndexError is
    raised.
    '''
    args, va, vkw, defaults = inspect.getargspec(function)
    if argName not in args:
        raise NameError("The given arg does not exist in the %s function" % function)

    args.reverse()
    idx = args.index(argName)

    try:
        return list(reversed(defaults))[idx]
    except IndexError:
        raise IndexError("The function %s has no default for the %s arg" % (function, argName))


def findInFile(file, search):
    '''
    Returns the number of times the search string is found in the file, or None.
    '''
    fh = open(file)
    cfg = fh.read()
    fh.close()

    count = 0
    found = cfg.find(search)
    if found > -1:
        count = 1
        while True:
            found = cfg.find(search, found + 1)
            if found == -1:
                break
            count += 1
    if count:
        return count
    else:
        return None


def replaceInFile(file, search, replace):
    '''
    In the given file, replace the value of "search" string with the value of "replace" string.
    '''
    from tempfile import mkstemp
    from shutil import move

    # Create a temp file
    fh, tempPath = mkstemp(suffix='txt')
    tempFile = open(tempPath, 'w')
    oldFile = open(file)
    for line in oldFile:
        tempFile.write(line.replace(search, replace))
    # Close the temp file
    tempFile.close()
    os.close(fh)
    oldFile.close()
    # Delete the original file
    os.remove(file)
    # Move new file
    move(tempPath, file)


def removeLineInFile(file, search):
    '''
    In the given file, remove all lines that equal the search string.
    Do not include linefeeds in the search string (they are stripped).
    Returns the number of lines removed.
    '''
    from tempfile import mkstemp
    from shutil import move

    # Create a temp file
    fh, tempPath = mkstemp(suffix='txt')
    tempFile = open(tempPath, 'w')
    oldFile = open(file)
    count = 0
    for line in oldFile:
        # write out only the lines that don't equal the search string
        if line.rstrip(os.linesep) != search:
            tempFile.write(line)
        else:
            count += 1
    # Close the temp file
    tempFile.close()
    os.close(fh)
    oldFile.close()
    # Delete the original file
    os.remove(file)
    # Move new file
    move(tempPath, file)

    return count


def removeLineInFileThatContains(file, search):
    '''
    In the given file, remove all lines that contains the search string.
    Do not include linefeeds in the search string (they are stripped).
    Returns the number of lines removed.
    '''
    from tempfile import mkstemp
    from shutil import move

    # Create a temp file
    fh, tempPath = mkstemp(suffix='txt')
    tempFile = open(tempPath, 'w')
    oldFile = open(file)
    count = 0
    for line in oldFile:
        # write out only the lines that don't contain the search string
        if search not in line.rstrip(os.linesep):
            tempFile.write(line)
        else:
            count += 1
    # Close the temp file
    tempFile.close()
    os.close(fh)
    oldFile.close()
    # Delete the original file
    os.remove(file)
    # Move new file
    move(tempPath, file)

    return count

# end
