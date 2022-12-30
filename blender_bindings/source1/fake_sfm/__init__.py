import importlib
import importlib.util
import logging
import sys
from importlib.abc import Loader, MetaPathFinder
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from pathlib import Path

logger = logging.getLogger('SFMHook')


class HookedImporter(MetaPathFinder, Loader):
    def __init__(self, hooks=None):
        self.hooks = hooks

    def find_spec(self, name, path, target=None):
        if name not in self.hooks:
            return None

        spec = importlib.util.spec_from_loader(name, self)
        return spec

    def create_module(self, spec):
        # req'd in 3.6
        logger.info('hooking import: %s', spec.name)
        my_spec = importlib.util.spec_from_loader(spec.name, loader=None)
        module = importlib.util.module_from_spec(my_spec)
        # module = importlib.util._Module(spec.name)
        mod = self.hooks[spec.name]
        for attr in dir(mod):
            if attr.startswith('__'):
                continue
            module.__dict__[attr] = getattr(mod, attr)
        return module

    def exec_module(self, module):
        # module is already loaded (imported by line `import idb` above),
        # so no need to re-execute.
        #
        # req'd in 3.6.
        return

    def install(self):
        if isinstance(sys.meta_path[0], HookedImporter):
            return
        sys.meta_path.insert(0, self)


class dummy:
    pass


def load_script(script_path: Path):
    from . import sfm, sfm_utils, vs
    hooks = {
        'vs': vs,
        'sfm': sfm.SFM(),
    }
    importer = HookedImporter(hooks=hooks)
    importer.install()
    assert script_path.exists()
    with script_path.open('r') as f:
        script_data = f.read()

    refactoring_tool = RefactoringTool(fixer_names=get_fixers_from_package('lib2to3.fixes'))
    node3 = refactoring_tool.refactor_string(script_data + '\n', 'script')

    compiled = compile(str(node3), str(script_path), 'exec')
    exec(compiled, {'sfm': sfm.SFM(), 'sfmUtils': sfm_utils.SfmUtils(), 'print': sfm.sfm.sfm_logger.print})
