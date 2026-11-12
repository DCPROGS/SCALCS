"""scalcs must be a regular package, not a namespace package.

Without scalcs/__init__.py, Python treats scalcs as a namespace package and
merges every directory on the path that contains a scalcs folder. A test
pointed at one copy of SCALCS - an older release, a branch under review - then
silently imports any module that copy lacks from another copy on the path: an
installed release in site-packages, a checkout on PYTHONPATH.

With __init__.py the first scalcs found on the path is used whole, and a
module it lacks is an ImportError.

What this does NOT prevent: an editable install (pip install -e) registers its
own finder, which resolves any scalcs.* name directly to its own source tree
and ignores which copy of scalcs was imported. It is consulted after the normal
path search, so it still fills in any module the first copy lacks. That is what
produced the false pass during the reproduction of Colquhoun & Lape (2012), and
nothing inside the package can stop it. To test one copy of SCALCS while an
editable install of another exists, use a clean environment, or remove that
finder from sys.meta_path and then check every imported module's __file__.
"""
import os
import shutil
import subprocess
import sys

import pytest

import scalcs
from scalcs import version


def test_scalcs_is_a_regular_package():
    assert scalcs.__file__ is not None
    assert os.path.basename(scalcs.__file__) == '__init__.py'
    assert type(scalcs.__path__) is list         # a namespace package's is not


#: Run in a subprocess with two copies of scalcs on the path. Any editable-
#: install finder is removed first, so that only the path search is tested -
#: see the module docstring for why that finder is out of the package's reach.
_TWO_COPIES = '''
import sys
STANDARD = ('_frozen_importlib', '_frozen_importlib_external')
sys.meta_path = [f for f in sys.meta_path
                 if getattr(type(f), '__module__', '').startswith(STANDARD)
                 or getattr(f, '__module__', '').startswith(STANDARD)]
for name in [m for m in sys.modules if m.split('.')[0] == 'scalcs']:
    del sys.modules[name]
import scalcs.qmatlib
print('qmatlib came from', scalcs.qmatlib.__file__)
'''


def test_another_copy_on_the_path_cannot_fill_in_modules(tmp_path):
    """Two copies of scalcs on the path: first a copy laid out like this one
    but without qmatlib, then the real one. Importing scalcs.qmatlib must
    fail, rather than quietly come from the second copy."""
    real_package = os.path.dirname(os.path.abspath(version.__file__))
    real_root = os.path.dirname(real_package)
    first = tmp_path / 'scalcs'
    first.mkdir()
    for name in ('__init__.py', 'version.py'):
        if os.path.exists(os.path.join(real_package, name)):
            shutil.copy(os.path.join(real_package, name), first / name)

    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join([str(tmp_path), real_root]))
    run = subprocess.run([sys.executable, '-c', _TWO_COPIES], cwd=tmp_path,
                         env=env, capture_output=True, text=True)
    assert run.returncode != 0, run.stdout
    assert "No module named 'scalcs.qmatlib'" in run.stderr, run.stderr


def test_the_version_is_on_the_package():
    assert scalcs.__version__ == version.__version__


def test_importing_the_package_imports_nothing_heavy():
    """`import scalcs` must stay cheap: scalcs.schemes, for one, relies on
    nothing importing matplotlib behind its back."""
    code = ('import sys, scalcs; '
            'print(sorted(m for m in ("numpy", "scipy", "matplotlib") '
            'if m in sys.modules))')
    run = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True)
    assert run.stdout.strip() == '[]'
