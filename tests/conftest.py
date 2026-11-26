import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skyrace import sitl  # noqa: E402

needs_sitl = pytest.mark.skipif(
    not sitl.available(),
    reason="SITL library not built (python scripts/build_sitl.py --skyforge-fc <checkout>); CI builds it")


@pytest.fixture(scope="session")
def sitl_lib():
    return sitl.load()
