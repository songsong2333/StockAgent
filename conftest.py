"""pytest 根 conftest: 确保项目根在 sys.path(无 src layout 时让 tests/ 能 import strategy/collector)。"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
