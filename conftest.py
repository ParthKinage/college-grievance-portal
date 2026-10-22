"""
pytest setup: make "src" importable even when pytest is run from another folder
(for example from the parent folder with "pytest complaint-router").
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
