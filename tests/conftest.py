import os
import sys
import pytest

# Add the project root directory to the Python path
# This allows importing modules from src in the tests
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Also add the src directory to the path
# This allows tests to import modules with 'from src.module import X'
src_dir = os.path.join(root_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
