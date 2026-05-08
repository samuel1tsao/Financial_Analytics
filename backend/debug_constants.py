import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _constants
print("Attributes in _constants:")
for attr in sorted(dir(_constants)):
    if not attr.startswith("__"):
        print(attr)
