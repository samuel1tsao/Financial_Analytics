import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

modules = [
    '_constants',
    '_data_worker',
    '_ml_worker',
    '_rl_worker',
    '_sim_worker',
    '_scoring_worker',
    'backend.vector_encoder',
]

for mod in modules:
    try:
        __import__(mod)
        print(f"OK: {mod}")
    except ImportError as e:
        print(f"FAIL (ImportError): {mod} -> {e}")
    except Exception as e:
        print(f"FAIL (Other Error): {mod} -> {e}")
