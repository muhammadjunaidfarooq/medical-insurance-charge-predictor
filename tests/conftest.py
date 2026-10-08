import sys
from pathlib import Path

# Let tests import main.py and predictor.py from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
