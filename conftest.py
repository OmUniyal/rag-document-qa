# conftest.py — project root
# Adds the project root to sys.path so 'src' is importable from anywhere
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))