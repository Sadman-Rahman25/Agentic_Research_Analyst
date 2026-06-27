"""Pytest configuration — adds project root to sys.path so tests can `from src...`"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))