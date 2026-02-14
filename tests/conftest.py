import sys
from pathlib import Path

# Add app directory to path so tests can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
