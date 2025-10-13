#!/usr/bin/env python3
"""檢測演示腳本"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lchunk.detectors.intelligent_hybrid import main

if __name__ == "__main__":
    main()
