#!/usr/bin/env python3
"""模型比較腳本"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from lchunk.training.model_comparison import main

if __name__ == "__main__":
    main()
