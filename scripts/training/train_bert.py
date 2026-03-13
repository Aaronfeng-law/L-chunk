#!/usr/bin/env python3
"""BERT 訓練腳本"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from lchunk.training.bert_trainer import main

if __name__ == "__main__":
    main()
