#!/usr/bin/env python3
"""L-chunk 安裝腳本"""

from setuptools import setup, find_packages
from pathlib import Path

# 讀取 README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="lchunk",
    version="0.1.0",
    description="法律文檔層級符號檢測系統",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="L-chunk Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.8.0",
        "transformers>=4.57.0",
        "numpy>=1.26.4,<2.0.0",
        "pandas>=2.3.3",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.10.7",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ]
    },
    entry_points={
        "console_scripts": [
            "lchunk-detect=lchunk.detectors.intelligent_hybrid:main",
            "lchunk-train=lchunk.training.bert_trainer:main",
            "lchunk-compare=lchunk.training.model_comparison:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
