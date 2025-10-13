# 項目重組報告

**重組時間**: 2025-10-13 14:59:02
**源目錄**: /home/soogoino/Publics/Projects/L-chunk
**目標目錄**: L-chunk

## 📁 目錄結構變更

### 核心源代碼 (src/lchunk/)
- `detectors/`: 所有檢測器模組
- `analyzers/`: 分析和分割器
- `training/`: 訓練相關代碼
- `models/`: 數據模型定義
- `utils/`: 工具函數

### 數據組織
- `data/processed/filtered/`: 過濾後的數據
- `data/samples/`: 樣本數據
- `data/training/`: 訓練數據

### 模型和結果
- `models/bert/`: BERT 模型文件
- `results/`: 各種檢測和分析結果


## ✨ 改進項目

1. **清晰的模組分離**: 檢測器、分析器、訓練分開
2. **標準 Python 包結構**: 符合 PEP 8 和最佳實踐
3. **便捷的命令行工具**: 統一的入口點
4. **完整的安裝配置**: setup.py 和 pyproject.toml
5. **相對導入**: 避免路徑問題
## 📜 命令行工具


## 🚀 使用新結構

```bash
# 進入新目錄
cd L-chunk

# 安裝開發模式
pip install -e .

# 使用命令行工具
lchunk-detect data/samples/
```

## ⚠️ 注意事項

1. 所有相對導入已更新
2. 腳本路徑已調整
3. 保持了原有功能完整性
4. 添加了適當的 __init__.py 文件