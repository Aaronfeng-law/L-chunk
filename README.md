# L-chunk: 法律文檔層級符號檢測系統

L-chunk 旨在自動解析法律文檔的層級結構，結合規則與 BERT 模型，輸出易於閱讀及後續分析的 Markdown 報告。

## 🏗️ 專案結構

```text
L-chunk/
├── src/lchunk/          # 核心程式碼 (檢測器、分析器、模型)
├── scripts/             # 指令稿 (含 Markdown 轉換工具)
├── data/                # 原始/處理後數據與樣本
├── models/              # 訓練後模型與檢測器設定
├── output/              # 產出 (Markdown、報告、偵測結果)
├── results/             # 評估報告與比較結果
├── tests/               # 單元與整合測試
└── docs/                # 補充文件與 API 說明
```

## 🚀 快速開始

### 先決條件

- Python 3.10 以上版本
- 建議使用 [uv](https://github.com/astral-sh/uv) 管理虛擬環境與依賴

### 安裝步驟

```bash
# 選用 uv 建立虛擬環境並安裝專案 (建議)
uv venv
source .venv/bin/activate
uv pip install -e .

# 或使用 pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Python API 範例

```python
from lchunk import AdaptiveHybridDetector

detector = AdaptiveHybridDetector()
result = detector.process_single_file("path/to/document.json")

print(result.learning_region)
print(len(result.line_based_chunks))
```

### CLI：Markdown 轉換器

利用 `scripts/markdown/md_converter.py` 將偵測結果轉換成 Markdown：

```bash
uv run scripts/markdown/md_converter.py \
    data/samples/TPDM,109,易,187,20250116,1.json \
    --output-dir output/markdown
```

> 預設會輸出到 `output/markdown/`。若輸入為目錄，可搭配 `--max-files` 限制轉換數量。

### CLI：自適應檢測器

使用 `run_adaptive_detector.py` 可直接在終端執行自適應檢測（支援單檔與整批處理）：

```bash
uv run run_adaptive_detector.py \
    data/samples \
    --output-dir output/adaptive \
    --log-file logs/adaptive_detector.log
```

- 預設輸出：Markdown 與 JSON 檔案寫入 `output/adaptive/`，詳細日誌寫入 `logs/adaptive_detector.log`
- 常用參數：`--model-path` 指定權重、`--max-files` 限制批次數量、`--verbose` 顯示即時進度
- 匯出格式：`--export-format human|machine`（預設 human）。選擇 `machine` 時會輸出 JSON，並將 Lv -1 內容與對應上層 (Lv ≥ 1) 合併
- 若模型路徑不存在，流程會自動退化為純規則檢測並於日誌提示

## 🎯 核心特性

- **三層檢測流程**：嚴格規則 → 軟規則 → BERT 分類
- **自適應層級學習**：動態推斷段落層級與日期等特殊標記
- **行為維度輸出**：維持標頭、日期區塊逐行輸出，便於法律文件校對
- **批量與評估工具**：內建批次轉換、模型比較與性能報告

## 🔧 開發指南

```bash
# 執行測試
uv run pytest

# 重新訓練層級偵測模型 (範例)
uv run scripts/training/train_bert.py --config config/training.yaml
```

> 若調整 Markdown 轉換邏輯，可使用專案資料夾中的範例 JSON 重新產出報告檢查格式。

## 📊 目前性能

- **BERT 模型**：F1 = 0.9947 (最佳)
- **隨機森林**：F1 = 0.9497
- **邏輯回歸**：F1 = 0.9471

## 🤝 貢獻

歡迎先透過 Issue 討論，再提交 Pull Request。請確保程式碼通過測試並符合專案風格。

## 📄 許可證

GNU GPLv3
