# API 文檔

本專案採用統一的處理管線架構 (Pipeline Architecture)，由 `pipeline.py` 作為核心互動入口，並將處理流程拆分為多個獨立模組。

## 核心管線 (src/lchunk/pipeline.py)

### PipelineOrchestrator
- **角色**：統一的資料處理協調者
- **職責**：依序調用 Splitter → UltraStrict → Hybrid → Adaptive，並將狀態與結果封裝於 `JudgmentArtifact` 物件中傳遞。
- **主要方法**：
  - `process_file(input_path: Path) -> JudgmentArtifact | None`：執行單一檔案的完整處理管線。

### JudgmentArtifact
- **角色**：生命週期內的單一真相來源 (Single Source of Truth)
- **屬性**：
  - `metadata`：文檔基礎資訊
  - `full_lines`：包含特徵與檢測結果的 `DocumentLine` 列表
  - `sections`：文章段落分割結果
  - `hierarchy_tree`：自適應檢測產生的階層式結構樹

## 檢測器 (src/lchunk/detectors)

### UltraStrictDetector
- **角色**：首層「嚴格規則」檢測器 (Phase 2)
- **能力**：依據全形數字、PUA 符號與頓號模式進行 100% 確認的層級符號辨識

### HybridLevelSymbolDetector
- **角色**：混合檢測主體 (Phase 3)
- **能力**：套用軟規則與 BERT 模型填補嚴格規則未覆蓋的地帶

### AdaptiveHybridDetector
- **角色**：自適應層級學習與全文推廣 (Phase 4)
- **能力**：學習文檔特定的層級編碼規則，將其應用於全文，最後建構出嵌套的 `hierarchy_tree` 並產出基於行的分塊結果 (`LineBasedChunk`)。

## 分析器 (src/lchunk/analyzers)

### splitter_refactor.py
- **角色**：文獻結構分割器 (Phase 1)
- **主要方法**：
  - `split_judgment_document(input_path: Path) -> JudgmentArtifact | None`：讀取 JSON 並將文檔分割成表頭、主文、事實、理由等段落。

## 轉換器 (src/lchunk/converters)

### md_exporter.py / md_converter.py
- **角色**：Markdown 產生模組
- **主要類別**：
  - `PipelineMarkdownConverter`：讀取 `JudgmentArtifact.hierarchy_tree`（或 Debug JSON），渲染出具有階層縮排、斜體日期與附錄標記的高品質 Markdown。
- **主要功能**：
  - `export_to_markdown(...)`：API 呼叫介面，支援批次轉換與格式相容。

## 統一 CLI 工具 (`pipeline.py`)

`pipeline.py` 是唯一建議使用的命令列入口，整合了各種操作模式：

```bash
# 1. 執行核心管線及除錯 JSON 保存
uv run pipeline.py <input> --full --save

# 2. 生成 Markdown
uv run pipeline.py <input> --markdown

# 3. 讀取儲存的 Debug JSON 直接轉成 Markdown
uv run pipeline.py output/debug/ --markdown --md-from-debug

# 4. 指定特定模型路徑
uv run pipeline.py <input> --markdown --model-path <path>
```

