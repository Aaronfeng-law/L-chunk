# API 文檔

## 檢測器 (src/lchunk/detectors)

### UltraStrictDetector

- 角色：第一層「終極嚴格規則」檢測器
- 能力：依據 PUA 符號與頓號模式進行 100% 確認的層級符號辨識
- 方法摘要：`detect_ultra_strict(lines: List[str]) -> List[HybridDetectionResult]`

### HybridLevelSymbolDetector

- 角色：三層混合檢測主體 (嚴格 → 軟規則 → BERT)
- 主要方法：
  - `detect_hybrid_markers(lines: List[str])`：依序套用嚴格規則、軟規則與 BERT 模型
  - `detect_hierarchy_levels()`：統整偵測結果，產生層級映射
- 依賴：若提供 `model_path`，自動載入微調後的 BERT 分類器

### AdaptiveHybridDetector

- 角色：自適應層級學習與全文應用
- 功能：
  - `process_single_file(path: Path)`：完成結構分析、規則學習、全文檢測、行級分塊與統計
  - `process_sample_directory(path: Path, output_dir: Path | None, max_files: int | None)`：批次處理並輸出 Markdown/JSON 報告
  - `generate_batch_report(results, region_stats, output_dir)`：產生 Markdown/JSON 彙整報告
- 日誌：使用 Python `logging`，詳細輸出預設寫入 `logs/adaptive_detector.log`

## 分析器 (src/lchunk/analyzers)

### analyze_filtered_dataset

- 描述：分析 `data/processed/filtered` 資料集，產生章節統計與整體摘要
- 回傳：統計字典，可用於模型比較或報告撰寫

### process_single_file

- 描述：`splitter.process_single_file(path)` 分析單一 JSON 文檔，產出章節結構資訊
- 回傳：`success: bool`, `result: Dict`，供 `AdaptiveHybridDetector` 後續步驟使用

## CLI 工具

### scripts/markdown/md_converter.py

- 用途：將行級分塊結果轉換為排版良好的 Markdown
- 執行：`uv run scripts/markdown/md_converter.py <input> --output-dir <dir>`

### run_adaptive_detector.py

- 用途：命令列執行自適應檢測 (單檔或整批)
- 主要參數：
  - `input_path`：JSON 檔或資料夾
  - `--model-path`：BERT 權重位置（預設為專案模型資料夾）
  - `--output-dir`：報告輸出資料夾（Markdown/JSON）
  - `--log-file`：詳細日誌輸出位置
  - `--max-files`：批量處理上限
  - `--verbose`：在終端顯示進度

