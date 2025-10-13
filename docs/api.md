# API 文檔

## 檢測器

### UltraStrictDetector
嚴格格式檢測器，基於 PUA 字符和頓號的精確匹配。

### HybridLevelSymbolDetector  
三層混合檢測器，結合規則和 BERT 分類。

### IntelligentHybridDetector
自適應檢測器，能夠從文檔中學習層級規則。

## 分析器

### analyze_filtered_dataset
分析整個數據集的統計信息。

### process_single_file
處理單個 JSON 文檔。

詳細使用方法請參考源代碼中的文檔字符串。
