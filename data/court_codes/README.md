# 台灣法院資料夾名稱與法院代碼對照表

## 概述
本專案提取了台灣法院系統中所有資料夾名稱（包含法院名稱和類型）與對應的四字母法院代碼的對照關係。

## 生成的檔案

### 原始完整資料

1. **`court_mapping_grouped.json`** - 完整的分組法院資料（原始版本）
   - 按法院分組的完整對照表
   - 包含所有詳細資訊和案件類型
   - 適合需要完整資料結構的應用

### 分離的映射文件（推薦用於快速查詢）

1. **`code_to_fullcourt.json`** - 完整法院資訊映射
   - 包含所有詳細資料和案件類型資訊
   - 支援基礎法院代碼和子法院代碼查詢
   - 208 個映射項目
   - 最詳細的映射版本

2. **`code_to_base_court.json`** - 基礎法院代碼映射
   - 基礎法院代碼到法院名稱的簡單映射
   - 用於快速查詢基礎法院名稱
   - 75 個映射項目
   - 格式：`"SJE": "三重簡易庭"`

3. **`code_to_sub_court.json`** - 子法院代碼映射
   - 子法院代碼到完整法院名稱的簡單映射
   - 用於快速查詢特定案件類型的法院
   - 135 個映射項目
   - 格式：`"SJEM": "三重簡易庭刑事"`

### 舊版文件（保留供參考）

- `court_data.json` - 原始提取的資料
- `court_mapping_organized.json` - 組織化的對照表

## 資料結構（推薦使用 court_mapping_grouped.json）

每個法院條目包含：
- `court_name`: 法院基礎名稱（如：鳳山簡易庭）
- `jurisdiction`: 法院層級（最高法院、高等法院、地方法院、簡易庭、憲法法庭、懲戒法院、司法院、專業法院）
- `case_types`: 案件類型物件
  - 每個案件類型包含：
    - `court_code`: 四字母代碼
    - `full_name`: 完整資料夾名稱

## 範例

```json
"鳳山簡易庭": {
  "court_name": "鳳山簡易庭",
  "jurisdiction": "簡易庭",
  "case_types": {
    "刑事": {
      "court_code": "FSEM",
      "full_name": "鳳山簡易庭刑事"
    },
    "民事": {
      "court_code": "FSEV",
      "full_name": "鳳山簡易庭民事"
    }
  }
}
```

## 統計資料

- **總資料夾數**: 135個（含案件類型分類）
- **總獨立法院數**: 79個（合併案件類型後）
- **總檔案數**: 80,291個
- **提取日期**: 2025年1月28日

## 檔案命名模式

檔案名稱格式：`{法院代碼},{年度},{案件類型},{案件編號},{日期},{版本}.json`

例如：`SJEV,113,重小,1311,20250109,1.json`
- SJEV: 三重簡易庭民事
- 113: 民國113年
- 重小: 案件類型
- 1311: 案件編號
- 20250109: 2025年1月9日
- 1: 版本號

## 使用方式

### 快速映射查詢（推薦）

使用分離的映射文件進行快速查詢：

```python
import json

# 1. 快速查詢基礎法院名稱
with open('code_to_base_court.json', 'r', encoding='utf-8') as f:
    base_mapping = json.load(f)['mapping']

court_name = base_mapping.get('SJE')  # "三重簡易庭"

# 2. 快速查詢子法院完整名稱
with open('code_to_sub_court.json', 'r', encoding='utf-8') as f:
    sub_mapping = json.load(f)['mapping']

full_name = sub_mapping.get('SJEM')  # "三重簡易庭刑事"

# 3. 獲取完整法院資訊
with open('code_to_fullcourt.json', 'r', encoding='utf-8') as f:
    full_mapping = json.load(f)['mapping']

court_info = full_mapping.get('SJEM')
# 包含 court_name, jurisdiction, case_type, full_name 等完整資訊
```

### 使用映射管理器（完整示例）

查看 `mapping_usage_example.py` 文件以獲得完整的使用示例，包括：
- 基礎法院代碼查詢
- 子法院代碼查詢
- 完整法院資訊查詢
- 法院名稱搜尋
- 法院層級資訊查詢

運行示例：

```bash
python mapping_usage_example.py
```

### 原始數據查詢（完整結構）

```python
import json

with open('court_mapping_grouped.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 根據法院基礎名稱查找所有案件類型
court_base_name = "鳳山簡易庭"
court_info = data["courts"][court_base_name]
print(f"法院名稱: {court_info['court_name']}")
print(f"法院層級: {court_info['jurisdiction']}")

# 查看所有案件類型和代碼
for case_type, info in court_info["case_types"].items():
    print(f"{case_type}: {info['court_code']} -> {info['full_name']}")

# 根據代碼查找法院名稱  
court_code = "FSEV"
full_court_name = data["code_to_court"][court_code]
print(f"代碼 {court_code} 對應: {full_court_name}")
```

## 腳本說明

1. `extract_court_data.py`: 提取原始資料
   - 從法院資料夾中提取檔案名稱和法院代碼
   - 提供詳細的處理進度和統計資訊
   - 驗證法院代碼格式

2. `organize_court_data.py`: 組織化資料並分類
   - 生成 `court_mapping_organized.json`（包含file_count）
   - 生成 `court_mapping_grouped.json`（推薦使用，按法院分組）
   - 提供法院類型和層級分類
