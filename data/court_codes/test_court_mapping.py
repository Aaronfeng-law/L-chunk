import json 


with open("data/court_codes/court_mapping_grouped.json", "r", encoding="utf-8") as f:
    court_data = json.load(f)

# print(court_data)

test_court_code = "TPDV"

for base_court_name, court_info in court_data.get("courts", {}).items():
    case_types = court_info.get("case_types", {})
    for case_type, type_info in case_types.items():
        court_code = type_info.get("sub_court_code")
        full_name = type_info.get("full_name")
        if court_code == test_court_code:
            print(f"Found base court name {base_court_name}")
            print(f"Found sub court code {test_court_code}: {full_name}")
            
