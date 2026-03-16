import json
import sys
import os
from pathlib import Path
import dotenv
import argparse
from typing import Iterable
dotenv.load_dotenv()

def extract_unique_jcase(jcase_types_json, path_to_directory):
    """
    Extracts unique JCASE values from all JSON files in the specified directory.
    Returns a set of unique JCASE values.
    """
    
    with open(jcase_types_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        jcase_types = set(data.get("jcase_types", []))
        
    unique_jcases = set()
    
    for file_name in os.listdir(path_to_directory):
        if file_name.endswith('.json'):
            with open(os.path.join(path_to_directory, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                jcase_value = data.get('JCASE')
                # print(f"Processing file: {file_name}, JCASE value: {jcase_value}")
                if jcase_value is not None and jcase_value not in jcase_types:
                    unique_jcases.add(jcase_value)
    return unique_jcases
                    
def extract_unique_jtitle(jtitle_types_json, path_to_directory):
    
    """
    Extracts unique JTITLE values from all JSON files in the specified directory.
    Returns a set of unique JTITLE values.
    """
    with open(jtitle_types_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        jtitle_types = set(data.get("jtitle_types", []))
    
    unique_jtitles = set()
    
    for file_name in os.listdir(path_to_directory):
        if file_name.endswith('.json'):
            with open(os.path.join(path_to_directory, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                jtitle_value = data.get('JTITLE')
                # print(f"Processing file: {file_name}, JTITLE value: {jtitle_value}")
                if jtitle_value is not None and jtitle_value not in jtitle_types:
                    unique_jtitles.add(jtitle_value)
    return unique_jtitles

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique JCASE and JTITLE values from JSON files."
    )
    parser.add_argument(
        "--jcase-types-path",
        type=Path,
        default=os.getenv("JCASE_TYPES_PATH"),
        help="Path to the jcase_types.json file.",
    )
    parser.add_argument(
        "--jtitle-types-path",
        type=Path,
        default=os.getenv("JTITLE_TYPES_PATH"),
        help="Path to the jtitle_types.json file.",
    )
    parser.add_argument(
        "--json-directory",
        "-d",
        type=Path,
        default="data/samples/",
        help="Directory containing JSON judgment files, defaults to 'data/samples/'.",
    )
    return parser.parse_args(list(argv))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    unique_jcases = extract_unique_jcase(args.jcase_types_path, args.json_directory)
    unique_jtitles = extract_unique_jtitle(args.jtitle_types_path, args.json_directory)
    
    #append unique values to the respective JSON files
    if unique_jcases:
        with open(args.jcase_types_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            if "jcase_types" not in data:
                data["jcase_types"] = []
            data["jcase_types"].extend(list(unique_jcases))
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.truncate()
        
    if unique_jtitles:
        with open(args.jtitle_types_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            if "jtitle_types" not in data:
                data["jtitle_types"] = []
            data["jtitle_types"].extend(list(unique_jtitles))
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.truncate()
    

    print("Unique JCASE values not in jcase_types.json:")
    if unique_jcases:
        for jcase in unique_jcases:
            print(jcase)
    else:
        print("No new unique JCASE values found.")
    
    print("\nUnique JTITLE values:")
    if unique_jtitles:
        for jtitle in unique_jtitles:
            print(jtitle)
    else:
        print("No new unique JTITLE values found.")
