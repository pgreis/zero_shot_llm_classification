import json
import yaml

def read_json(file_path:str):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_yaml(file_path:str):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)