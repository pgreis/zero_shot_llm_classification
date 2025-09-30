import os
from src.helpers import read_yaml, read_json

CONFIG = read_yaml(os.path.join("utils", "config.yaml"))
TEMPLATES = read_yaml(os.path.join("utils", "prompts.yaml"))
SYSTEM_TEMPLATE = TEMPLATES["classification_v1"]["system"]
USER_TEMPLATE = TEMPLATES["classification_v1"]["user"]
OUTPUT_SCHEMA = read_json("utils/output_schema.json")