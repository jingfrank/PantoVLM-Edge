#!/bin/bash
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

CFG_FILE="config/config_prod.yaml"
if [ ! -f "$CFG_FILE" ] && [ -f "config_prod.yaml" ]; then
    CFG_FILE="config_prod.yaml"
fi

python3 main.py "$CFG_FILE"
