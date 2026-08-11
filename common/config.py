"""AppConfig 配置加载工具（优先 openpyxl，避免 pandas 依赖冲突）。"""

import os


def convert_to_hex(version_text):
    numbers = [int(part) for part in str(version_text).split(".")]
    hex_bytes = [hex(number)[2:].zfill(2) for number in numbers]
    return "0x00" + "".join(hex_bytes)


def get_max_input_index_from_rows(rows):
    """rows: list of row tuples/lists；输入序号在第 4 列 (index 3)。"""
    values = []
    for row in rows:
        if len(row) <= 3:
            continue
        cell = row[3]
        if isinstance(cell, int):
            values.append(cell)
        elif isinstance(cell, float) and cell.is_integer():
            values.append(int(cell))
    return max(values) if values else 0


def _load_yaml_app_config(app_config_path, algo_id):
    import yaml

    with open(app_config_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}

    version_text = cfg.get("algo_version", "1.0.0")
    algo_version = int(convert_to_hex(version_text), 16)
    max_input_index = int(cfg.get("max_input_index", 0))
    return algo_id, algo_version, max_input_index


def _load_excel_with_openpyxl(app_config_path, algo_id):
    from openpyxl import load_workbook

    wb = load_workbook(app_config_path, data_only=True)
    ws = wb.active
    rows = []
    version_text = "1.0.0"
    for r in range(1, ws.max_row + 1):
        row = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
        rows.append(row)
        label = row[1] if len(row) > 1 else None
        value = row[2] if len(row) > 2 else None
        if label is not None and "版本" in str(label) and value is not None:
            version_text = str(value)

    algo_version = int(convert_to_hex(version_text), 16)
    max_input_index = get_max_input_index_from_rows(rows)
    return algo_id, algo_version, max_input_index


def _load_excel_app_config(app_config_path, algo_id):
    try:
        return _load_excel_with_openpyxl(app_config_path, algo_id)
    except ImportError:
        pass

    import pandas as pd

    config_df = pd.read_excel(app_config_path, header=None)
    version_text = "1.0.0"
    for idx in range(min(15, len(config_df))):
        label = config_df.iloc[idx, 1] if config_df.shape[1] > 1 else None
        value = config_df.iloc[idx, 2] if config_df.shape[1] > 2 else None
        if label is not None and "版本" in str(label) and value is not None:
            version_text = str(value)
            break

    algo_version = int(convert_to_hex(version_text), 16)
    values = []
    for row in config_df.iloc[:, 3].tolist():
        if isinstance(row, int):
            values.append(row)
    max_input_index = max(values) if values else 0
    return algo_id, algo_version, max_input_index


def load_app_config(app_config_path, algo_id):
    if not os.path.exists(app_config_path):
        version_text = "1.0.0"
        return algo_id, int(convert_to_hex(version_text), 16), 0

    ext = os.path.splitext(app_config_path)[1].lower()
    if ext in (".yaml", ".yml"):
        return _load_yaml_app_config(app_config_path, algo_id)
    return _load_excel_app_config(app_config_path, algo_id)
