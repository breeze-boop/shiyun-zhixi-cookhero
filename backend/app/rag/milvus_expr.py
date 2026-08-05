from __future__ import annotations


def milvus_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def split_milvus_and_expression(expr: str) -> list[str]:
    clauses: list[str] = []
    start = 0
    in_string = False
    escaped = False
    index = 0
    marker = " and "
    while index < len(expr):
        char = expr[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if expr.startswith(marker, index):
            clause = expr[start:index].strip(" ()")
            if clause:
                clauses.append(clause)
            index += len(marker)
            start = index
            continue
        index += 1
    clause = expr[start:].strip(" ()")
    if clause:
        clauses.append(clause)
    return clauses
