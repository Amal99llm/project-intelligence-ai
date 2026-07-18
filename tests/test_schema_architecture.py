"""Architecture guard: production DDL belongs only in schema_management."""

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_MODULE = ROOT / "modules" / "schema_management.py"
DDL_SQL = re.compile(
    r"\b(?:(?:CREATE|ALTER|DROP|TRUNCATE)\s+(?:TABLE|INDEX|SCHEMA)|REINDEX)\b",
    re.IGNORECASE,
)
DDL_METHODS = {
    "create_all", "drop_all", "create_table", "drop_table",
    "create_index", "drop_index", "add_column", "alter_column",
}


def _production_python_files():
    yield ROOT / "app.py"
    yield ROOT / "scheduler.py"
    yield ROOT / "config.py"
    yield from (ROOT / "modules").glob("*.py")
    yield from (ROOT / "scripts").glob("*.py")


def test_no_production_ddl_outside_schema_management():
    violations = []
    for path in _production_python_files():
        if path == SCHEMA_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in DDL_METHODS:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} calls {node.func.attr}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and DDL_SQL.search(node.value):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno} contains raw DDL")
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name == "_schema_initialization_lock":
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} accesses the private schema lock")
    assert not violations, "Schema DDL must use apply_schema_changes():\n" + "\n".join(violations)


def test_only_initializer_script_calls_public_schema_entry_point():
    callers = []
    for path in _production_python_files():
        if path == SCHEMA_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (
                    node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute)
                    else None
                )
                if name == "apply_schema_changes":
                    callers.append(path.relative_to(ROOT).as_posix())
    assert callers == ["scripts/init_db.py"]
