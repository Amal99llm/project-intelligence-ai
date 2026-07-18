import ast
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sqlalchemy import inspect

from modules.database import Base, engine, init_db
from modules.processor import process_backlog
from modules.project_repository import fetch_enriched_projects


ROOT = Path(__file__).resolve().parents[1]


def test_init_db_is_idempotent():
    Base.metadata.drop_all(engine)
    init_db()
    init_db()
    assert {"backlog_projects", "audit_log"}.issubset(inspect(engine).get_table_names())


def test_concurrent_init_attempts_are_serialized():
    Base.metadata.drop_all(engine)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: init_db(), range(16)))
    assert {"backlog_projects", "audit_log"}.issubset(inspect(engine).get_table_names())


def test_separate_initializer_processes_do_not_race(tmp_path):
    database_path = tmp_path / "startup-race.db"
    env = os.environ.copy()
    env["DB_URL"] = f"sqlite:///{database_path.as_posix()}"
    commands = [
        subprocess.Popen(
            [sys.executable, "-m", "scripts.init_db"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    results = [process.communicate(timeout=30) + (process.returncode,) for process in commands]
    assert all(returncode == 0 for _, _, returncode in results), results
    with sqlite3.connect(database_path) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert {"backlog_projects", "audit_log"}.issubset(tables)


def test_application_import_has_no_schema_initialization():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "init_db"
    ]
    imports = [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        and node.module == "modules.database" for alias in node.names
    ]
    assert not calls
    assert "init_db" not in imports


def test_railway_starts_initializer_before_gunicorn():
    procfile = (ROOT / "Procfile").read_text(encoding="utf-8")
    railway = (ROOT / "railway.toml").read_text(encoding="utf-8")
    expected = "python -m scripts.init_db && gunicorn app:app"
    assert expected in procfile
    assert expected in railway
    assert "preDeployCommand" not in railway


def test_project_loading_still_works_after_idempotent_initialization(seeded_db, today):
    init_db()
    count = process_backlog(pd.DataFrame([{
        "Project Definition": "STARTUP-TEST-001",
        "Project Name (English)": "Startup Test Project",
        "Status": "Ongoing",
    }]))
    rows = fetch_enriched_projects(today=today)
    assert count == 1
    assert any(row["project_code"] == "STARTUP-TEST-001" for row in rows)
