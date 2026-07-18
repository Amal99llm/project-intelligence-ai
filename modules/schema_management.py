"""The application's single production entry point for schema DDL.

SQLite's table-existence check and subsequent CREATE are not one atomic
operation. Concurrent WSGI startup processes previously passed the check
together and then raced to create the same table. Every present and future
schema mutation—metadata creation, indexes, and migrations—must be added to
``apply_schema_changes`` so it executes while the same cross-process lock is
held. Production modules must never call the private lock or execute DDL
directly.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading

from modules.database import Base, engine


_SCHEMA_THREAD_LOCK = threading.Lock()


@contextmanager
def _schema_initialization_lock():
    """Serialize the complete SQLite schema section; never suppress errors."""
    if engine.url.get_backend_name() != "sqlite" or not engine.url.database:
        yield
        return

    database_path = Path(engine.url.database).resolve()
    lock_path = database_path.with_name(f"{database_path.name}.schema.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _SCHEMA_THREAD_LOCK, lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def apply_schema_changes() -> None:
    """Apply every application schema mutation under one exclusive lock.

    Future table creation, index creation, or migration calls belong inside
    this locked block. Do not expose or call the private lock elsewhere.
    """
    with _schema_initialization_lock():
        Base.metadata.create_all(engine)
        # Add all future migrations and other DDL here, in execution order.
