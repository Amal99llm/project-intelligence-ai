# Schema management

`modules.schema_management.apply_schema_changes()` is the application's only
public production entry point for schema DDL. The deployment initializer calls
it before Gunicorn starts; web workers never initialize or migrate the schema.

## Why the lock exists

SQLite checks whether a table exists separately from issuing `CREATE TABLE`.
When multiple Gunicorn workers previously imported the application together,
two processes could pass that check and race to create the same table. The
cross-process lock in `modules/schema_management.py` serializes the complete
schema section and is released in a `finally` block. Database errors are not
suppressed.

## Contributor rule

Every future schema mutation must be placed inside the locked block in
`apply_schema_changes()`, including:

- SQLAlchemy metadata creation;
- table or index creation;
- ordered migrations;
- raw DDL, if it ever becomes necessary.

Do not import or call `_schema_initialization_lock()` from another module, and
do not execute `create_all`, `drop_all`, `CREATE`, `ALTER`, or `DROP` statements
in production code elsewhere. `tests/test_schema_architecture.py` enforces this
boundary and will fail when an unprotected DDL path is introduced.

The Railway start command remains:

```text
python -m scripts.init_db && gunicorn app:app ...
```

For the current file-backed SQLite deployment this must remain a start-command
step, not a Railway pre-deploy command: Railway pre-deploy containers do not
mount persistent volumes and their filesystem changes are not retained.
