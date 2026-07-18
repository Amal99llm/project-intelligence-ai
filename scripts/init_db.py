"""Initialize the application database schema, then exit."""

from modules.schema_management import apply_schema_changes


if __name__ == "__main__":
    apply_schema_changes()
