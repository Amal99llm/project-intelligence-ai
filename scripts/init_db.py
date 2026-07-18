"""Initialize the application database schema, then exit."""

from modules.database import init_db


if __name__ == "__main__":
    init_db()
