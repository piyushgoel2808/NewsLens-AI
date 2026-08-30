#!/usr/bin/env python3
"""Schema Inspector for NewsLens-AI.

Usage:
    uv run python scripts/show_schema.py               # View all tables and schemas
    uv run python scripts/show_schema.py articles      # View schema for a specific table
    uv run python scripts/show_schema.py --list        # List table names only
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sqlalchemy import create_engine, inspect
from app.core.config import get_settings


def inspect_database(target_table: str | None = None, list_only: bool = False) -> None:
    settings = get_settings()
    engine = create_engine(settings.database.sync_url)
    inspector = inspect(engine)

    tables = sorted(inspector.get_table_names())

    if list_only:
        print(f"\nDatabase Tables in '{settings.database.db}' ({len(tables)} total):\n")
        for t in tables:
            print(f"  • {t}")
        print()
        return

    if target_table:
        if target_table not in tables:
            print(f"❌ Table '{target_table}' not found in database '{settings.database.db}'.")
            print(f"Available tables: {', '.join(tables)}")
            sys.exit(1)
        tables = [target_table]

    print(f"\n{'=' * 80}")
    print(f"NewsLens-AI MySQL Schema Inspector [{settings.database.db}]")
    print(f"{'=' * 80}\n")

    for table_name in tables:
        columns = inspector.get_columns(table_name)
        pk_info = inspector.get_pk_constraint(table_name)
        pks = set(pk_info.get("constrained_columns", []))
        fks = inspector.get_foreign_keys(table_name)
        fk_map = {
            col: f"{fk['referred_table']}.{fk['referred_columns'][0]}"
            for fk in fks
            for col in fk.get("constrained_columns", [])
        }
        indexes = inspector.get_indexes(table_name)

        print(f"Table: \033[1;36m{table_name}\033[0m ({len(columns)} columns)")
        print("-" * 80)
        print(f"  {'Column':<26} {'Type':<20} {'Nullable':<10} {'Key / Details'}")
        print("  " + "-" * 76)

        for col in columns:
            name = col["name"]
            col_type = str(col["type"])
            nullable = "YES" if col.get("nullable", True) else "NO"

            details = []
            if name in pks:
                details.append("PK")
            if name in fk_map:
                details.append(f"FK -> {fk_map[name]}")

            details_str = ", ".join(details) if details else ""
            print(f"  {name:<26} {col_type:<20} {nullable:<10} {details_str}")

        if indexes:
            idx_strs = []
            for idx in indexes:
                cols = ", ".join(idx["column_names"])
                unique = "UNIQUE " if idx.get("unique") else ""
                idx_strs.append(f"{unique}{idx['name']} ({cols})")
            print("  Indexes: " + "; ".join(idx_strs))

        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args or "-l" in args:
        inspect_database(list_only=True)
    elif args:
        inspect_database(target_table=args[0])
    else:
        inspect_database()
