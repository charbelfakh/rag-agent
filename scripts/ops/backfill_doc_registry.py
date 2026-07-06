"""Backfill SQLite document registry from Qdrant. Run: ``python -m scripts.ops.backfill_doc_registry``."""
import argparse

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from providers.doc_registry import get_doc_registry, reset_doc_registry
from providers.factory import get_vector_store, reset_providers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print source count from Qdrant without writing registry",
    )
    args = parser.parse_args()

    reset_providers()
    reset_doc_registry()
    store = get_vector_store()
    sources = store.list_sources()
    print(f"Found {len(sources)} sources in Qdrant")

    if args.dry_run:
        for row in sources[:10]:
            print(f"  {row.get('vendor', '')}/{row.get('source', '')}")
        if len(sources) > 10:
            print(f"  ... and {len(sources) - 10} more")
        return 0

    registry = get_doc_registry()
    count = registry.backfill_from_vector_store(store)
    print(f"Backfilled {count} documents into {registry.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
