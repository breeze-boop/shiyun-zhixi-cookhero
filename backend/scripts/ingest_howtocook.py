from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.main import build_application_state
from scripts.howtocook_loader import HowToCookLoader


def _validate_howtocook_source(source: Path) -> Path:
    resolved = source.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"HowToCook dishes directory not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"HowToCook dishes path is not a directory: {resolved}")
    if "sample_recipes" in resolved.parts:
        raise ValueError(
            "HowToCook ingest source must come from the official HowToCook checkout, "
            "not data/sample_recipes"
        )
    return resolved


async def ingest(source: Path) -> None:
    source = _validate_howtocook_source(source)
    settings = get_settings()
    state = build_application_state(settings)
    state.rag_service.initialize_storage()
    if state.rag_service.cache_manager and state.rag_service.cache_manager.vector_backend:
        state.rag_service.cache_manager.vector_backend.create_collection()
    documents = HowToCookLoader(source).load()
    await state.rag_service.index_parsed_documents(documents)
    print(f"Indexed {len(documents)} parent documents from {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HowToCook dishes into PostgreSQL and Milvus.")
    parser.add_argument(
        "--source",
        default="../data/HowToCook/dishes",
        help="Path to HowToCook dishes directory",
    )
    args = parser.parse_args()
    asyncio.run(ingest(Path(args.source)))


if __name__ == "__main__":
    main()
