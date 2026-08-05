from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


HOWTOCOOK_REPO = "https://github.com/Anduin2017/HowToCook.git"


def sync_howtocook(target: Path) -> Path:
    if (target / ".git").exists():
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", HOWTOCOOK_REPO, str(target)], check=True)
    dishes = target / "dishes"
    if not dishes.exists():
        raise FileNotFoundError(f"HowToCook dishes directory not found: {dishes}")
    if not dishes.is_dir():
        raise NotADirectoryError(f"HowToCook dishes path is not a directory: {dishes}")
    return dishes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync HowToCook dishes knowledge base.")
    parser.add_argument("--target", default="data/HowToCook", help="Local repository path")
    args = parser.parse_args()
    dishes = sync_howtocook(Path(args.target))
    print(f"Synced dishes directory: {dishes}")


if __name__ == "__main__":
    main()

