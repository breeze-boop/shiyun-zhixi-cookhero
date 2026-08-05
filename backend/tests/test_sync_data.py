from pathlib import Path

import pytest

from scripts import sync_data


def test_sync_howtocook_rejects_clone_without_dishes_directory(tmp_path, monkeypatch) -> None:
    target = tmp_path / "HowToCook"

    def fake_run(command: list[str], check: bool) -> None:
        target.mkdir(parents=True)
        (target / ".git").mkdir()

    monkeypatch.setattr(sync_data.subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError, match="HowToCook dishes directory not found"):
        sync_data.sync_howtocook(target)
