"""Fail loudly if an API key ever ends up in a tracked file."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")


def test_no_api_key_in_tracked_files():
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    leaks = []
    for name in files:
        path = ROOT / name
        if path.is_file() and path.stat().st_size < 20_000_000:
            if KEY.search(path.read_text(encoding="utf-8", errors="ignore")):
                leaks.append(name)
    assert not leaks, f"API key pattern found in: {leaks}"


def test_env_is_gitignored():
    assert ".env" in (ROOT / ".gitignore").read_text().split()
