"""Publish the app as a Hugging Face Space (free CPU hosting).

    python -m scripts.deploy_space --space <hf-username>/guichet              # first time
    python -m scripts.deploy_space --space <hf-username>/guichet --code-only  # updates

--code-only uploads the app without touching the Space's secrets, so it needs
only HF_TOKEN. The GitHub Action in ops/deploy.yml uses it on every push to main.

Needs in .env (never committed):
    ANTHROPIC_API_KEY   stored as a Space *secret*, never uploaded as a file
    HF_TOKEN            a Hugging Face token with write access

Also creates a private dataset <hf-username>/guichet-logs where the app syncs
its anonymised usage logs, unless --no-logs is passed.

Only what the app needs is uploaded: app.py, the guichet package, the corpus,
the precomputed embeddings, and a requirements file. No .env, no logs, no
evaluation data.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIREMENTS = """fastapi>=0.110
uvicorn>=0.29
fastembed>=0.9.0
rank-bm25>=0.2.2
numpy>=2.0
httpx>=0.27
beautifulsoup4>=4.12
lxml>=5.0
anthropic>=0.40
huggingface_hub>=0.25
"""

SPACE_README = """---
title: Guichet
emoji: 🏛️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: French admin questions answered from official pages
---

Answers questions about French administration (titres de séjour, visas, student
jobs, CAF, health insurance, taxes) from service-public.gouv.fr, in English or
French, quoting the official page word for word. Built and evaluated in
[github.com/AdamosZ/Guichet](https://github.com/AdamosZ/Guichet).

Not legal advice. Corpus text © DILA, Licence Ouverte / Etalab 2.0.
"""

DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Hugging Face runs the container as user 1000: keep writable paths in /tmp.
ENV HOST=0.0.0.0 PORT=7860 GUICHET_LOG_DIR=/tmp/user_logs HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["python", "app.py"]
"""

PACKAGE_FILES = ["__init__.py", "chunk.py", "corpus.py", "generate.py", "index.py",
                 "langid.py", "logbook.py", "normalize.py", "service.py"]


def build_bundle(target: Path) -> None:
    (target / "guichet").mkdir(parents=True)
    for name in PACKAGE_FILES:
        shutil.copy(ROOT / "guichet" / name, target / "guichet" / name)
    for name in ("app.py", "server.py"):
        shutil.copy(ROOT / name, target / name)
    shutil.copytree(ROOT / "static", target / "static")
    (target / "Dockerfile").write_text(DOCKERFILE)
    (target / "data/corpus").mkdir(parents=True)
    for name in ("chunks.jsonl", "documents.jsonl"):
        shutil.copy(ROOT / "data/corpus" / name, target / "data/corpus" / name)

    # Precompute the chunk embeddings so the Space starts in seconds, not minutes.
    from guichet.chunk import load_chunks
    from guichet.index import DenseIndex

    cache = target / "data/cache"
    cache.mkdir(parents=True)
    index = DenseIndex(load_chunks(ROOT / "data/corpus/chunks.jsonl"), cache_dir=ROOT / "data/cache")
    shutil.copy(index._cache_path(), cache / index._cache_path().name)  # only the current corpus
    (target / "requirements.txt").write_text(REQUIREMENTS)
    (target / "README.md").write_text(SPACE_README)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="e.g. adamosz/guichet")
    parser.add_argument("--no-logs", action="store_true")
    parser.add_argument("--daily-cap", default="300")
    parser.add_argument("--code-only", action="store_true", help="upload the app, leave secrets as they are")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    from guichet.generate import _load_env

    _load_env()
    token = os.environ.get("HF_TOKEN")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not token or (not key and not args.code_only):
        raise SystemExit("Put HF_TOKEN and ANTHROPIC_API_KEY in .env first.")

    api = HfApi(token=token)
    owner = args.space.split("/")[0]
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    if not args.code_only:
        api.add_space_secret(args.space, "ANTHROPIC_API_KEY", key)
        api.add_space_variable(args.space, "DAILY_CAP", args.daily_cap)
    if not args.no_logs and not args.code_only:
        logs_repo = f"{owner}/guichet-logs"
        api.create_repo(logs_repo, repo_type="dataset", private=True, exist_ok=True)
        api.add_space_variable(args.space, "HF_LOG_REPO", logs_repo)
        api.add_space_secret(args.space, "HF_TOKEN", token)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        print("building bundle (embedding the corpus if not cached, a few minutes the first time)…")
        build_bundle(bundle)
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=bundle,
                          commit_message=os.environ.get("DEPLOY_MESSAGE", "Deploy Guichet"))
    print(f"\ndeployed: https://huggingface.co/spaces/{args.space}")
    print("first start takes a few minutes while it installs; refresh the page until it loads.")


if __name__ == "__main__":
    main()
