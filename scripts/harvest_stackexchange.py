"""Collect candidate questions from Stack Exchange (CC BY-SA) for the golden set.

Only titles and question bodies are collected, never answers. Kept for
provenance; the selected, paraphrased items live in evals/golden.jsonl.

    python -m scripts.harvest_stackexchange > candidates_se.json

The unauthenticated API allows ~300 requests a day per IP.
"""

from __future__ import annotations

import gzip
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.stackexchange.com/2.3/search/advanced?"

QUERIES = [
    ("expatriates", {"tagged": "france"}),
    ("travel", {"tagged": "france;schengen"}),
    ("travel", {"tagged": "france;visas"}),
    ("travel", {"tagged": "france;residence-permits"}),
    ("travel", {"tagged": "france;students"}),
    ("expatriates", {"q": "titre de sejour"}),
    ("expatriates", {"q": "CAF"}),
    ("expatriates", {"q": "prefecture"}),
]


def get(params: dict) -> dict:
    with urllib.request.urlopen(API + urllib.parse.urlencode(params), timeout=30) as response:
        data = response.read()
    try:
        data = gzip.decompress(data)
    except OSError:
        pass
    return json.loads(data)


def harvest(pages: int = 3) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for site, query in QUERIES:
        for page in range(1, pages + 1):
            params = {"order": "desc", "sort": "votes", "site": site, "pagesize": 100,
                      "page": page, "filter": "withbody", **query}
            try:
                result = get(params)
            except Exception as exc:  # quota or throttling: keep what we have
                print(f"skip {site} {query}: {exc}", file=sys.stderr)
                break
            for item in result.get("items", []):
                if item["link"] in seen:
                    continue
                seen.add(item["link"])
                body = re.sub(r"<[^>]+>", " ", html.unescape(item.get("body", "")))
                out.append({
                    "source": f"stackexchange/{site}",
                    "url": item["link"],
                    "title": html.unescape(item["title"]),
                    "body": re.sub(r"\s+", " ", body)[:700],
                    "created": time.strftime("%Y-%m-%d", time.gmtime(item["creation_date"])),
                    "license": item.get("content_license"),
                })
            if not result.get("has_more"):
                break
            time.sleep(0.5)
    return out


if __name__ == "__main__":
    json.dump(harvest(), sys.stdout, ensure_ascii=False, indent=1)
