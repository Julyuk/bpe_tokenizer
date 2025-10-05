#!/usr/bin/env python3
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request

WIKI_API_EN = "https://en.wikipedia.org/w/api.php"
WIKI_API_UK = "https://uk.wikipedia.org/w/api.php"

TRAIN_EN_TITLES = [
    "Software engineering",
    "Computer programming",
    "Database",
    "Operating system",
]
TRAIN_UK_TITLES = [
    "Інженерія програмного забезпечення",
    "Комп'ютерне програмування",
    "База даних",
    "Операційна система",
]
VAL_EN_TITLES = [
    "Information security",
    "Computer network",
]
VAL_UK_TITLES = [
    "Інформаційна безпека",
    "Комп'ютерна мережа",
]


def fetch_extracts(api_base: str, titles: list[str]) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": "|".join(titles),
        "redirects": 1,
    }
    url = f"{api_base}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "bpe-tokenizer-educational/1.0"})
    with urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    pages = data.get("query", {}).get("pages", {})
    texts: list[str] = []
    for page in pages.values():
        extract = page.get("extract")
        if extract:
            texts.append(extract)
    return "\n\n".join(texts)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def main():
    base = Path("data")
    train_en = fetch_extracts(WIKI_API_EN, TRAIN_EN_TITLES)
    train_uk = fetch_extracts(WIKI_API_UK, TRAIN_UK_TITLES)
    val_en = fetch_extracts(WIKI_API_EN, VAL_EN_TITLES)
    val_uk = fetch_extracts(WIKI_API_UK, VAL_UK_TITLES)

    write_text(base / "train_en.txt", train_en)
    write_text(base / "train_uk.txt", train_uk)
    write_text(base / "val_en.txt", val_en)
    write_text(base / "val_uk.txt", val_uk)

    print("Wrote:")
    for p in ["train_en.txt", "train_uk.txt", "val_en.txt", "val_uk.txt"]:
        fp = base / p
        print(f" - {fp} ({fp.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
