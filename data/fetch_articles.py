"""Fetch articles from arbitrary URLs and save as a JSON dataset."""

import argparse
import hashlib
import json
import sys

import trafilatura


def fetch_article(url: str) -> dict | None:
    """Download a URL and extract its main article content."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not text:
        return None

    metadata = trafilatura.extract(
        downloaded,
        include_comments=False,
        output_format="json",
    )
    meta = json.loads(metadata) if metadata else {}

    return {
        "id": hashlib.sha256(url.encode()).hexdigest()[:16],
        "url": url,
        "title": meta.get("title", ""),
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch articles from URLs into a JSON dataset.")
    parser.add_argument("-o", "--output", default="data/thetech/data.json", help="Output JSON path")
    args = parser.parse_args()

    urls = [
        "https://thetech.com/2025/10/30/meng-enrollment-changes",
        "https://thetech.com/2025/10/30/6-1200-ase",
        "https://thetech.com/2025/11/06/2025-urop-mixer",
        "https://thetech.com/2025/11/06/pumpkin-drop-2025",
        "https://thetech.com/2025/11/06/2025-cambridge-city-council-election",
        "https://thetech.com/2025/11/14/co-2029-demographics",
        "https://thetech.com/2025/11/21/2025-mithic-event",
        "https://thetech.com/2025/11/21/2025-rhodes",
        "https://thetech.com/2025/12/04/mit-libraries-closure-2025",
        "https://thetech.com/2025/12/11/commencement-speaker-26"
    ]

    articles = []
    for url in urls:
        print(f"Fetching: {url} ... ", end="", flush=True)
        article = fetch_article(url)
        if article and article["text"]:
            articles.append(article)
            print(f"OK ({len(article['text'])} chars)")
        else:
            print("FAILED or empty")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)

    print(f"\nSaved {len(articles)} articles to {args.output}")


if __name__ == "__main__":
    main()
