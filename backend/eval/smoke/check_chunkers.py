"""Sanity-check both chunkers against the real GDPR text."""

import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.chunker import HierarchicalChunker, RecursiveOverlapChunker  # noqa: E402
from src.parser import PDFParser  # noqa: E402


def report(name, chunks, raw):
    lengths = sorted(len(c["text"]) for c in chunks)
    covered = sum(c["end"] - c["start"] for c in chunks)
    print(f"\n=== {name} ===")
    print(f"n_chunks        = {len(chunks)}")
    print(f"unique_articles = {len({c.get('article') for c in chunks} - {None, 'N/A'})}")
    print(f"len p50/p95/max = {lengths[len(lengths)//2]}/{lengths[int(.95*len(lengths))]}/{lengths[-1]}")
    print(f"covered chars   = {covered:,} / {len(raw):,}")

    bad = [c for c in chunks if raw[c["start"]:c["end"]] != c["text"]]
    print(f"offset mismatches = {len(bad)}")
    assert not bad, bad[:1]

    overlapping = sum(1 for a, b in zip(chunks, chunks[1:]) if b["start"] < a["end"])
    print(f"adjacent overlaps = {overlapping}")
    return Counter(c.get("chapter") for c in chunks)


raw = PDFParser.parse(str(BACKEND / "example_data" / "gdpr.pdf"))
h = HierarchicalChunker().chunk(raw)
r = RecursiveOverlapChunker(512, 128).chunk(raw)

by_chapter = report("hierarchical", h, raw)
report("recursive-512-128", r, raw)

print("\nchapters:", dict(by_chapter))
print("\nfirst hierarchical metadata:", [c["metadata"] for c in h[:3]])
print("first recursive metadata   :", [c["metadata"] for c in r[:3]])
print("\nsample recursive chunk:")
print(repr(r[40]["text"][:300]))
