"""One-off diagnostic: what does PDFParser actually produce for gdpr.pdf?

Run:
    python backend/eval/smoke/inspect_parse.py [path/to.pdf]

Answers one question: are CHAPTER/Article headings on their own lines, or did
PyMuPDF glue columns / running headers / page numbers into them?
"""

import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.parser import PDFParser  # noqa: E402

DEFAULT_PDF = BACKEND / "example_data" / "gdpr.pdf"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def main(pdf_path: Path) -> None:
    text = PDFParser.parse(str(pdf_path))

    print("=" * 70)
    print(f"FILE: {pdf_path}")
    print("=" * 70)
    print(f"len(text) = {len(text):,}")

    print("\n--- first 3000 chars, repr() ---")
    print(repr(text[:3000]))

    lines = text.split("\n")
    lengths = sorted(len(line) for line in lines)
    print(f"\n--- line length distribution (n_lines={len(lines):,}) ---")
    print(f"p50 = {percentile(lengths, 0.50)}")
    print(f"p95 = {percentile(lengths, 0.95)}")
    print(f"max = {lengths[-1] if lengths else 0}")

    probes = [
        (r"Article \d+", r"Article\s+\d+", r"^\s*Article\s+\d+\s*$"),
        (r"CHAPTER [IVXL]+", r"CHAPTER\s+[IVXLCDM]+", r"^\s*CHAPTER\s+[IVXLCDM]+\s*$"),
    ]
    for label, anywhere, own_line in probes:
        n_any = len(re.findall(anywhere, text))
        n_line = len(re.findall(own_line, text, re.M))
        n_line_ci = len(re.findall(own_line, text, re.M | re.I))
        print(f"\n--- {label} ---")
        print(f"anywhere in line        : {n_any}")
        print(f"as a whole line         : {n_line}")
        print(f"as a whole line (icase) : {n_line_ci}")

    # What the CURRENT chunker regexes see, for comparison.
    print("\n--- current LegalChunker patterns ---")
    cur_chapter = r"(?:CHAPTER)\s+(?:[IVXLCDM]+|\d+)(?:\s*[-:.]?\s*[^\n]*)?"
    cur_article = r"(?:^|\n)\s*(?:Article)\s+\d+\s\n"
    print(f"chapter_pattern matches : {len(re.findall(cur_chapter, text))}")
    print(f"article_pattern matches : {len(re.findall(cur_article, text, re.I))}")

    # Sample the lines that contain a heading but are not exactly a heading.
    print("\n--- sample lines containing 'Article N' that are NOT a bare heading ---")
    shown = 0
    for line in lines:
        if re.search(r"Article\s+\d+", line) and not re.fullmatch(r"\s*Article\s+\d+\s*", line):
            print(repr(line[:160]))
            shown += 1
            if shown >= 10:
                break

    print("\n--- sample lines containing 'CHAPTER' ---")
    for line in lines:
        if re.search(r"CHAPTER", line, re.I):
            print(repr(line[:160]))

    print("\n--- 10 longest lines (truncated) ---")
    for line in sorted(lines, key=len, reverse=True)[:10]:
        print(f"[{len(line):>4}] {repr(line[:200])}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    main(target)
