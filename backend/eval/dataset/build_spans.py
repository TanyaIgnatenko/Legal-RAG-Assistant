"""Step 2.2 — resolve article numbers to char spans in the raw text.

Ground truth has to live in the same coordinate space for every chunking
strategy, otherwise the ablation compares labels rather than retrieval. So the
dataset stores `evidence_spans` as absolute char offsets into the parsed text,
resolved here once and committed.

Usage:
    python -m eval.dataset.build_spans                 # gdpr_qa.json -> gdpr_qa_resolved.json
    python -m eval.dataset.build_spans --dump          # list every article span
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.chunker import ARTICLE_RE, CHAPTER_RE, _trimmed_span  # noqa: E402
from src.parser import PDFParser  # noqa: E402

HERE = Path(__file__).resolve().parent
GDPR_PDF = BACKEND / "example_data" / "gdpr.pdf"
SOURCE = HERE / "gdpr_qa.json"
RESOLVED = HERE / "gdpr_qa_resolved.json"


def normalise(name: str) -> str:
    """'Article  83 ' -> 'Article 83' so lookups are not whitespace-sensitive."""
    return re.sub(r"\s+", " ", name).strip().title()


def article_spans(text: str) -> dict[str, tuple[int, int]]:
    """Map every article heading to [heading start, next heading start).

    An article ends where the next article *or* the next chapter begins,
    whichever comes first — otherwise the last article of a chapter would
    swallow the following chapter's title.

    Spans are trimmed of surrounding whitespace exactly as chunk spans are, so
    that an article span can sit inside the chunk covering it. Without this, a
    trailing newline makes containment fail and coverage reads as 0.
    """
    articles = list(ARTICLE_RE.finditer(text))
    boundaries = sorted(m.start() for m in articles + list(CHAPTER_RE.finditer(text)))

    spans: dict[str, tuple[int, int]] = {}
    duplicates: list[str] = []
    for match in articles:
        start = match.start()
        end = next((b for b in boundaries if b > start), len(text))
        name = normalise(match.group(1))
        if name in spans:
            duplicates.append(name)
        spans[name] = _trimmed_span(text, start, end)

    if duplicates:
        raise ValueError(f"article headings are not unique: {sorted(set(duplicates))}")
    return spans


def article_title(text: str, span: tuple[int, int]) -> str:
    """The line following the heading — the article's own title."""
    body = text[span[0]:span[1]].split("\n")
    return body[1].strip() if len(body) > 1 else ""


def flatten(s: str) -> str:
    """Collapse whitespace so a quote survives the PDF's line wrapping."""
    return re.sub(r"\s+", " ", s).strip()


def locate_quote(text: str, span: tuple[int, int], quote: str) -> tuple[int, int]:
    """Find a proof quote inside an article and return its raw char span.

    The quote is written on one line but the PDF wraps it across several, so
    every whitespace run is matched loosely.
    """
    pattern = r"\s+".join(re.escape(word) for word in flatten(quote).split(" "))
    match = re.search(pattern, text[span[0]:span[1]])
    if not match:
        raise ValueError(f"quote not locatable: {quote[:70]!r}")
    return span[0] + match.start(), span[0] + match.end()


def verify_proofs(item: dict, required: list[str], text: str,
                  spans: dict[str, tuple[int, int]]) -> None:
    """Each `proof` quote must occur verbatim inside its required article.

    This is what stops a plausible-but-invented citation from reaching a run:
    the dataset does not build unless every quote is really in the PDF, at the
    article it is attributed to.
    """
    proofs = item.get("proof", [])
    if not required:
        if proofs:
            raise ValueError(f"{item['id']}: proof given but no required_articles")
        return
    if len(proofs) != len(required):
        raise ValueError(
            f"{item['id']}: {len(proofs)} proof quotes for {len(required)} articles"
        )
    for article, quote in zip(required, proofs):
        start, end = spans[article]
        if flatten(quote) not in flatten(text[start:end]):
            raise ValueError(
                f"{item['id']}: quote not found in {article}: {quote[:70]!r}"
            )


def resolve(dataset: list[dict], text: str) -> list[dict]:
    """Attach evidence_spans to each question, in document order."""
    spans = article_spans(text)
    out = []
    for item in dataset:
        required = [normalise(a) for a in item.get("required_articles", [])]
        missing = [a for a in required if a not in spans]
        if missing:
            raise KeyError(f"{item['id']}: no such article heading: {missing}")
        verify_proofs(item, required, text, spans)

        item = dict(item)
        item["required_articles"] = required
        item["acceptable_articles"] = [normalise(a) for a in item.get("acceptable_articles", [])]
        ordered = sorted(required, key=lambda a: spans[a])
        item["evidence_spans"] = [list(spans[a]) for a in ordered]

        # The answering sentence itself, not the whole article. An article can
        # run to 8913 chars, so no 512-char chunk could ever contain one whole
        # and coverage would read as 0 for every fixed-size chunker.
        by_article = dict(zip(required, item.get("proof", [])))
        item["quote_spans"] = [
            list(locate_quote(text, spans[a], by_article[a])) for a in ordered
        ]
        out.append(item)
    return out


def dataset_hash(dataset: list[dict]) -> str:
    payload = json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=GDPR_PDF)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=RESOLVED)
    ap.add_argument("--dump", action="store_true", help="list article spans and exit")
    args = ap.parse_args()

    text = PDFParser.parse(str(args.pdf))

    if args.dump:
        spans = article_spans(text)
        print(f"{len(spans)} articles in {args.pdf.name}\n")
        for name, (start, end) in sorted(spans.items(), key=lambda kv: kv[1]):
            print(f"{name:<12} [{start:>6},{end:>6})  {end - start:>5} chars  "
                  f"{article_title(text, (start, end))}")
        return 0

    if not args.source.exists():
        print(f"missing dataset: {args.source}")
        return 1

    dataset = json.loads(args.source.read_text(encoding="utf-8"))
    resolved = resolve(dataset, text)

    args.out.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    spans_total = sum(len(i["evidence_spans"]) for i in resolved)
    print(f"resolved {len(resolved)} questions, {spans_total} evidence spans -> {args.out.name}")
    print(f"dataset sha256[:16] = {dataset_hash(resolved)}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
