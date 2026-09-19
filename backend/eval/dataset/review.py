"""Print the dataset for human review: question, article, verbatim proof quote.

Also reports lexical overlap between each question and its gold article, since
a question that reuses the article's own vocabulary tests string matching more
than it tests retrieval.

Usage:
    python -m eval.dataset.review
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from eval.dataset.build_spans import GDPR_PDF, RESOLVED, article_spans, article_title  # noqa: E402
from src.parser import PDFParser  # noqa: E402

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "be", "been", "was", "were", "as", "at", "that", "this", "it",
    "its", "from", "which", "who", "whom", "what", "when", "how", "shall",
    "may", "must", "can", "do", "does", "did", "has", "have", "had", "not",
    "any", "all", "there", "their", "they", "them", "he", "she", "his", "her",
    "him", "one", "if", "so", "such", "than", "then", "up", "out", "into",
    "about", "over", "under", "only", "other", "more", "most", "no", "nor",
    "but", "before", "after", "own", "same", "s", "t", "will", "would", "you",
    "i", "we", "us", "our", "my", "me",
}


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def main() -> int:
    if not RESOLVED.exists():
        print(f"missing {RESOLVED}; run: python -m eval.dataset.build_spans")
        return 1

    raw = PDFParser.parse(str(GDPR_PDF))
    spans = article_spans(raw)
    dataset = json.loads(RESOLVED.read_text(encoding="utf-8"))

    overlaps = []
    for item in dataset:
        print("=" * 78)
        print(f"{item['id']}  [{item['qtype']}]  chapter {item['chapter']}")
        print(f"Q: {item['question']}")
        print(f"A: {item['reference_answer']}")

        if not item["required_articles"]:
            print("required: (none — nothing in the document should support an answer)")
            print()
            continue

        for article, quote, span in zip(
            item["required_articles"], item["proof"], item["evidence_spans"]
        ):
            body = raw[spans[article][0]:spans[article][1]]
            shared = tokens(item["question"]) & tokens(body)
            ratio = len(shared) / max(1, len(tokens(item["question"])))
            overlaps.append(ratio)

            print(f"\nrequired: {article} — {article_title(raw, spans[article])}")
            print(f"  span   : [{span[0]}, {span[1]})  {span[1] - span[0]} chars")
            print(f"  quote  : \"{re.sub(r'\\s+', ' ', quote).strip()}\"")
            print(f"  lexical overlap with question: {ratio:.0%}  shared={sorted(shared)}")
        if item["acceptable_articles"]:
            print(f"\nacceptable (not required, not penalised): {item['acceptable_articles']}")
        print()

    print("=" * 78)
    by_type = Counter(i["qtype"] for i in dataset)
    by_chapter = Counter(i["chapter"] for i in dataset if i["chapter"] != "N/A")
    print(f"total questions : {len(dataset)}")
    print(f"by qtype        : {dict(by_type)}")
    print(f"chapters covered: {len(by_chapter)}/11  {dict(sorted(by_chapter.items()))}")
    print(f"mean lexical overlap question<->gold article: {sum(overlaps)/len(overlaps):.0%}")
    print(f"max  lexical overlap: {max(overlaps):.0%}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
