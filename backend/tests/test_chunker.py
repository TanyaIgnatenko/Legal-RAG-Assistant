"""Chunker invariants — above all, that offsets point where they claim to."""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from src.chunker import (  # noqa: E402
    ARTICLE_RE,
    CHAPTER_RE,
    HierarchicalChunker,
    RecursiveOverlapChunker,
    get_chunker,
)
from src.parser import PDFParser  # noqa: E402

GDPR_PDF = BACKEND / "example_data" / "gdpr.pdf"


@pytest.fixture(scope="module")
def gdpr_text() -> str:
    if not GDPR_PDF.exists():
        pytest.skip(f"missing {GDPR_PDF}")
    return PDFParser.parse(str(GDPR_PDF))


@pytest.fixture(params=["hierarchical", "recursive-512-128"])
def chunker(request):
    return get_chunker(request.param)


# ── offsets ──────────────────────────────────────────────────────────

def test_offsets_match_raw_text(chunker, gdpr_text):
    """raw_text[start:end] must reproduce the chunk text exactly."""
    chunks = chunker.chunk(gdpr_text)
    assert chunks
    for c in chunks:
        assert gdpr_text[c["start"]:c["end"]] == c["text"], c["metadata"]


def test_offsets_are_sane(chunker, gdpr_text):
    chunks = chunker.chunk(gdpr_text)
    for c in chunks:
        assert 0 <= c["start"] < c["end"] <= len(gdpr_text)
    starts = [c["start"] for c in chunks]
    assert starts == sorted(starts), "chunks must be emitted in document order"


def test_every_chunk_has_the_contract_fields(chunker, gdpr_text):
    for c in chunker.chunk(gdpr_text):
        assert {"text", "start", "end", "metadata"} <= set(c)
        assert c["text"].strip() == c["text"], "chunk text must be pre-trimmed"


# ── heading regexes ──────────────────────────────────────────────────

def test_headings_must_own_their_line():
    """The old pattern matched 'referred to in Article 5' mid-sentence."""
    prose = "rights as referred to in Article 5 shall apply under Chapter VIII.\n"
    assert not ARTICLE_RE.search(prose)
    assert not CHAPTER_RE.search(prose)

    headings = "CHAPTER II \nPrinciples \nArticle 5 \nPrinciples relating to processing \n"
    assert CHAPTER_RE.search(headings).group(1).strip() == "CHAPTER II"
    assert ARTICLE_RE.search(headings).group(1).strip() == "Article 5"


def test_heading_regex_does_not_span_lines():
    assert not CHAPTER_RE.search("CHAPTER\nII\n")
    assert not ARTICLE_RE.search("Article\n5\n")


# ── hierarchical structure ───────────────────────────────────────────

def test_hierarchical_recovers_gdpr_structure(gdpr_text):
    chunks = HierarchicalChunker().chunk(gdpr_text)
    articles = {c["article"] for c in chunks} - {"N/A"}
    chapters = {c["chapter"] for c in chunks}
    assert 90 <= len(chunks) <= 110, len(chunks)
    assert len(articles) >= 90, len(articles)
    assert 10 <= len(chapters) <= 12, chapters


def test_hierarchical_chunks_do_not_overlap(gdpr_text):
    chunks = HierarchicalChunker().chunk(gdpr_text)
    for a, b in zip(chunks, chunks[1:]):
        assert b["start"] >= a["end"]


def test_hierarchical_metadata_is_a_single_line(gdpr_text):
    for c in HierarchicalChunker().chunk(gdpr_text):
        assert "\n" not in c["metadata"]


def test_hierarchical_falls_back_to_paragraphs():
    text = "first para line\n\nsecond para line\n\nthird para line"
    chunks = HierarchicalChunker().chunk(text)
    assert [c["text"] for c in chunks] == [
        "first para line", "second para line", "third para line",
    ]
    for c in chunks:
        assert text[c["start"]:c["end"]] == c["text"]


# ── recursive ────────────────────────────────────────────────────────

def test_recursive_respects_chunk_size(gdpr_text):
    chunker = RecursiveOverlapChunker(512, 128)
    for c in chunker.chunk(gdpr_text):
        assert len(c["text"]) <= chunker.chunk_size


def test_recursive_covers_the_whole_document(gdpr_text):
    """Unlike hierarchical, it has no preamble blind spot."""
    chunks = RecursiveOverlapChunker(512, 128).chunk(gdpr_text)
    assert chunks[0]["start"] < 200
    assert chunks[-1]["end"] > len(gdpr_text) - 200

    # No gaps between consecutive chunks (whitespace-only seams aside).
    for a, b in zip(chunks, chunks[1:]):
        assert not gdpr_text[a["end"]:b["start"]].strip()


def test_recursive_actually_overlaps(gdpr_text):
    chunks = RecursiveOverlapChunker(512, 128).chunk(gdpr_text)
    overlapping = sum(1 for a, b in zip(chunks, chunks[1:]) if b["start"] < a["end"])
    assert overlapping > 0.9 * (len(chunks) - 1)


def test_recursive_metadata_is_dense_and_ordered(gdpr_text):
    chunks = RecursiveOverlapChunker(512, 128).chunk(gdpr_text)
    assert [c["metadata"] for c in chunks] == [f"chunk {i}" for i in range(len(chunks))]


def test_recursive_rejects_overlap_larger_than_chunk():
    with pytest.raises(ValueError):
        RecursiveOverlapChunker(chunk_size=128, overlap=128)


def test_recursive_handles_text_with_no_separators():
    text = "x" * 1300
    chunks = RecursiveOverlapChunker(512, 128).chunk(text)
    assert chunks
    for c in chunks:
        assert text[c["start"]:c["end"]] == c["text"]
        assert len(c["text"]) <= 512


def test_recursive_name_encodes_its_parameters():
    assert RecursiveOverlapChunker(512, 128).name == "recursive-512-128"
    assert RecursiveOverlapChunker(256, 64).name == "recursive-256-64"
