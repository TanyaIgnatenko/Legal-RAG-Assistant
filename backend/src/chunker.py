"""Text chunking strategies for legal documents.

Every chunker returns a list of dicts shaped like::

    {
        "text":     str,   # == raw_text[start:end]
        "start":    int,   # absolute char offset into the ORIGINAL text
        "end":      int,
        "metadata": str,   # human-readable citation label
        "chapter":  str,   # hierarchical chunkers only
        "article":  str,
    }

``start``/``end`` are the contract that makes chunking strategies comparable:
ground-truth evidence is expressed as char spans in the raw text, so a chunk's
relevance can be decided by span overlap rather than by metadata equality.
"""

import re
from typing import Dict, List, Protocol, runtime_checkable

# Headings must occupy a whole line. `[ \t]` rather than `\s` so the pattern
# cannot creep across a line break under re.MULTILINE.
CHAPTER_RE = re.compile(r'^[ \t]*(CHAPTER[ \t]+[IVXLCDM]+)[ \t]*$', re.M | re.I)
ARTICLE_RE = re.compile(r'^[ \t]*(Article[ \t]+\d+)[ \t]*$', re.M | re.I)

# Whitespace trimmed off chunk edges; offsets are corrected by the trim width
# so that raw_text[start:end] == chunk["text"] exactly.
_WS = " \t\r\n\f\v"


@runtime_checkable
class Chunker(Protocol):
    """Interface every chunking strategy implements."""

    name: str

    def chunk(self, text: str) -> List[Dict]:
        ...


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) past surrounding whitespace."""
    while start < end and text[start] in _WS:
        start += 1
    while end > start and text[end - 1] in _WS:
        end -= 1
    return start, end


def _emit(text: str, start: int, end: int, **fields) -> Dict | None:
    """Build a chunk dict for [start, end), or None if it is all whitespace."""
    start, end = _trimmed_span(text, start, end)
    if start >= end:
        return None
    return {"text": text[start:end], "start": start, "end": end, **fields}


class HierarchicalChunker:
    """Split a legal document along its CHAPTER / Article structure.

    One chunk per article; a chapter with no articles becomes a single chunk.
    Text before the first chapter heading (in GDPR: the 173 recitals, ~45% of
    the document) is intentionally not emitted — it is preamble, not normative
    text. `orphan_ratio` in the chunk metrics reports that gap explicitly.
    """

    name = "hierarchical"

    def chunk(self, text: str) -> List[Dict]:
        chunks: List[Dict] = []

        chapters = list(CHAPTER_RE.finditer(text))
        articles = list(ARTICLE_RE.finditer(text))

        for i, chapter in enumerate(chapters):
            chapter_start = chapter.start()
            chapter_end = chapters[i + 1].start() if i + 1 < len(chapters) else len(text)
            chapter_name = chapter.group(1).strip()
            chapter_title = self._line_after(text, chapter.end())
            label = f"{chapter_name} ({chapter_title})" if chapter_title else chapter_name

            inner = [a for a in articles if chapter_start <= a.start() < chapter_end]

            if not inner:
                chunk = _emit(
                    text, chapter_start, chapter_end,
                    chapter=chapter_name,
                    chapter_title=chapter_title,
                    article="N/A",
                    metadata=label,
                )
                if chunk:
                    chunks.append(chunk)
                continue

            for j, article in enumerate(inner):
                article_end = inner[j + 1].start() if j + 1 < len(inner) else chapter_end
                article_name = article.group(1).strip()
                chunk = _emit(
                    text, article.start(), article_end,
                    chapter=chapter_name,
                    chapter_title=chapter_title,
                    article=article_name,
                    metadata=f"{chapter_name} - {article_name}",
                )
                if chunk:
                    chunks.append(chunk)

        if chunks:
            return chunks

        # No chapter headings at all — fall back to paragraphs.
        offset = 0
        for i, para in enumerate(text.split("\n\n")):
            chunk = _emit(
                text, offset, offset + len(para),
                chapter="N/A",
                chapter_title="",
                article=f"Section {i + 1}",
                metadata=f"Section {i + 1}",
            )
            if chunk:
                chunks.append(chunk)
            offset += len(para) + 2
        return chunks

    @staticmethod
    def _line_after(text: str, pos: int) -> str:
        """The next non-empty line after `pos` — a chapter heading's title."""
        nl = text.find("\n", pos)
        if nl == -1:
            return ""
        end = text.find("\n", nl + 1)
        return text[nl + 1: end if end != -1 else len(text)].strip()


class RecursiveOverlapChunker:
    """Fixed-size chunks with overlap, split on progressively finer separators.

    Tries to break on paragraph, then line, then sentence, then word boundaries,
    falling back to a hard cut when no separator fits. Carries `overlap`
    characters of tail context into each following chunk.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.name = f"recursive-{chunk_size}-{overlap}"

    def chunk(self, text: str) -> List[Dict]:
        atoms = self._split(text, 0, len(text), self.SEPARATORS)
        spans = self._merge(text, atoms)

        chunks: List[Dict] = []
        for i, (start, end) in enumerate(spans):
            chunk = _emit(
                text, start, end,
                metadata=f"chunk {i}",
            )
            if chunk:
                chunks.append(chunk)
        # Re-label after dropping whitespace-only spans so indices stay dense.
        for i, chunk in enumerate(chunks):
            chunk["metadata"] = f"chunk {i}"
        return chunks

    def _split(self, text: str, start: int, end: int, separators: List[str]) -> List[tuple[int, int]]:
        """Break [start, end) into pieces no longer than chunk_size where possible."""
        if end - start <= self.chunk_size:
            return [(start, end)]

        for depth, sep in enumerate(separators):
            if text.find(sep, start, end) == -1:
                continue
            pieces: List[tuple[int, int]] = []
            cursor = start
            while cursor < end:
                hit = text.find(sep, cursor, end)
                if hit == -1:
                    piece_end = end
                else:
                    # Keep the separator with the piece it terminates.
                    piece_end = min(hit + len(sep), end)
                if piece_end > cursor:
                    pieces.extend(self._split(text, cursor, piece_end, separators[depth + 1:]))
                cursor = piece_end if piece_end > cursor else cursor + 1
            return pieces

        # No separator left: hard cut.
        return [(s, min(s + self.chunk_size, end))
                for s in range(start, end, self.chunk_size)]

    def _merge(self, text: str, atoms: List[tuple[int, int]]) -> List[tuple[int, int]]:
        """Greedily pack atoms into <= chunk_size windows, overlapping the tails."""
        spans: List[tuple[int, int]] = []
        window: List[tuple[int, int]] = []

        def flush() -> None:
            if window:
                spans.append((window[0][0], window[-1][1]))

        for atom in atoms:
            if window and atom[1] - window[0][0] > self.chunk_size:
                flush()
                window = self._tail(window)
                # An oversized atom cannot share a window with anything.
                if window and atom[1] - window[0][0] > self.chunk_size:
                    window = []
            window.append(atom)
        flush()

        # Drop a trailing window fully contained in its predecessor (pure overlap).
        if len(spans) > 1 and spans[-1][0] >= spans[-2][0] and spans[-1][1] <= spans[-2][1]:
            spans.pop()
        return spans

    def _tail(self, window: List[tuple[int, int]]) -> List[tuple[int, int]]:
        """The trailing atoms of `window` that fit inside the overlap budget."""
        if not self.overlap:
            return []
        kept: List[tuple[int, int]] = []
        end = window[-1][1]
        for atom in reversed(window):
            if end - atom[0] > self.overlap:
                break
            kept.insert(0, atom)
        return kept


CHUNKERS = {
    "hierarchical": HierarchicalChunker,
    "recursive-512-128": lambda: RecursiveOverlapChunker(512, 128),
}


def get_chunker(name: str) -> Chunker:
    """Build a chunker by its registry name."""
    if name not in CHUNKERS:
        raise KeyError(f"unknown chunker {name!r}; known: {sorted(CHUNKERS)}")
    return CHUNKERS[name]()


class LegalChunker(HierarchicalChunker):
    """Backwards-compatible alias for the pre-refactor entry point."""

    @staticmethod
    def chunk_gdpr(text: str) -> List[Dict]:
        return HierarchicalChunker().chunk(text)
