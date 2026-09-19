"""Step 3.4 — generation metrics.

* correctness, faithfulness — LLM judge (Gemini, temperature 0, JSON verdict
  {score: 0|1, reasoning}), sampled N times; the median is the verdict and
  the spread is logged.
* citation_accuracy — no LLM: cited "Article N" numbers are checked against
  the articles the context actually contains.
* abstention — no LLM: did the answer decline? Scored separately on
  unanswerable / out_of_scope questions and never averaged with the rest.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from ..cache import LLMCache
from .chunking import overlap

# Same model as the generator: the free tier caps gemini-3.5-flash at 20
# requests/day. Self-grading is a known bias, reported in the limitations.
JUDGE_MODEL = "gemini-3.5-flash-lite"
JUDGE_TEMPERATURE = 0.0
JUDGE_RUNS = 3

CITATION_RE = re.compile(r"\bArt(?:icle|\.)?\s*(\d{1,3})\b", re.I)

ABSTAIN_PATTERNS = [
    r"does not contain sufficient information",
    r"do(?:es)? not (?:contain|provide|include|specify|mention|address)",
    r"(?:no|not) (?:enough|sufficient) information",
    r"I can only answer questions about the provided legal documents",
    r"(?:cannot|can't|unable to) (?:answer|find|determine)",
    r"not (?:covered|addressed|specified) (?:in|by)",
    r"outside the scope",
]
ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.I)

CORRECTNESS_PROMPT = """You are grading an answer to a question about the EU GDPR.

Compare the CANDIDATE answer with the REFERENCE answer. Score 1 if the
candidate states the reference's key facts (numbers, conditions, actors)
without contradicting it; extra correct detail is fine, wording does not
matter. Score 0 if a key fact is missing, wrong, or contradicted, or if the
candidate declines to answer while the reference gives a substantive answer.
If the reference says the question cannot be answered or is out of scope,
score 1 only if the candidate likewise declines or says the document does
not cover it.

QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{answer}

Reply with JSON only: {{"score": 0 or 1, "reasoning": "<one or two sentences>"}}"""

FAITHFULNESS_PROMPT = """You are checking whether an answer is grounded in its context.

Score 1 if every factual claim in the ANSWER can be inferred from the CONTEXT.
Score 0 if any claim is absent from or contradicted by the CONTEXT. An answer
that only says the context is insufficient is grounded (score 1).

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only: {{"score": 0 or 1, "reasoning": "<one or two sentences>"}}"""


# ── non-LLM metrics ──────────────────────────────────────────────────

def cited_articles(answer: str) -> set[int]:
    return {int(n) for n in CITATION_RE.findall(answer or "")}


def context_articles(context_chunks: Sequence[dict],
                     article_spans: dict[str, tuple[int, int]]) -> set[int]:
    """Articles a context actually contains.

    Derived from offsets, not chunk metadata: recursive chunks carry no
    article labels, and the check must mean the same thing for both chunkers.
    Articles the context text itself cross-references also count, since
    citing them is still grounded.
    """
    found: set[int] = set()
    for chunk in context_chunks:
        span = (chunk["start"], chunk["end"])
        for name, art_span in article_spans.items():
            if overlap(span, art_span) > 0:
                found.add(int(name.split()[-1]))
        found |= cited_articles(chunk.get("text", ""))
    return found


def citation_accuracy(answer: str, context_chunks: Sequence[dict],
                      article_spans: dict[str, tuple[int, int]]) -> bool | None:
    """True if every article the answer cites is present in its context.

    None when the answer cites nothing — that is a separate failure
    (`has_citation`) and should not be folded into accuracy.
    """
    cited = cited_articles(answer)
    if not cited:
        return None
    return cited <= context_articles(context_chunks, article_spans)


def abstained(answer: str) -> bool:
    return bool(ABSTAIN_RE.search(answer or ""))


# ── LLM judge ────────────────────────────────────────────────────────

@dataclass
class Verdict:
    score: int
    reasoning: str
    samples: list[int] = field(default_factory=list)

    @property
    def spread(self) -> int:
        return max(self.samples) - min(self.samples) if self.samples else 0


def parse_verdict(raw: str) -> tuple[int, str]:
    """Pull {score, reasoning} out of a judge reply, tolerating code fences."""
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        raise ValueError(f"judge returned no JSON: {raw[:120]!r}")
    data = json.loads(match.group(0))
    score = int(data["score"])
    if score not in (0, 1):
        raise ValueError(f"judge score out of range: {score}")
    return score, str(data.get("reasoning", ""))


class Judge:
    """Gemini as a binary grader, cached per (prompt, sample index)."""

    def __init__(self, api_key: str, cache: LLMCache,
                 model: str = JUDGE_MODEL, runs: int = JUDGE_RUNS):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model = model
        self.runs = runs
        self.cache = cache
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=JUDGE_TEMPERATURE,
            response_mime_type="application/json",
            max_output_tokens=1024,
            max_retries=0,
        )

    def _sample(self, prompt: str, run: int) -> tuple[int, str]:
        from langchain_core.messages import HumanMessage
        from src.rag_system import response_text

        from ..llm import with_backoff

        key = self.cache.key(self.model, prompt, JUDGE_TEMPERATURE, salt=f"run{run}")
        raw = self.cache.get(key)
        if raw is None:
            raw = response_text(with_backoff(
                lambda: self.llm.invoke([HumanMessage(content=prompt)])).content)
            parse_verdict(raw)             # only cache replies that parse
            self.cache.set(key, raw)
        return parse_verdict(raw)

    def _grade(self, prompt: str) -> Verdict:
        samples = [self._sample(prompt, run) for run in range(self.runs)]
        scores = [s for s, _ in samples]
        median = int(statistics.median(scores))
        reasoning = next(r for s, r in samples if s == median)
        return Verdict(median, reasoning, scores)

    def correctness(self, question: str, answer: str, reference: str) -> Verdict:
        return self._grade(CORRECTNESS_PROMPT.format(
            question=question, reference=reference, answer=answer))

    def faithfulness(self, answer: str, context_chunks: Sequence[dict]) -> Verdict:
        context = "\n---\n".join(c.get("text", "") for c in context_chunks) or "(empty)"
        return self._grade(FAITHFULNESS_PROMPT.format(context=context, answer=answer))
