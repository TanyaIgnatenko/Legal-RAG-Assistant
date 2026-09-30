"""Step 4 — evaluation runner.

    python -m eval.runner --chunker hierarchical --top-k 3 --fast
    python -m eval.runner --chunker recursive-512-128 --top-k 3 --full

--fast  parse / chunk / retrieval metrics only. No LLM, seconds.
--full  plus oracle / retrieved / no-context generation and the LLM judge.

Writes eval/results/{config}.jsonl (one Trace per line) and
eval/results/{config}.meta.json (models, dataset hash, chunker code hash,
date) so a results table can still be interpreted a month later.

Ground truth is `quote_spans` — the answering sentence — rather than whole
articles: an article can run to 8913 chars, so no 512-char chunk could ever
contain one and the fixed-size chunker would have no comparable questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from eval.cache import EmbeddingCache, LLMCache  # noqa: E402
from eval.cost import TokenCounter, price  # noqa: E402
from eval.dataset.build_spans import GDPR_PDF, RESOLVED, article_spans, dataset_hash  # noqa: E402
from eval.metrics.chunking import (  # noqa: E402
    chunks_overlapping,
    coverage,
    fragmentation,
    structural_metrics,
)
from eval.metrics.retrieval import DIAGNOSTIC_DEPTH, rank_of_gold, retrieval_metrics  # noqa: E402
from eval.smoke.parse import parse_smoke  # noqa: E402
from src.chunker import get_chunker  # noqa: E402
from src.parser import PDFParser  # noqa: E402
from src.rag_system import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, RAGDemo  # noqa: E402

RESULTS = BACKEND / "eval" / "results"
GROUND_TRUTH = "quote_spans"
UNSUFFIXED_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class EvalConfig:
    chunker: str                      # "hierarchical" | "recursive-512-128"
    top_k: int
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @property
    def name(self) -> str:
        # Result filenames predate the switch to bge, and the chunking ablation
        # was run before the embedding model was a dimension at all. The old
        # encoder therefore keeps the bare name so committed traces stay
        # addressable; every other model carries a suffix.
        if self.embedding_model == UNSUFFIXED_EMBEDDING_MODEL:
            return f"{self.chunker}-k{self.top_k}"
        return f"{self.chunker}-k{self.top_k}-{self.embedding_model.split('/')[-1]}"


@dataclass
class Trace:
    qid: str
    qtype: str
    chapter: str
    coverage: float
    fragmentation: float
    hit_at_k: bool | None
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    rank_of_gold: int | None
    retrieved_metadata: list[str]
    n_context_chars: int
    a_oracle: str | None = None
    a_retrieved: str | None = None
    a_noctx: str | None = None
    correct_oracle: bool | None = None
    correct_retrieved: bool | None = None
    correct_noctx: bool | None = None
    faithful: bool | None = None
    citation_ok: bool | None = None
    has_citation: bool | None = None
    abstained: bool | None = None
    tokens: dict = field(default_factory=dict)
    cost_usd: dict = field(default_factory=dict)
    judge_spread: dict = field(default_factory=dict)
    judge_reasoning: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)


class Workspace:
    """Everything shared across configs: text, dataset, one index per chunker."""

    def __init__(self, api_key: str = "", embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.raw = PDFParser.parse(str(GDPR_PDF))
        self.dataset = json.loads(RESOLVED.read_text(encoding="utf-8"))
        self.article_spans = article_spans(self.raw)
        self.llm_cache = LLMCache()
        self.emb_cache = EmbeddingCache()
        self.tokens = TokenCounter(api_key)
        self.api_key = api_key
        self.embedding_model = embedding_model
        self._systems: dict[tuple[str, str], tuple[RAGDemo, list[dict]]] = {}
        self._judge = None

    def system(self, chunker_name: str,
               embedding_model: str | None = None) -> tuple[RAGDemo, list[dict]]:
        """A RAGDemo indexed with this chunker and embedding model.

        Indexes are cached per (chunker, embedding model) so an ablation that
        sweeps top_k builds each one once.
        """
        embedding_model = embedding_model or self.embedding_model
        key = (chunker_name, embedding_model)
        if key not in self._systems:
            from langchain_community.vectorstores import FAISS

            chunker = get_chunker(chunker_name)
            rag = RAGDemo(self.api_key, chunker=chunker,
                          embedding_model=embedding_model,
                          llm_cache=self.llm_cache)
            if rag.llm is not None:
                rag.llm.max_retries = 0      # eval.llm.with_backoff owns retries
            chunks = chunker.chunk(self.raw)
            texts = [c["text"] for c in chunks]
            vectors = self.emb_cache.get_or_compute(
                embedding_model, texts, rag.embeddings.embed_documents)
            docs = RAGDemo._to_langchain_docs(chunks)
            rag.vectorstore = FAISS.from_embeddings(
                list(zip(texts, vectors.tolist())), rag.embeddings,
                metadatas=[d.metadata for d in docs])
            rag.retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 3})
            self._systems[key] = (rag, chunks)
        return self._systems[key]

    @property
    def judge(self):
        if self._judge is None:
            from eval.metrics.generation import Judge
            self._judge = Judge(self.api_key, self.llm_cache)
        return self._judge


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def evaluate(config: EvalConfig, ws: Workspace, full: bool) -> list[Trace]:
    rag, chunks = ws.system(config.chunker, config.embedding_model)
    traces = []

    for item in ws.dataset:
        gold = [tuple(s) for s in item[GROUND_TRUTH]]
        latency = {}

        t = time.perf_counter()
        ranked = [c for c, _ in rag.retrieve(item["question"], top_k=DIAGNOSTIC_DEPTH)]
        latency["retrieve"] = _ms(t)
        retrieved = ranked[:config.top_k]
        r = retrieval_metrics(retrieved, gold)

        trace = Trace(
            qid=item["id"], qtype=item["qtype"], chapter=item["chapter"],
            coverage=coverage(gold, chunks),
            fragmentation=fragmentation(gold, chunks),
            hit_at_k=r["hit_at_k"], recall_at_k=r["recall_at_k"],
            precision_at_k=r["precision_at_k"], mrr=r["mrr"],
            rank_of_gold=rank_of_gold(ranked, gold),
            retrieved_metadata=[c["metadata"] for c in retrieved],
            n_context_chars=sum(len(c["text"]) for c in retrieved),
            latency_ms=latency,
        )
        if full:
            _generate_and_judge(trace, item, rag, ws, retrieved,
                                chunks_overlapping(gold, chunks))
        traces.append(trace)
    return traces


def _generate_and_judge(trace: Trace, item: dict, rag: RAGDemo, ws: Workspace,
                        retrieved: list[dict], gold_chunks: list[dict]) -> None:
    from eval.metrics.generation import abstained, cited_articles, citation_accuracy

    from eval.llm import with_backoff

    q = item["question"]
    for label, context in (("oracle", gold_chunks), ("retrieved", retrieved), ("noctx", [])):
        t = time.perf_counter()
        # Not rag.generate(): that turns API errors into an answer string,
        # which the judge would then grade as if the model had said it.
        prompt = rag.build_prompt(rag._sanitize_input(q), context)
        answer = with_backoff(lambda: rag._complete(prompt))
        setattr(trace, f"a_{label}", answer)
        trace.latency_ms[f"generate_{label}"] = _ms(t)

        # What this question would cost in production, generation only; the
        # judge is an evaluation expense and is not counted here.
        n_in = ws.tokens.count(rag.llm_model, prompt)
        n_out = ws.tokens.count(rag.llm_model, answer)
        trace.tokens[label] = {"input": n_in, "output": n_out}
        trace.cost_usd[label] = price(rag.llm_model, n_in, n_out)

    judge = ws.judge
    for label in ("oracle", "retrieved", "noctx"):
        v = judge.correctness(q, getattr(trace, f"a_{label}"), item["reference_answer"])
        setattr(trace, f"correct_{label}", bool(v.score))
        trace.judge_spread[f"correct_{label}"] = v.samples
        trace.judge_reasoning[f"correct_{label}"] = v.reasoning

    v = judge.faithfulness(trace.a_retrieved, retrieved)
    trace.faithful = bool(v.score)
    trace.judge_spread["faithful"] = v.samples
    trace.judge_reasoning["faithful"] = v.reasoning

    trace.has_citation = bool(cited_articles(trace.a_retrieved))
    trace.citation_ok = citation_accuracy(trace.a_retrieved, retrieved, ws.article_spans)
    trace.abstained = abstained(trace.a_retrieved)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_results(config: EvalConfig, traces: list[Trace], ws: Workspace, full: bool) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{config.name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for tr in traces:
            f.write(json.dumps(asdict(tr), ensure_ascii=False) + "\n")

    _, chunks = ws.system(config.chunker, config.embedding_model)
    meta = {
        "config": asdict(config) | {"name": config.name},
        "mode": "full" if full else "fast",
        "ground_truth": GROUND_TRUTH,
        "generator_model": DEFAULT_LLM_MODEL if full else None,
        "judge_model": ws.judge.model if full else None,
        "judge_runs": ws.judge.runs if full else None,
        "dataset_file": RESOLVED.name,
        "dataset_hash": dataset_hash(ws.dataset),
        "chunker_code_hash": _sha_file(BACKEND / "src" / "chunker.py"),
        "prompt_code_hash": _sha_file(BACKEND / "src" / "rag_system.py"),
        "parse": parse_smoke(ws.raw),
        "chunks": structural_metrics(chunks, ws.raw),
        "n_questions": len(traces),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (RESULTS / f"{config.name}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def run(config: EvalConfig, ws: Workspace, full: bool) -> list[Trace]:
    traces = evaluate(config, ws, full)
    path = write_results(config, traces, ws, full)
    scored = [t for t in traces if t.hit_at_k is not None and t.coverage == 1.0]
    hit = sum(t.hit_at_k for t in scored) / len(scored) if scored else float("nan")
    print(f"{config.name:<26} hit@{config.top_k}={hit:.2f} on {len(scored)} q  -> {path.name}")
    if full:
        print(f"  {ws.llm_cache.stats()}")
    return traces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default="hierarchical")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="no LLM (default)")
    mode.add_argument("--full", action="store_true", help="generation + judge")
    args = ap.parse_args()

    ws = Workspace(api_key=os.getenv("GEMINI_API_KEY", ""))
    run(EvalConfig(args.chunker, args.top_k, args.embedding_model), ws, full=args.full)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
