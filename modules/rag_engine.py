"""
modules/rag_engine.py
---------------------
Step 8b: RAG Engine for contract questions.
Searches ChromaDB for relevant chunks, builds context, calls LLM.

Fix (see audit history): this module always supported a `project_code`
filter, but modules.ai_engine's contract-intent branch never passed one,
so every contract question searched across every project's contracts with
no scope at all -- a real cross-project data-leak/wrong-answer path (a
question about Project A's contract could be answered using Project B's
contract text).

`answer_contract_query` now REQUIRES a resolved `project_code` -- there is
no default and no "search everything" fallback. The caller (modules.ai_engine)
is responsible for resolving the project via modules.project_entity_resolver
*before* calling this, and must not call it at all if no single project was
resolved with confidence.

As defense in depth (in case of a metadata bug or a ChromaDB driver that
ignores `where`), every retrieved chunk's own `project_code` metadata is
re-checked against the requested one here; any chunk that doesn't match is
dropped rather than trusted. If nothing usable remains, the function
refuses rather than answering from the model's general knowledge.
"""

from __future__ import annotations

import logging
import re

try:
    import chromadb
except ImportError:  # Allows non-RAG tests and local BI features to run independently.
    chromadb = None
from openai import AzureOpenAI

import config

logger = logging.getLogger(__name__)

_chroma_client     = None
_chroma_collection = None
_openai_client     = None

NO_CONTRACT_DATA_MESSAGE_TEMPLATE = (
    "لا توجد بيانات عقد كافية لمشروع {name} ({code}) للإجابة على هذا السؤال. "
    "الرجاء رفع ملف العقد الخاص بهذا المشروع أولًا إذا لم يكن مرفوعًا."
)
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "لا أستطيع تأكيد الإجابة من نص العقد المتاح حاليًا لهذا المشروع."
)
CONFLICTING_EVIDENCE_MESSAGE = (
    "النصوص المتاحة من عقد المشروع تحتوي قيمًا متعارضة مرتبطة بالسؤال، "
    "لذلك لا أستطيع اختيار قيمة منها دون مرجع أوضح."
)


_NUMBER = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?")
_DURATION = re.compile(r"\d+(?:[.,]\d+)?\s*(?:يوم|يوما|أيام|day|days|%|٪)", re.IGNORECASE)


def _numeric_tokens(text: str) -> set[str]:
    return {match.group(0).replace(",", "") for match in _NUMBER.finditer(text or "")}


def _has_conflicting_contract_values(query: str, chunks: list[dict]) -> bool:
    """Flag contradictory answer-shaped durations/percentages across chunks.

    General numbers such as clause identifiers are intentionally ignored.
    """
    values = set()
    for chunk in chunks:
        values.update(normalize.group(0).replace(" ", "") for normalize in _DURATION.finditer(chunk.get("text", "")))
    return len(values) > 1


class ContractQueryError(Exception):
    """Raised when answer_contract_query is called without a resolved
    project scope -- callers must resolve a project first, never catch
    this to silently fall back to an unscoped search."""


def _get_chroma():
    global _chroma_client, _chroma_collection
    if chromadb is None:
        raise ContractQueryError("ChromaDB is not installed in this environment")
    if _chroma_collection is None:
        _chroma_client     = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION
        )
    return _chroma_collection


def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    return _openai_client


def _embed_query(query: str) -> list[float]:
    client = _get_openai()
    resp   = client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
    return resp.data[0].embedding


def search_contracts(query: str, project_code: str) -> list[dict]:
    """
    Search ChromaDB for the most relevant contract chunks, scoped to a
    single project. `project_code` is required -- there is no unscoped
    search mode in this module; an unscoped RAG search across every
    project's contracts is exactly the leak/wrong-answer path this module
    exists to prevent.

    Returns list of { text, contract_ref, project_code, chunk_id, score }.
    Any chunk whose own metadata.project_code doesn't match the requested
    project_code is dropped (defense in depth against a Chroma `where`
    filter being bypassed or metadata being malformed at ingestion time).
    """
    if not project_code or not str(project_code).strip():
        raise ContractQueryError("search_contracts requires a resolved project_code")
    project_code = str(project_code).strip()

    collection = _get_chroma()
    if collection.count() == 0:
        return []

    query_embedding = _embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config.TOP_K_RESULTS,
        where={"project_code": project_code},
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    dropped = 0
    for doc, meta, dist, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0] if results.get("ids") else [None] * len(results["documents"][0]),
    ):
        meta_project_code = str(meta.get("project_code", "")).strip()
        if meta_project_code != project_code:
            # Should be unreachable given the `where` filter above; kept as
            # an explicit, logged safety net rather than a silent trust.
            dropped += 1
            continue
        chunks.append({
            "text":         doc,
            "contract_ref": meta.get("contract_ref", ""),
            "project_code": meta_project_code,
            "chunk_id":     chunk_id,
            "score":        round(1 - dist, 4),   # cosine similarity
        })

    if dropped:
        logger.warning(
            "search_contracts: dropped %d chunk(s) with mismatched project_code "
            "metadata for requested project %s -- check ingestion metadata.",
            dropped, project_code,
        )
    return chunks


def answer_contract_query(
    query: str, project_code: str, project_display_name: str | None = None,
) -> str:
    """
    Full RAG pipeline for contract questions, scoped to exactly one
    project. `project_code` is required and must already have been
    resolved with confidence (modules.project_entity_resolver) by the
    caller -- this function does not attempt to guess or search broadly.

    Raises ContractQueryError if project_code is missing/empty, so a
    caller can never accidentally fall through to an unscoped search.
    """
    if not project_code or not str(project_code).strip():
        raise ContractQueryError("answer_contract_query requires a resolved project_code")
    project_code = str(project_code).strip()
    display = project_display_name or project_code

    chunks = search_contracts(query, project_code)

    if not chunks:
        return NO_CONTRACT_DATA_MESSAGE_TEMPLATE.format(name=display, code=project_code)

    # A very low top similarity score means nothing retrieved is actually
    # relevant to the question, even though it's the right project --
    # refuse rather than let the model reach for general knowledge to
    # paper over a weak match.
    if max(c["score"] for c in chunks) < 0.15:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    if _has_conflicting_contract_values(query, chunks):
        return CONFLICTING_EVIDENCE_MESSAGE

    context = "\n\n---\n\n".join(
        f"[Project: {c['project_code']} | Contract: {c['contract_ref']} | Chunk: {c['chunk_id']}]\n{c['text']}"
        for c in chunks
    )

    system_prompt = (
        "You are an enterprise contract analyst assistant. You are given contract excerpts "
        f"belonging ONLY to project {project_code} ({display}). "
        "Answer strictly and only from these excerpts. "
        "If the answer is not contained in these excerpts, say so clearly in the same language "
        "as the question -- do not use outside knowledge, do not guess, and do not generalize "
        "from how similar contracts are typically structured. "
        "Never state a figure, date, or clause that is not literally present in the excerpts. "
        "Cite the contract reference and chunk when possible. "
        "Respond in the same language the user used."
    )

    user_prompt = (
        f"Project: {project_code} ({display})\n\n"
        f"Contract excerpts:\n\n{context}\n\n"
        f"Question: {query}"
    )

    client   = _get_openai()
    response = client.chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()
    allowed_numbers = _numeric_tokens(context)
    if not _numeric_tokens(answer).issubset(allowed_numbers):
        logger.warning("RAG answer rejected: it introduced a number absent from retrieved chunks")
        return INSUFFICIENT_EVIDENCE_MESSAGE
    logger.info("RAG answer generated for project %s | query: %.80s", project_code, query)
    return f"{answer}\n\n(المصدر: عقد مشروع {display} — {project_code})"
