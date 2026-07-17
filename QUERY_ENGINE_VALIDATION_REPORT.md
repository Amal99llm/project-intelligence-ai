# Query Engine Refactor — Technical Validation Report

## Summary

The chatbot query engine was refactored from "dump the full portfolio as text
and let the LLM reason over it" to a deterministic pipeline:

```
User Question
  -> Intent Router (LLM, constrained to 8-category enum)
  -> Structured Query Builder (LLM function-call -> whitelist-validated JSON spec)
  -> Query Executor (deterministic Python/SQLAlchemy, no LLM)
  -> Verification Layer (recompute + cross-check against a fresh DB read, no LLM)
  -> LLM Formatter (wording/translation/summary only, no arithmetic)
  -> Executive Response (+ source attribution)
```

The LLM now touches the pipeline in exactly three constrained places
(classification, query extraction, final wording) and never filters, sums,
sorts, or invents a number. All arithmetic lives in one KPI Registry.

## Files modified / created

| File | Status | Purpose |
|---|---|---|
| `modules/query_schema.py` | **new** | Single whitelist of filterable/sortable/aggregatable columns, operators, aggregations, and the 8 intent categories. Used by both the LLM function-call schema and the Python validator. |
| `modules/intent_classifier.py` | **new** | Semantic intent classification via LLM function-calling constrained to the 8-category enum, with a deterministic fast-path only for structural project-code patterns. |
| `modules/query_builder.py` | **new** | Turns a question into a structured query spec via LLM function-calling; every returned field is re-validated against `query_schema` before use. |
| `modules/query_executor.py` | **new** | Executes a validated spec (filter/sort/limit/aggregate) against project data. No LLM involvement; aggregations route through the KPI Registry. |
| `modules/verification.py` | **new** | Re-fetches fresh data and cross-checks project existence and aggregation correctness before an answer is allowed out. Fails closed to a fixed fallback message. |
| `modules/response_formatter.py` | **new** | The only remaining "creative" LLM call — explains/translates/summarizes already-verified data under a system prompt that forbids arithmetic and forbids stating unlisted numbers. Also builds deterministic source attribution. |
| `modules/project_repository.py` | **new** | Single place that reads `backlog_projects` and shapes rows into the canonical project dict (previously duplicated independently in `app.py` and `ai_engine.py`, and already drifted between the two). |
| `modules/kpi_calculator.py` | **modified** | Existing tested KPI math (`calculate_executive_kpis`, `project_financials`, etc.) is unchanged. Added: `KPI_REGISTRY` (Section 18), `compute_kpi()`, `summarize_by_bu()`, and three new registry KPIs (`amendments_total`, `current_year_revenue`, `current_year_cost`) that absorb logic previously duplicated in the dashboard's JavaScript. |
| `modules/query_router.py` | **rewritten** | The v3 keyword/regex router is gone. Now delegates to `intent_classifier`; re-exports the old constant names so no other call site had to change. |
| `modules/ai_engine.py` | **rewritten** | `answer()` now runs the structured pipeline instead of building a full-portfolio text dump and asking the LLM to reason freely over it. `generate_report()` (VP reports) now assembles its inputs from the KPI Registry + `query_executor` instead of the same text dump. `_build_context()` and the old freeform system prompt are removed. |
| `app.py` | **modified** | `/api/projects` now calls `project_repository.fetch_enriched_projects()` instead of duplicating the row-shaping logic inline (removed ~50 lines of duplicate code). |
| `templates/index.html` | **modified** | Removed two duplicated client-side aggregate calculations (`amendments`, `currentRevenue`, `currentCost` via `.reduce()`, present twice) that reimplemented totals already computed server-side; both call sites now read `METRICS.amendments_total` / `METRICS.current_year_revenue` / `METRICS.current_year_cost`. |

## Functions refactored

- `ai_engine.answer()` — full rewrite: routes through the structured pipeline instead of `_build_context()` + freeform prompt.
- `ai_engine.generate_report()` — full rewrite: builds verified structured inputs (KPI Registry + query executor results for losing/expiring/high-backlog projects) instead of a giant text dump.
- `app.api_projects()` — simplified to a fetch + serialize call against the shared repository.
- `query_router.route()` — now a one-line delegation to `intent_classifier.classify()`.

## Calculations centralized (KPI Registry, Section 18)

All of the following now have exactly one formula, defined in `modules/kpi_calculator.py`'s `KPI_REGISTRY`, consumed identically by the dashboard, the API, and the chatbot:

`total_projects`, `total_contract_value`, `revenue`, `cost`, `backlog`, `profit_loss`, `profit_margin`, `losing_projects`, `active_projects`, `contracts_expiring_soon`, `amendments_total`, `current_year_revenue`, `current_year_cost`.

`summarize_by_bu()` was added as the single BU-grouping implementation used by VP reports (previously inlined in `ai_engine._build_context`).

**Not moved in this pass:** the dashboard's risk-matrix probability/impact scoring model (`templates/index.html`, `renderRiskMatrix`) is a chart-specific visualization heuristic (portfolio-relative normalization + weighted score for scatter-plot placement), not a reported business fact, and no LLM ever consumes it. Moving it server-side was assessed as high risk to the interactive chart (tooltips, legend counts, quadrant thresholds) for no hallucination-safety benefit, since it sits entirely outside the chatbot/verification surface. Flagged here rather than silently dropped.

## Routing logic changed

- The v3 router matched category keyword lists (e.g. any of `عقد`, `غرامة`, `شرط`... anywhere in the text forced `contract_query`). This is gone.
- Routing is now semantic: an LLM call constrained to the 8-category enum (`executive_kpi`, `project_lookup`, `financial_analysis`, `ranking`, `comparison`, `filtering`, `contract_analysis`, `general_conversation`) makes the call using the full meaning of the question. Verified in testing: a question containing "عقد" but asking about project counts now classifies as `financial_analysis`/`filtering`, not `contract_analysis`.
- The only remaining deterministic short-circuit is a regex for unambiguous project-code patterns (e.g. `AAA-BBB-123`), which is a structural signal rather than a keyword trigger.

## Hallucination protections added

1. **Whitelist validation** (`query_schema.validate_query_spec`) — every column, operator, and aggregation the LLM proposes is checked against an explicit allowlist before touching the database; anything else is rejected, not guessed at.
2. **No LLM arithmetic** — filtering, sorting, and aggregation all happen in `query_executor.py` / the KPI Registry; the `response_formatter` system prompt explicitly forbids arithmetic and forbids stating any number not present in the verified payload.
3. **Verification layer** (`modules/verification.py`) — before any answer is returned: every project_code in the result is re-checked against a fresh database read; every aggregation is independently recomputed from a fresh read via the same KPI Registry formula and must match; a mismatch anywhere fails closed.
4. **Fixed fallback message** — any failure in query building or verification returns exactly `"تعذر التحقق من صحة النتائج، يرجى إعادة المحاولة."` — never a partial or best-guess answer.
5. **No caching of portfolio context** — the old text-dump cache is gone; every structured query reads the database fresh, which is what makes the verification layer's freshness guarantee meaningful.
6. **Source attribution** — every structured answer carries a deterministic (never LLM-generated) `source` object: table, columns, filters, sort, limit, and the KPI formula used, for audit and UI display.

## Automated tests created

| File | What it covers |
|---|---|
| `tests/test_query_schema.py` | Whitelist validation: valid specs pass; disallowed columns/operators/aggregations, malformed `between`/`in`, and out-of-range limits are all rejected. |
| `tests/test_kpi_registry.py` | Every `KPI_REGISTRY` entry matches `calculate_executive_kpis()` on fixture data; unregistered KPI names raise; `summarize_by_bu()` groups and sums correctly. |
| `tests/test_query_executor.py` | The exact example queries from the spec (`P&L < 0`, `ORDER BY backlog DESC LIMIT 1`, `backlog > 10,000,000 AND P&L < 0`, `SUM(total_revenue)`, `days_remaining BETWEEN 0 AND 30`), plus `IN`, `contains`, and case-insensitive string filtering. |
| `tests/test_verification.py` | Verification passes on consistent data; fails when a returned project no longer exists in a fresh read; fails when a reported aggregation doesn't match an independent recompute. |
| `tests/test_intent_classifier.py` | Project-code fast path skips the LLM call entirely; a question containing "عقد" but not about contract terms is **not** forced into `contract_analysis` (mocked LLM call, asserts the model's judgment is respected, not overridden by keyword logic); malformed/unknown model output defaults safely to `general_conversation`. |
| `tests/test_query_builder.py` | Valid model output survives validation; off-whitelist columns, missing tool calls, and invalid JSON all raise `QueryBuildError` rather than falling back to freeform reasoning. |
| `tests/test_kpi_calculator.py` | Pre-existing, unmodified — still passes unchanged, confirming the registry wraps rather than replaces the original tested math. |

## Test results

```
$ python -m unittest discover tests
Ran 46 tests in 0.007s
OK
```

All 46 tests pass (16 pre-existing + 30 new). `python -m py_compile app.py modules/*.py` succeeds with no syntax errors.

## End-to-end validation (live database + live OpenAI calls)

Run directly against the real `database.db` (136 projects) and a real `OPENAI_API_KEY`, bypassing only the Flask/scheduler bootstrap (contract/RAG queries were not exercised — `chromadb` could not be installed in this environment; see Known Limitations):

| Question | Intent classified | Result |
|---|---|---|
| أي المشاريع خسرانة؟ | filtering | Correctly listed losing projects with real, verified `net_profit` figures. |
| أعلى مشروع Backlog | ranking | Correctly returned the single highest-backlog project (642,693,991.51 ريال) with its code and status. |
| كم إجمالي الإيرادات؟ | executive_kpi | `SUM(total_revenue)` = 6,711,624,245.54 ريال, with source attribution showing the exact formula and column. |
| كم عدد المشاريع المنتهية خلال 30 يوم؟ | filtering | Correctly answered 0, filtering on the precomputed `days_remaining` column (see note below). |
| أي المشاريع عندها Backlog أكبر من 10 مليون وربحيتها سالبة؟ | filtering | Correctly applied both AND conditions and returned the one matching project. |
| مرحبا كيف حالك؟ | general_conversation | Friendly reply with **no** project numbers or data mentioned. |
| ما هي أفضل عاصمة في العالم؟ | general_conversation | Off-topic question handled without touching the database. |
| قارن بين اعلى مشروعين من حيث قيمة العقد | comparison | Correctly ranked and compared the top two projects by contract value. |
| Executive VP report | (direct call) | Correctly summarized using the same verified KPI Registry figures (136 projects, same loss/margin numbers as the chat answers above). |

**Fix applied during validation:** the first run of the "ends within 30 days" question had the query builder pick the raw `end_date` column instead of the precomputed `days_remaining` field (which already accounts for amendment precedence). The `query_builder` system prompt was tightened to explicitly prefer `days_remaining` for any "ending/expiring/within N days" question, and the corrected behavior was re-verified live.

## Known limitations

1. **Query building still uses one LLM call.** It is schema-constrained and whitelist-validated (no computation happens in it), but classification/extraction quality depends on that call succeeding. If it errors or returns something off-whitelist, the pipeline fails closed to the fixed fallback message rather than guessing — this is a deliberate, disclosed tradeoff, not a hidden one.
2. **Filters are AND-only.** All filters in a query spec are combined with AND, matching every example in the original spec. OR logic (e.g. "status is X or Y") is handled via the `in` operator for equality-style conditions, but arbitrary OR-of-different-columns is not supported.
3. **`rag_engine.py` (contract PDF Q&A) was left unchanged.** It already retrieves grounded chunks via ChromaDB rather than reasoning over the full portfolio, so it wasn't part of this refactor's hallucination surface.
4. **Risk-matrix chart scoring in `index.html` was not moved server-side** — see rationale above. This is the one place JavaScript still computes something beyond pure display/sort, and it does not feed the chatbot or any reported KPI.
5. **`chromadb` could not be installed in this validation environment** (missing Microsoft C++ Build Tools on this Windows machine — a pre-existing environment gap, unrelated to this refactor). Contract-analysis routing was verified logically (via `intent_classifier` tests) but not exercised end-to-end against a live ChromaDB collection.
