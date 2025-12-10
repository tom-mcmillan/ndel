# NDEL Philosophy: A Non-Deterministic Semantic Protocol

NDEL stands for Non-Deterministic Expression Language.

NDEL separates three concerns:
- **Static signals (library)**: Extract structural hints from Python and SQL—datasets, transformations, features, models, metrics—and build a semantic model with lineage.
- **Semantic interpretation (LLM)**: An LLM phrases and contextualizes those signals into NDEL text. The wording is non-deterministic and improves as LLMs improve.
- **DSL structure (NDEL)**: A linguistic framework and protocol for describing pipelines; not a compiler or code generator.

Think of NDEL as the missing semantic layer between raw DS/ML code and LLM reasoning. You hand an LLM a pipeline graph, privacy-filtered and domain-shaped, plus a DSL schema. The LLM then writes the NDEL description within that envelope—different LLMs may phrase things differently, and that is expected.

## Why non-determinism?
- **Flexibility**: Different teams and domains can adopt their own dialects and phrasings.
- **Domain adaptation**: LLMs can reinterpret the same structure with domain-specific nuance.
- **Human-aligned variability**: Two capable LLMs might describe the same feature differently (e.g., "monthly sessions" vs. "sessions per month")—both are acceptable views through the NDEL lens.

Illustrative variation:
- LLM A: "compute monthly sessions per user to capture activity intensity"
- LLM B: "derive sessions_per_month as a normalized engagement signal"
Both are faithful to the same transformation but differ in narrative.

## Guidance for teams
- Expect and embrace variability in phrasing; the semantics matter more than exact wording.
- Review and version-control configs (e.g., `ndel_profile.py`); regenerated configs may shift phrasing as prompts/LLMs evolve.
- Use NDEL as a protocol for reasoning over pipelines, not as a deterministic AST-to-DSL translator.
