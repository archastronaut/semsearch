# Local Embedding-Model Support in Rust: Candle vs. Alternatives

Research note, 2026-08-10.

## Abstract

Candle (Hugging Face's Rust ML framework) can load and run small sentence-embedding
models such as all-MiniLM-L6-v2 fully offline: it natively supports safetensors
weights and the `tokenizers` crate. However, its public API has shipped
semver-breaking changes inside a nominal *patch* release, and its official BERT
example has independently failed from unrelated compile errors and from an
unresolved tensor-loading mismatch on a stock HF checkpoint. For a background
indexing daemon, `fastembed-rs` (an ONNX Runtime wrapper via the `ort` crate) is
the lower-risk alternative: it ships MiniLM/BGE/mpnet models built-in with no
manual weight/tokenizer wiring, at the cost of depending on `ort`, which is
itself still pre-1.0.

## Method

Automated multi-agent research: question decomposed into 5 search angles → 6
parallel web searches per angle → 16 sources fetched and deduplicated → 75
falsifiable claims extracted → top 25 claims adversarially verified (3
independent votes each, ≥2 refutes kills a claim) → confirmed claims
synthesized below. 17/25 claims confirmed, 8/25 refuted (listed for
transparency). No claim survived on a single source alone where corroboration
was checked.

## Findings

| # | Claim | Confidence | Source |
|---|---|---|---|
| 1 | Candle loads embedding weights from safetensors (`candle_core::safetensors::load`) and integrates with the HF `tokenizers` crate. | High | [candle repo](https://github.com/huggingface/candle) |
| 2 | candle-core 0.9.1→0.9.2 (a patch release) added 7 new `DType` enum variants, breaking downstream crate `burn-import` via non-exhaustive match — a real semver violation, later patched with `#[non_exhaustive]`. | High | [candle#3333](https://github.com/huggingface/candle/issues/3333) |
| 3 | candle-transformers' official BERT/embedding example has failed to *compile* from unrelated borrow-checker errors elsewhere in the crate (building one example compiles the whole crate); recurred across multiple issues, including against a tagged release. | High | [candle#2598](https://github.com/huggingface/candle/issues/2598) |
| 4 | Loading a stock HF `bert-base-uncased` safetensors checkpoint into candle-transformers' BERT raises an unresolved `TensorNotFound` error. Exact root cause unconfirmed (two proposed explanations were checked and refuted). | Medium | [candle#1887](https://github.com/huggingface/candle/issues/1887) |
| 5 | Candle's official BERT sentence-embedding example uses a pooling strategy that produces different output vectors than HF's own sentence-transformers implementation for the same model — a correctness gap, not a build issue. | Medium | [candle#380](https://github.com/huggingface/candle/issues/380) |
| 6 | Candle-based embedding wrapper crates exist (`candle_embed`) but are effectively unmaintained — no release in ~2 years against ~10 candle-core releases in the same window. | Medium | [crates.io/candle_embed](https://crates.io/crates/candle_embed) |
| 7 | `fastembed-rs` is a purpose-built, synchronous (no Tokio), ONNX-Runtime-backed crate shipping all-MiniLM-L6/L12-v2, mpnet, BGE-small (default), and nomic-embed as built-in models, cached locally after first download. | High | [fastembed-rs](https://github.com/anush008/fastembed-rs), [crates.io](https://crates.io/crates/fastembed) |
| 8 | `fastembed-rs` pins its inference backend `ort` to a pre-1.0 release candidate (`2.0.0-rc.13`); `ort`'s prior 1.x line was yanked for use-after-free/leak bugs — fastembed's long-term stability is bounded by this upstream dependency. | Medium | [crates.io/fastembed](https://crates.io/crates/fastembed) |
| 9 | `rust-bert` depends on libtorch via `tch-rs` (manual install + env vars, or a "download-libtorch" build feature) rather than staying pure-Rust — a heavier deployment footprint than candle or fastembed-rs. | High | [rust-bert](https://github.com/guillaume-be/rust-bert) |

## Conclusion

Candle satisfies the baseline technical requirement (safetensors + tokenizer
loading) but carries documented, non-hypothetical API-churn and
checkpoint-loading risk. For a background daemon prioritizing low integration
risk, default to **fastembed-rs**. Choose candle only if direct model-format
flexibility or avoiding the `ort` dependency chain is a hard requirement — and
pin the candle version tightly, testing the load against the exact checkpoint
before building further architecture on top.

## Open questions

- No verified CPU throughput/latency benchmarks survived adversarial review — only anecdotal reports.
- Whether `ort`'s prebuilt-binary download complicates a fully offline, single-binary daemon build, comparable to libtorch's footprint for rust-bert.
- Whether candle#1887 (TensorNotFound) is still reproducible on the latest candle-transformers release.

## Provenance

Generated via automated deep-research (5 search angles, 16 sources, 75 claims
extracted, 25 verified with 3-vote adversarial checking). Full claim-level
detail, refuted claims, and per-source metadata available on request.
