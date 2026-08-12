# Local AI Embeddings in Rust

Can Rust run small AI embedding models (like all-MiniLM) fully offline, no cloud calls? **Yes** — but the tools differ a lot in reliability.

## The options

| Tool | Ease of use | Catch |
|---|---|---|
| **fastembed-rs** | Easiest — models built in | Relies on an early-version library (`ort`) |
| **candle** (Hugging Face's Rust ML lib) | Medium — flexible, but fragile | Has had breaking updates and broken example code |
| **rust-bert** | Hardest | Needs PyTorch (libtorch) installed alongside it |

## Recommendation

Start with `fastembed-rs`. Only reach for `candle` if you hit a real limitation it solves and you're OK debugging occasional breakage.

## Which model to embed with

Skip the default `all-MiniLM-L6-v2` — it's tuned for sentence similarity, not search. Use one of these instead (both are in fastembed-rs's built-in list):

| Model | Context length | Best for |
|---|---|---|
| **BGE-small-en-v1.5** | Short (~512 tokens) | Small chunks, faster, lower memory |
| **nomic-embed-text** | Long (~8k tokens) | Bigger chunks, less chunking work, slightly heavier |

**Short vs. long context:** context length = how much text the model can embed in one pass. Short-context models force you to split files into more, smaller chunks before embedding. Long-context models let you embed bigger pieces (or whole files) at once — simpler pipeline, fewer chunks to manage — at the cost of more compute per embedding.

---
*Based on live GitHub issues, crate docs, and changelogs as of Aug 2026.*
