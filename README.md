# Local AI Embeddings in Rust

Can Rust run small AI embedding models (like all-MiniLM) fully offline, no cloud calls? **Yes** — but the tools differ a lot in reliability.

## Bottom line

**Use `fastembed-rs`.** It's the easiest and most reliable way to run embedding models locally in Rust — popular models work out of the box, no manual setup.

## The options

| Tool | Ease of use | Catch |
|---|---|---|
| **fastembed-rs** | Easiest — models built in | Relies on an early-version library (`ort`) |
| **candle** (Hugging Face's Rust ML lib) | Medium — flexible, but fragile | Has had breaking updates and broken example code |
| **rust-bert** | Hardest | Needs PyTorch (libtorch) installed alongside it |

## Problems found with candle specifically

- A routine update once broke other projects that depended on it.
- Its own official example code has failed to even compile.
- Loading a standard AI model into it has caused unexplained errors.

## Recommendation

Start with `fastembed-rs`. Only reach for `candle` if you hit a real limitation it solves and you're OK debugging occasional breakage.

---
*Based on live GitHub issues, crate docs, and changelogs as of Aug 2026.*
