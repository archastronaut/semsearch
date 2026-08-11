# Local Semantic File Search

The goal is to find notes by meaning, not exact wording. Full-text search
only matches literal keywords.

Candle had version-stability issues, so the embedding layer uses fastembed-rs
instead, running nomic-embed-text locally.

Next steps: wire up the vector store, then add hybrid keyword + semantic
search with tantivy.
