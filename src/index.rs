use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::chunk;
use crate::embed::Embedder;

/// Discover and chunk every .md/.txt file under `dir` (a directory, or a
/// single file). Cheap — no model work — so callers can bail out early when
/// there is nothing to embed, before paying the model load/download.
pub fn collect_chunks(dir: &Path) -> Result<Vec<chunk::Chunk>> {
    let files = chunk::discover_files(dir);

    // Fail fast on problem files rather than silently skipping them — a
    // silent skip would mean silently incomplete search results later.
    let mut all_chunks = Vec::new();
    for path in &files {
        // serde serializes PathBuf as a UTF-8 string, so a non-UTF-8 file
        // name would make save() fail *after* the expensive embedding pass,
        // with an error that names no file. Reject it here instead.
        anyhow::ensure!(
            path.to_str().is_some(),
            "non-UTF-8 file name (unsupported): {}",
            path.display()
        );
        all_chunks.extend(chunk::read_and_chunk(path)?);
    }
    Ok(all_chunks)
}

/// A chunk of text plus the embedding vector computed for it. This is the
/// on-disk unit: build once with `Index::build`, save it, and later runs
/// load it back instead of re-running the model over every file.
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexedChunk {
    pub path: PathBuf,
    pub index: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Index {
    pub chunks: Vec<IndexedChunk>,
}

/// One scored search result. Borrows from the `Index` it was produced from
/// rather than cloning chunks, since a search only returns a handful.
pub struct SearchHit<'a> {
    pub chunk: &'a IndexedChunk,
    pub score: f32,
}

impl Index {
    /// Embed pre-collected chunks in one batch. This is the slow step
    /// (model inference on every chunk) — everything after this is just
    /// math on the vectors. Takes chunks and embedder separately so callers
    /// can check for emptiness *before* paying the model load, and so `ask`
    /// can reuse one loaded model for both the documents and the query.
    pub fn from_chunks(chunks: Vec<chunk::Chunk>, embedder: &mut Embedder) -> Result<Self> {
        if chunks.is_empty() {
            return Ok(Self::default());
        }

        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = embedder.embed_documents(&texts)?;

        let chunks = chunks
            .into_iter()
            .zip(embeddings)
            .map(|(c, embedding)| IndexedChunk {
                path: c.path,
                index: c.index,
                text: c.text,
                embedding,
            })
            .collect();

        Ok(Self { chunks })
    }

    /// Serialize with postcard (compact binary, no schema ceremony) and
    /// write to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = postcard::to_allocvec(self).context("failed to serialize index")?;
        fs::write(path, bytes)
            .with_context(|| format!("failed to write index to {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read index from {}", path.display()))?;
        postcard::from_bytes(&bytes).context("failed to deserialize index")
    }

    /// Rank every chunk by cosine similarity to `query_embedding` and return
    /// the top `top_k`. Brute-force linear scan — fine up to tens of
    /// thousands of chunks; an ANN index (e.g. hnsw) only earns its keep if
    /// this ever shows up as the bottleneck.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchHit<'_>> {
        let mut scored: Vec<SearchHit> = self
            .chunks
            .iter()
            .map(|c| SearchHit {
                chunk: c,
                score: cosine_similarity(query_embedding, &c.embedding),
            })
            .collect();

        // total_cmp rather than partial_cmp: it's a true total order, so the
        // sort can't panic even if a corrupted index file smuggles in a NaN.
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(top_k);
        scored
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_score_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_score_zero() {
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_scores_zero_not_nan() {
        let score = cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn search_ranks_closest_first_and_respects_top_k() {
        let index = Index {
            chunks: vec![
                IndexedChunk {
                    path: "a.md".into(),
                    index: 0,
                    text: "a".into(),
                    embedding: vec![1.0, 0.0],
                },
                IndexedChunk {
                    path: "b.md".into(),
                    index: 0,
                    text: "b".into(),
                    embedding: vec![0.0, 1.0],
                },
                IndexedChunk {
                    path: "c.md".into(),
                    index: 0,
                    text: "c".into(),
                    embedding: vec![0.9, 0.1],
                },
            ],
        };

        let hits = index.search(&[1.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk.text, "a");
        assert_eq!(hits[1].chunk.text, "c");
    }

    #[test]
    fn top_k_larger_than_chunk_count_returns_all() {
        let index = Index {
            chunks: vec![IndexedChunk {
                path: "a.md".into(),
                index: 0,
                text: "a".into(),
                embedding: vec![1.0, 0.0],
            }],
        };
        assert_eq!(index.search(&[1.0, 0.0], 100).len(), 1);
    }

    #[test]
    fn empty_index_returns_no_hits() {
        assert!(Index::default().search(&[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn save_load_roundtrip_preserves_chunks() {
        let index = Index {
            chunks: vec![IndexedChunk {
                path: "notes/a.md".into(),
                index: 3,
                text: "hello world".into(),
                embedding: vec![0.25, -1.5, 3.0],
            }],
        };

        // Pid-unique name so two concurrent test runs (second checkout, CI)
        // can't race on the same file in the shared temp dir.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "semsearch-test-roundtrip-{}.postcard",
            std::process::id()
        ));
        index.save(&path).unwrap();
        let loaded = Index::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.chunks.len(), 1);
        let (a, b) = (&index.chunks[0], &loaded.chunks[0]);
        assert_eq!(a.path, b.path);
        assert_eq!(a.index, b.index);
        assert_eq!(a.text, b.text);
        assert_eq!(a.embedding, b.embedding);
    }
}
