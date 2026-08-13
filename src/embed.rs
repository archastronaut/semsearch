use std::path::PathBuf;

use anyhow::Result;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

/// nomic-embed-text-v1.5 requires task-specific prefixes to perform well —
/// unlike most sentence-transformer models, these aren't optional cosmetics,
/// they're how the model was fine-tuned. Documents and queries get different
/// prefixes because the model expects asymmetric retrieval.
/// See: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
const DOCUMENT_PREFIX: &str = "search_document: ";
const QUERY_PREFIX: &str = "search_query: ";

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Loads nomic-embed-text-v1.5. First run downloads the ONNX weights;
    /// later runs are fully offline.
    pub fn new() -> Result<Self> {
        let mut options = TextInitOptions::new(EmbeddingModel::NomicEmbedTextV15)
            .with_show_download_progress(true);

        // fastembed's default cache is ./.fastembed_cache relative to the
        // *current directory*, which would re-download the ~500 MB model in
        // every directory the tool is invoked from. Pin it to one shared
        // location so `semsearch` works the same from anywhere. An empty
        // HOME would silently produce a *relative* .cache/semsearch — treat
        // it like unset, and say so instead of degrading quietly.
        match std::env::var_os("HOME").filter(|home| !home.is_empty()) {
            Some(home) => {
                options = options.with_cache_dir(PathBuf::from(home).join(".cache/semsearch"));
            }
            None => eprintln!(
                "warning: HOME is not set; model cache falls back to \
                 .fastembed_cache in the current directory"
            ),
        }

        let model = TextEmbedding::try_new(options)?;
        Ok(Self { model })
    }

    /// Embed document/passage chunks for indexing. Takes `&[&str]` so
    /// callers can borrow into whatever owns the text; the prefix `format!`
    /// below makes the one owned copy fastembed actually needs.
    pub fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{DOCUMENT_PREFIX}{t}"))
            .collect();
        self.model.embed(prefixed, None)
    }

    /// Embed a search query. Uses a different prefix than documents, since
    /// nomic-embed-text is trained for asymmetric query <-> passage retrieval.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{QUERY_PREFIX}{text}");
        let mut out = self.model.embed(vec![prefixed], None)?;
        Ok(out.remove(0))
    }
}
