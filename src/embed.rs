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
    /// Loads nomic-embed-text-v1.5. First run downloads the ONNX weights to
    /// fastembed's local cache (~/.cache/fastembed by default); later runs
    /// are fully offline.
    pub fn new() -> Result<Self> {
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::NomicEmbedTextV15)
                .with_show_download_progress(true),
        )?;
        Ok(Self { model })
    }

    /// Embed document/passage chunks for indexing.
    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{DOCUMENT_PREFIX}{t}"))
            .collect();
        self.model.embed(prefixed, None)
    }

    /// Embed a search query. Uses a different prefix than documents, since
    /// nomic-embed-text is trained for asymmetric query <-> passage retrieval.
    /// Unused until milestone 5 (search) wires it up.
    #[allow(dead_code)]
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{QUERY_PREFIX}{text}");
        let mut out = self.model.embed(vec![prefixed], None)?;
        Ok(out.remove(0))
    }
}
