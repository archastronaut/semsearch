mod chunk;
mod embed;

use std::env;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let files = chunk::discover_files(&dir);
    println!(
        "Found {} .md/.txt file(s) under {}",
        files.len(),
        dir.display()
    );

    let mut all_chunks = Vec::new();
    for path in &files {
        let chunks = chunk::read_and_chunk(path)?;
        println!("  {} -> {} chunk(s)", path.display(), chunks.len());
        all_chunks.extend(chunks);
    }
    println!(
        "\nTotal: {} file(s), {} chunk(s)",
        files.len(),
        all_chunks.len()
    );

    if all_chunks.is_empty() {
        return Ok(());
    }

    println!("\nLoading nomic-embed-text-v1.5 (first run downloads the model)...");
    let mut embedder = embed::Embedder::new()?;

    let texts: Vec<String> = all_chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = embedder.embed_documents(&texts)?;

    println!(
        "Embedded {} chunk(s), dim = {}",
        embeddings.len(),
        embeddings[0].len()
    );
    let preview_len = 5.min(embeddings[0].len());
    println!(
        "First embedding preview: {:?}...",
        &embeddings[0][..preview_len]
    );

    Ok(())
}
