mod chunk;

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

    let mut total_chunks = 0;
    for path in &files {
        let chunks = chunk::read_and_chunk(path)?;
        println!("\n{} -> {} chunk(s)", path.display(), chunks.len());
        for c in &chunks {
            let preview: String = c.text.chars().take(80).collect();
            let ellipsis = if c.text.chars().count() > 80 { "..." } else { "" };
            println!("  [{}] {preview}{ellipsis}", c.index);
        }
        total_chunks += chunks.len();
    }

    println!("\nTotal: {} file(s), {total_chunks} chunk(s)", files.len());
    Ok(())
}
