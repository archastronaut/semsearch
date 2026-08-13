mod chunk;
mod embed;
mod index;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::index::Index;

#[derive(Parser)]
#[command(name = "semsearch", about = "Local semantic search over your notes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan a directory for .md/.txt files, embed every chunk, and save the
    /// index to disk. Run once, and again whenever your notes change.
    Index {
        /// Directory to scan
        #[arg(default_value = ".")]
        dir: PathBuf,

        /// Where to write the index
        #[arg(short, long, default_value = "index.postcard")]
        out: PathBuf,
    },

    /// Search a previously built index by meaning, not exact wording.
    Search {
        /// The query text
        query: String,

        /// Path to a previously built index
        #[arg(short, long, default_value = "index.postcard")]
        index: PathBuf,

        /// How many results to show
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Index { dir, out } => {
            // discover_files silently treats a missing directory as empty,
            // so a typo'd path would otherwise look like a successful run.
            anyhow::ensure!(dir.is_dir(), "not a directory: {}", dir.display());

            println!("Scanning {}...", dir.display());
            let index = Index::build(&dir)?;

            if index.chunks.is_empty() {
                println!("Nothing to save — no .md/.txt files found.");
                if out.exists() {
                    println!("Note: existing {} left untouched.", out.display());
                }
                return Ok(());
            }

            index.save(&out)?;
            println!(
                "Indexed {} chunk(s) -> {}",
                index.chunks.len(),
                out.display()
            );
        }

        Command::Search {
            query,
            index,
            top_k,
        } => {
            // An empty query would embed just the bare "search_query: "
            // prefix and rank everything against noise — refuse it instead.
            anyhow::ensure!(!query.trim().is_empty(), "query is empty");

            let index = Index::load(&index)?;
            if index.chunks.is_empty() {
                println!("Index is empty — nothing to search.");
                return Ok(());
            }

            let mut embedder = embed::Embedder::new()?;
            let query_embedding = embedder.embed_query(&query)?;

            for (rank, hit) in index.search(&query_embedding, top_k).iter().enumerate() {
                println!(
                    "{}. [{:.3}] {} #{}",
                    rank + 1,
                    hit.score,
                    hit.chunk.path.display(),
                    hit.chunk.index
                );
                for line in hit.chunk.text.lines() {
                    println!("   {line}");
                }
                println!();
            }
        }
    }

    Ok(())
}
