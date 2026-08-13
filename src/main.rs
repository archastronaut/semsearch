mod chunk;
mod embed;
mod index;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};

use crate::embed::Embedder;
use crate::index::{Index, SearchHit};

/// Cargo-style colored help: bold green section headers, cyan literals.
const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Cyan.on_default())
    .error(AnsiColor::Red.on_default().effects(Effects::BOLD))
    .valid(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .invalid(AnsiColor::Yellow.on_default().effects(Effects::BOLD));

#[derive(Parser)]
#[command(
    name = "semsearch",
    version,
    styles = STYLES,
    about = "Find notes by meaning, not keywords — fully offline",
    long_about = "Semantic search over local .md/.txt notes.\n\n\
        Text is embedded with nomic-embed-text-v1.5 running locally via \
        fastembed — nothing leaves your machine. Results are ranked by \
        meaning, so a query like \"how does memory get cleaned up?\" finds \
        a note about ownership and drops even with zero shared keywords.",
    after_help = "Examples:\n  \
        semsearch ask notes.md \"what did I write about caching?\"\n  \
        semsearch ask ~/notes \"that idea about embeddings\" -k 3\n  \
        semsearch index ~/notes\n  \
        semsearch search \"rust ownership\""
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Ask a question against a file or folder of notes, in one shot
    ///
    /// Points at a .md/.txt file (or a folder of them) and shows the
    /// passages most relevant to the question — chunks and embeds in a
    /// single run, no saved index involved.
    #[command(after_help = "Examples:\n  \
        semsearch ask notes.md \"what did I write about caching?\"\n  \
        semsearch ask ~/Documents/notes \"the bug with timezones\" -k 3")]
    Ask {
        /// A .md/.txt file, or a directory containing them
        path: PathBuf,

        /// The question to ask
        question: String,

        /// How many matching passages to show
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
    },

    /// Embed a folder of notes into a saved index
    ///
    /// Scans a directory for .md/.txt files, embeds every chunk, and saves
    /// the index to disk. Run once, and again whenever your notes change.
    #[command(after_help = "Examples:\n  \
        semsearch index ~/notes\n  \
        semsearch index ~/notes --out ~/notes.postcard")]
    Index {
        /// Directory to scan
        #[arg(default_value = ".")]
        dir: PathBuf,

        /// Where to write the index
        #[arg(short, long, default_value = "index.postcard")]
        out: PathBuf,
    },

    /// Search a previously built index by meaning, not exact wording
    #[command(after_help = "Examples:\n  \
        semsearch search \"rust ownership\"\n  \
        semsearch search \"that caching idea\" --index ~/notes.postcard -k 10")]
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
            let chunks = index::collect_chunks(&dir)?;

            // Check before constructing the Embedder: loading the model
            // (or downloading it, on a fresh machine) just to report an
            // empty directory would be a rude surprise.
            if chunks.is_empty() {
                println!("Nothing to save — no .md/.txt files found.");
                if out.exists() {
                    println!("Note: existing {} left untouched.", out.display());
                }
                return Ok(());
            }

            println!("Embedding {} chunk(s)...", chunks.len());
            let mut embedder = Embedder::new()?;
            let index = Index::from_chunks(chunks, &mut embedder)?;

            index.save(&out)?;
            println!(
                "Indexed {} chunk(s) -> {}",
                index.chunks.len(),
                out.display()
            );
        }

        Command::Ask {
            path,
            question,
            top_k,
        } => {
            anyhow::ensure!(!question.trim().is_empty(), "question is empty");

            // Resolve symlinks up front: WalkDir yields a symlink-to-file
            // root as a symlink entry, which discover_files' is_file filter
            // would drop — making a perfectly good `ask link.md` claim "no
            // content found". Canonicalizing also doubles as the existence
            // check.
            let path = path
                .canonicalize()
                .with_context(|| format!("no such file or directory: {}", path.display()))?;

            if path.is_file() {
                // For a directory, non-matching files are silently skipped;
                // but a single file the user named explicitly deserves a
                // clear error rather than a puzzling "no content found".
                anyhow::ensure!(
                    chunk::is_supported(&path),
                    "only .md and .txt files are supported: {}",
                    path.display()
                );
            }

            let chunks = index::collect_chunks(&path)?;
            if chunks.is_empty() {
                println!("No .md/.txt content found at {}", path.display());
                return Ok(());
            }

            println!("Embedding {} chunk(s)...", chunks.len());
            let mut embedder = Embedder::new()?;
            let index = Index::from_chunks(chunks, &mut embedder)?;

            let query_embedding = embedder.embed_query(&question)?;
            print_hits(&index.search(&query_embedding, top_k));
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

            let mut embedder = Embedder::new()?;
            let query_embedding = embedder.embed_query(&query)?;
            print_hits(&index.search(&query_embedding, top_k));
        }
    }

    Ok(())
}

fn print_hits(hits: &[SearchHit<'_>]) {
    for (rank, hit) in hits.iter().enumerate() {
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
