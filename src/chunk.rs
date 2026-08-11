use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use walkdir::WalkDir;

/// A chunk of text extracted from a source file, ready to be embedded later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub path: PathBuf,
    pub index: usize,
    pub text: String,
}

/// Walk `dir` recursively and return paths to all `.md`/`.txt` files.
pub fn discover_files(dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("md") | Some("txt")
            )
        })
        .collect()
}

/// Split raw text into paragraph-level chunks, separated by one or more blank lines.
/// Pure function (no I/O) so it's easy to unit test independent of the filesystem.
pub fn chunk_text(content: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            if !current.trim().is_empty() {
                chunks.push(current.trim().to_string());
            }
            current.clear();
        } else {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}

/// Read a file from disk and split it into paragraph-level chunks.
pub fn read_and_chunk(path: &Path) -> Result<Vec<Chunk>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    Ok(chunk_text(&content)
        .into_iter()
        .enumerate()
        .map(|(index, text)| Chunk {
            path: path.to_path_buf(),
            index,
            text,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_blank_lines() {
        let text = "First para.\nStill first.\n\nSecond para.\n\n\nThird para.";
        let chunks = chunk_text(text);
        assert_eq!(
            chunks,
            vec!["First para.\nStill first.", "Second para.", "Third para.",]
        );
    }

    #[test]
    fn ignores_leading_and_trailing_blank_lines() {
        let text = "\n\nOnly para.\n\n";
        assert_eq!(chunk_text(text), vec!["Only para."]);
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk_text("").is_empty());
    }

    #[test]
    fn handles_crlf_line_endings() {
        let text = "First.\r\n\r\nSecond.";
        assert_eq!(chunk_text(text), vec!["First.", "Second."]);
    }
}
