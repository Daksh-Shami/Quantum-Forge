use oq3_parser::{LexedStr, Output, TopEntryPoint};
use std::{fs, path::Path};
use thiserror::Error;

/// Custom error type for handling QASM processing errors.
#[derive(Debug, Error)]
pub enum QasmError {
    #[error("Failed to read QASM file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parsing QASM file failed")]
    ParsingError,
}

/// A structure that holds the QASM file content along with its lexed representation.
/// Instead of `Arc<str>`, we store `content` inline and use a raw pointer (`*const LexedStr`) for lexed.
/// This avoids reference counting and maintains efficiency.
pub struct OwnedLexed {
    pub content: String, // Owned heap-allocated content (inline)
    pub lexed: LexedStr<'static>, // Borrowed, but we manually control its lifetime
}

impl OwnedLexed {
    /// Reads a QASM file from `file_path` and produces an `OwnedLexed`.
    pub fn from_file<P: AsRef<Path>>(file_path: P) -> Result<Self, QasmError> {
        let content = fs::read_to_string(file_path)?;
        
        // SAFETY: We ensure that `lexed` references `content`, which is stored inline.
        let lexed = unsafe { std::mem::transmute::<LexedStr, LexedStr<'static>>(LexedStr::new(&content)) };

        Ok(Self { content, lexed })
    }
}

/// Parses a QASM file into an AST (`Output`).
///
/// # Arguments
/// * `file_path` - The path of the QASM file.
///
/// # Returns
/// * `Result<Output, QasmError>` - The parsed AST output, or an error.
pub fn parse_qasm_file<P: AsRef<Path>>(file_path: P) -> Result<Output, QasmError> {
    let owned = OwnedLexed::from_file(file_path)?;
    let input = owned.lexed.to_input();
    let output = TopEntryPoint::SourceFile.parse(&input);
    Ok(output)
}
