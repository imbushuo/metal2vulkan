//! Line and operand primitives shared by the pre-emit AIR-text lowerings.
//!
//! Those passes rewrite a call site before [`super::LlModule::parse`] ever sees it, so they work on
//! the raw text and need the same two things: a buffer that joins lines without a trailing blank,
//! and an argument splitter that respects LLVM's nested `<...>` / `(...)` type and constant syntax.

use std::fmt::Write as _;

pub(super) struct LineBuffer {
    pub(super) text: String,
    has_line: bool,
}

impl LineBuffer {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            text: String::with_capacity(capacity),
            has_line: false,
        }
    }

    fn begin_line(&mut self) {
        if self.has_line {
            self.text.push('\n');
        }
        self.has_line = true;
    }

    pub(super) fn push(&mut self, line: &str) {
        self.begin_line();
        self.text.push_str(line);
    }

    pub(super) fn push_fmt(&mut self, line: std::fmt::Arguments<'_>) {
        self.begin_line();
        let _ = self.text.write_fmt(line);
    }
}

/// Split a call argument list on TOP-LEVEL commas, respecting `<...>` / `(...)` / `[...]` / `{...}`
/// nesting — a constant vector operand `<2 x i64> <i64 8, i64 32>` contains an inner comma that must
/// NOT split the argument, and neither does the `addrspace(1)` in a pointer operand's type.
pub(super) fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '<' | '(' | '[' | '{' => {
                depth += 1;
                cur.push(ch);
            }
            '>' | ')' | ']' | '}' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// The last whitespace token of an operand — its value, past the type and any
/// `noundef`/`readonly`/`align N`/`captures(none)` attributes between them.
pub(super) fn last_token(arg: &str) -> &str {
    arg.trim().rsplit(' ').next().unwrap_or("")
}

/// The argument list of a call line: the text between the LAST balanced top-level paren group's
/// delimiters. `find('(')` alone would land inside the callee's `addrspace(3)`-style type parens,
/// and `rfind(')')` alone would run past the `#N` attribute group and `!noalias` metadata that
/// follow the call.
pub(super) fn call_arguments(line: &str) -> Option<&str> {
    let open = line.find('(')?;
    let mut depth = 0i32;
    for (offset, ch) in line[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&line[open + 1..open + offset]);
                }
            }
            _ => {}
        }
    }
    None
}
