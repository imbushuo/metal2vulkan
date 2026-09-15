/// A Metal `[[function_constant(N)]]` discovered from its AIR `air.fc_initializer` initializer
/// global. The default value is NOT recoverable from the IR — function constants are externally
/// specialized (`MTLFunctionConstantValues` / `specialize_function_constants`) and the initializer
/// global is `externally_initialized ... undef` — so only the index, symbol name, and LLVM type are
/// carried.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionConstant {
    /// The `[[function_constant(N)]]` index / SPIR-V spec-id.
    pub index: u32,
    /// The demangled-free base symbol (the mangled name before the `.MTL_FC_INIT_` marker).
    pub name: String,
    /// The LLVM-IR type of the constant (`i32`, `i1`, `<4 x i32>`, `float`, …).
    pub type_name: String,
    /// Itanium ABI type encoding carried after `MTL_FC_INIT_<index>_` (`j`, `Dv4_j`, `Dh`, …).
    /// Unlike LLVM's signless integer type, this preserves the Metal scalar signedness and lanes
    /// needed to bind an exact `MTLFunctionConstantValues` value.
    pub abi_type_encoding: String,
}

/// Scan LLVM-IR for `[[function_constant]]` initializer globals — module-scope declarations named
/// `@<base>.MTL_FC_INIT_<N>_<suffix>` (Apple's stable FC ABI marker, `section "air.fc_initializer"`).
/// Keys ONLY on that documented marker, never on a shader-specific name. Returns one entry per
/// distinct index, sorted, so a consumer can discover the module's spec-ids without scanning SPIR-V.
pub fn parse_function_constants(ll: &str) -> Vec<FunctionConstant> {
    declared_function_constants(ll, |_| true)
}

/// The module's function constants the caller supplied no value for.
///
/// [`crate::fc_air_specialize::specialize_air_function_constants`] rewrites the `undef` initializer
/// of every index it is handed into that index's literal, and nothing else writes one, so an
/// initializer still reading `undef` is exactly "no value reached this constant". AIR states no
/// default for it either — see the type comment above — so anything downstream that needs a value
/// here is inventing one.
pub(crate) fn function_constants_without_a_supplied_value(ll: &str) -> Vec<FunctionConstant> {
    declared_function_constants(ll, declares_no_value)
}

/// One entry per distinct index whose declaration's initializer satisfies `keep`, sorted by index.
fn declared_function_constants(ll: &str, keep: impl Fn(&str) -> bool) -> Vec<FunctionConstant> {
    let mut out: Vec<FunctionConstant> = Vec::new();
    for line in ll.lines() {
        let t = line.trim_start();
        if !t.starts_with('@') || !t.contains(".MTL_FC_INIT_") {
            continue;
        }
        let Some(eq) = t.find(" = ") else {
            continue;
        };
        let Some((base, marker)) = t[1..eq].trim().split_once(".MTL_FC_INIT_") else {
            continue;
        };
        let digits: String = marker.chars().take_while(|c| c.is_ascii_digit()).collect();
        let Ok(index) = digits.parse::<u32>() else {
            continue;
        };
        let abi_type_encoding = marker
            .strip_prefix(&digits)
            .and_then(|suffix| suffix.strip_prefix('_'))
            .unwrap_or_default()
            .to_string();
        if out.iter().any(|f| f.index == index) {
            continue;
        }
        let declared = fc_global_decl_type_and_initializer(&t[eq + 3..]);
        if !keep(declared.as_ref().map_or("", |(_, initializer)| initializer)) {
            continue;
        }
        out.push(FunctionConstant {
            index,
            name: base.to_string(),
            type_name: declared.map(|(type_name, _)| type_name).unwrap_or_default(),
            abi_type_encoding,
        });
    }
    out.sort_by_key(|f| f.index);
    out
}

/// Whether a declaration's initializer is LLVM's `undef` rather than a value. Matched as a whole
/// token so an identifier that merely starts with those letters cannot be read as one.
fn declares_no_value(initializer: &str) -> bool {
    initializer
        .strip_prefix("undef")
        .is_some_and(|rest| !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_'))
}

/// The declared LLVM type of a `constant`/`global` definition body (the token after the
/// `constant`/`global` keyword) and the initializer that follows it: a balanced `<...>`
/// vector/array or the first scalar token, then the rest.
///
/// One reader for both, so the type scan and the "did the caller supply a value" scan cannot
/// disagree about where the type ends.
fn fc_global_decl_type_and_initializer(decl: &str) -> Option<(String, &str)> {
    let after = decl
        .split(" constant ")
        .nth(1)
        .or_else(|| decl.split(" global ").nth(1))?;
    let s = after.trim_start();
    let (type_name, rest) = if let Some(rest) = s.strip_prefix('<') {
        let end = rest.find('>')?;
        (format!("<{}>", &rest[..end]), &rest[end + 1..])
    } else {
        let token = s.split_whitespace().next()?;
        (token.to_string(), &s[token.len()..])
    };
    Some((type_name, rest.trim_start()))
}
