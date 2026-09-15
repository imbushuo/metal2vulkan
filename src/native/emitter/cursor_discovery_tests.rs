use super::*;

#[test]
fn rejected_cursor_discovery_never_publishes_a_module() {
    let parsed = LlModule::parse("define void @k() {\nentry:\n ret void\n}\n").unwrap();
    let mut emitter = Emitter::new(parsed);
    emitter
        .cursor_calls
        .reject("reader".into(), "%a".into(), "first cursor refusal".into());
    emitter
        .cursor_calls
        .reject("reader".into(), "%b".into(), "second cursor refusal".into());
    let failure = emitter
        .emit_with_sidecar(None, None)
        .expect_err("unpublishable discovery");
    assert_eq!(failure.error, "first cursor refusal");
    assert_eq!(
        failure.rejected.cursor_call_sites,
        HashSet::from([
            ("reader".into(), "%a".into()),
            ("reader".into(), "%b".into())
        ]),
    );
}
