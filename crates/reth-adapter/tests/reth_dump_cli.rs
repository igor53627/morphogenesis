use std::process::Command;

#[test]
fn trustless_flag_exits_with_code_2_and_message() {
    let output = Command::new(env!("CARGO_BIN_EXE_reth_dump"))
        .args(["--db", "/tmp/nonexistent-reth-db", "--trustless"])
        .output()
        .expect("run reth_dump binary");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Error: trustless mode is not implemented"),
        "unexpected stderr: {stderr}"
    );
}
