use serde_json::Value;
use std::process::Command;

fn run_effective_config(args: &[&str], envs: &[(&str, &str)]) -> Value {
    let exe = env!("CARGO_BIN_EXE_morphogen-rpc-adapter");
    let mut cmd = Command::new(exe);
    cmd.args(args);
    cmd.env_remove("UPSTREAM_RPC_URL");
    cmd.env_remove("DICT_URL");
    cmd.env_remove("CAS_URL");
    for (key, value) in envs {
        cmd.env(key, value);
    }

    let output = cmd.output().expect("run morphogen-rpc-adapter");
    assert!(
        output.status.success(),
        "expected success, got status {:?}, stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("parse effective config json")
}

#[test]
fn effective_config_uses_env_urls() {
    let cfg = run_effective_config(
        &["--print-effective-config"],
        &[
            ("UPSTREAM_RPC_URL", "https://env-upstream.example"),
            ("DICT_URL", "https://env-dict.example/mainnet_compact.dict"),
            ("CAS_URL", "https://env-cas.example/cas"),
        ],
    );
    assert_eq!(cfg["upstream"], "https://env-upstream.example");
    assert_eq!(
        cfg["dict_url"],
        "https://env-dict.example/mainnet_compact.dict"
    );
    assert_eq!(cfg["cas_url"], "https://env-cas.example/cas");
}

#[test]
fn effective_config_cli_overrides_env_urls() {
    let cfg = run_effective_config(
        &[
            "--print-effective-config",
            "--upstream",
            "https://cli-upstream.example",
            "--dict-url",
            "https://cli-dict.example/mainnet_compact.dict",
            "--cas-url",
            "https://cli-cas.example/cas",
        ],
        &[
            ("UPSTREAM_RPC_URL", "https://env-upstream.example"),
            ("DICT_URL", "https://env-dict.example/mainnet_compact.dict"),
            ("CAS_URL", "https://env-cas.example/cas"),
        ],
    );
    assert_eq!(cfg["upstream"], "https://cli-upstream.example");
    assert_eq!(
        cfg["dict_url"],
        "https://cli-dict.example/mainnet_compact.dict"
    );
    assert_eq!(cfg["cas_url"], "https://cli-cas.example/cas");
}

#[test]
fn effective_config_defaults_without_env_or_flags() {
    let cfg = run_effective_config(&["--print-effective-config"], &[]);
    assert_eq!(cfg["upstream"], "https://ethereum-rpc.publicnode.com");
    assert_eq!(
        cfg["dict_url"],
        "http://localhost:8080/mainnet_compact.dict"
    );
    assert_eq!(cfg["cas_url"], "http://localhost:8080/cas");
}

#[test]
fn effective_config_partial_env_falls_back_to_defaults() {
    let cfg = run_effective_config(
        &["--print-effective-config"],
        &[("DICT_URL", "https://env-dict.example/mainnet_compact.dict")],
    );
    assert_eq!(cfg["upstream"], "https://ethereum-rpc.publicnode.com");
    assert_eq!(
        cfg["dict_url"],
        "https://env-dict.example/mainnet_compact.dict"
    );
    assert_eq!(cfg["cas_url"], "http://localhost:8080/cas");
}

#[test]
fn effective_config_invalid_env_urls_are_redacted_as_invalid() {
    let cfg = run_effective_config(
        &["--print-effective-config"],
        &[
            ("UPSTREAM_RPC_URL", "not-a-valid-url"),
            ("DICT_URL", "also-not-a-url"),
            ("CAS_URL", "still-not-a-url"),
        ],
    );
    assert_eq!(cfg["upstream"], "<invalid-url>");
    assert_eq!(cfg["dict_url"], "<invalid-url>");
    assert_eq!(cfg["cas_url"], "<invalid-url>");
}

#[test]
fn effective_config_redacts_credentials_and_query_values_but_keeps_path_shape() {
    let cfg = run_effective_config(
        &["--print-effective-config"],
        &[
            (
                "UPSTREAM_RPC_URL",
                "https://user:pass@rpc.example/v2/abcd1234abcd1234?token=secret&network=mainnet",
            ),
            (
                "DICT_URL",
                "https://dict.example/mainnet_compact.dict?sig=abc",
            ),
            ("CAS_URL", "https://cas.example/cas?api_key=def"),
        ],
    );
    assert_eq!(
        cfg["upstream"],
        "https://REDACTED:REDACTED@rpc.example/v2/REDACTED?token=REDACTED&network=REDACTED"
    );
    assert_eq!(
        cfg["dict_url"],
        "https://dict.example/mainnet_compact.dict?sig=REDACTED"
    );
    assert_eq!(cfg["cas_url"], "https://cas.example/cas?api_key=REDACTED");
}
