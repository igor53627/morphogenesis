use super::*;
use std::fs;

#[test]
fn resolve_config_prefers_cli_then_env_then_file() {
    let mut cli = CliArgs::default_for_tests();
    cli.bind_addr = Some("127.0.0.1:4100".to_string());

    let mut env = EnvConfig::default_for_tests();
    env.bind_addr = Some("127.0.0.1:4200".to_string());

    let file = FileConfig {
        bind_addr: Some("127.0.0.1:4300".to_string()),
        ..FileConfig::default()
    };

    cli.environment = Some("dev".to_string());
    cli.allow_synthetic_matrix = true;
    cli.matrix_size_bytes = Some(4096);

    let resolved = RuntimeConfig::resolve(cli, env, Some(file)).expect("config should resolve");
    assert_eq!(resolved.bind_addr, "127.0.0.1:4100".parse().unwrap());
    assert!(!resolved.bind_addr_is_default);
}

#[test]
fn resolve_config_defaults_to_loopback_bind_addr() {
    let mut cli = CliArgs::default_for_tests();
    cli.environment = Some("dev".to_string());
    cli.allow_synthetic_matrix = true;
    cli.matrix_size_bytes = Some(4096);

    let resolved = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
        .expect("config should resolve");
    assert_eq!(resolved.bind_addr, "127.0.0.1:3000".parse().unwrap());
    assert!(resolved.bind_addr_is_default);
}

#[cfg(feature = "cuda")]
#[test]
fn resolve_config_rejects_row_size_larger_than_gpu_page_size() {
    use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

    let mut cli = CliArgs::default_for_tests();
    cli.environment = Some("dev".to_string());
    cli.allow_synthetic_matrix = true;
    cli.row_size_bytes = Some(PAGE_SIZE_BYTES + 1);
    cli.chunk_size_bytes = Some((PAGE_SIZE_BYTES + 1) * 2);
    cli.matrix_size_bytes = Some((PAGE_SIZE_BYTES + 1) * 8);

    let err = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
        .expect_err("row size larger than PAGE_SIZE_BYTES should be rejected");
    assert!(err.to_string().contains("PAGE_SIZE_BYTES"));
}

#[cfg(feature = "cuda")]
#[test]
fn resolve_config_allows_row_size_equal_to_gpu_page_size() {
    use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

    let mut cli = CliArgs::default_for_tests();
    cli.environment = Some("dev".to_string());
    cli.allow_synthetic_matrix = true;
    cli.row_size_bytes = Some(PAGE_SIZE_BYTES);
    cli.chunk_size_bytes = Some(PAGE_SIZE_BYTES * 2);
    cli.matrix_size_bytes = Some(PAGE_SIZE_BYTES * 8);

    RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
        .expect("row_size_bytes == PAGE_SIZE_BYTES should be accepted when page PIR is enabled");
}

#[cfg(feature = "cuda")]
#[test]
fn resolve_config_allows_large_row_size_when_page_pir_disabled() {
    use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

    let mut cli = CliArgs::default_for_tests();
    cli.environment = Some("dev".to_string());
    cli.allow_synthetic_matrix = true;
    cli.disable_page_pir = true;
    cli.row_size_bytes = Some(PAGE_SIZE_BYTES + 1);
    cli.chunk_size_bytes = Some((PAGE_SIZE_BYTES + 1) * 2);
    cli.matrix_size_bytes = Some((PAGE_SIZE_BYTES + 1) * 8);

    let resolved = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
        .expect("row size larger than PAGE_SIZE_BYTES should be allowed when page PIR is disabled");
    assert!(
        resolved.page_config.is_none(),
        "page PIR should be disabled in resolved runtime config"
    );
}

#[test]
fn parse_admin_mtls_subject_header_rejects_invalid_name() {
    let err = parse_admin_mtls_subject_header(Some("bad header".to_string()))
        .expect_err("invalid header name should fail");
    assert!(err
        .to_string()
        .contains("MORPHOGEN_ADMIN_MTLS_SUBJECT_HEADER"));
}

#[test]
fn validate_admin_mtls_proxy_trust_rejects_allowlist_without_opt_in() {
    let subjects = vec!["spiffe://morphogenesis/control-plane".to_string()];
    let err = validate_admin_mtls_proxy_trust(&subjects, false)
        .expect_err("mTLS allowlist without trusted-proxy opt-in should fail");
    assert!(err
        .to_string()
        .contains("MORPHOGEN_ADMIN_TRUST_PROXY_HEADERS=true"));
}

#[test]
fn validate_admin_mtls_proxy_trust_accepts_safe_combinations() {
    let subjects = vec!["spiffe://morphogenesis/control-plane".to_string()];
    validate_admin_mtls_proxy_trust(&subjects, true).expect("explicit opt-in should pass");
    validate_admin_mtls_proxy_trust(&Vec::new(), false).expect("empty mTLS allowlist should pass");
}

#[test]
fn resolve_config_requires_matrix_source() {
    let cli = CliArgs::default_for_tests();
    let env = EnvConfig::default_for_tests();
    let err =
        RuntimeConfig::resolve(cli, env, None).expect_err("should reject missing matrix source");
    assert!(
        err.to_string().contains("matrix_file"),
        "unexpected error: {err}"
    );
}

#[test]
fn resolve_config_parses_page_prg_keys() {
    let mut cli = CliArgs::default_for_tests();
    cli.allow_synthetic_matrix = true;
    cli.matrix_size_bytes = Some(4096);
    cli.page_prg_key_0 = Some("00112233445566778899aabbccddeeff".to_string());
    cli.page_prg_key_1 = Some("ffeeddccbbaa99887766554433221100".to_string());

    let resolved = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
        .expect("config should resolve");
    let page_cfg = resolved.page_config.expect("page config should be enabled");

    assert_eq!(
        page_cfg.prg_keys[0],
        [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ]
    );
    assert_eq!(
        page_cfg.prg_keys[1],
        [
            0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
            0x11, 0x00,
        ]
    );
}

#[test]
fn resolve_config_rejects_bad_prg_key_len() {
    let mut cli = CliArgs::default_for_tests();
    cli.allow_synthetic_matrix = true;
    cli.matrix_size_bytes = Some(4096);
    cli.page_prg_key_0 = Some("1234".to_string());

    let err = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
        .expect_err("config should reject short key");
    assert!(err.to_string().contains("page_prg_key_0"));
}

#[test]
fn resolve_config_prod_requires_explicit_page_prg_keys() {
    let mut cli = CliArgs::default_for_tests();
    cli.environment = Some("prod".to_string());
    cli.allow_synthetic_matrix = true;
    cli.matrix_size_bytes = Some(4096);

    let err = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
        .expect_err("prod should require page PRG keys");
    assert!(err.to_string().contains("page_prg_key_0"));
}

#[test]
fn resolve_config_prod_rejects_zero_page_prg_keys() {
    let mut cli = CliArgs::default_for_tests();
    cli.environment = Some("prod".to_string());
    cli.allow_synthetic_matrix = true;
    cli.matrix_size_bytes = Some(4096);
    cli.page_prg_key_0 = Some("00000000000000000000000000000000".to_string());
    cli.page_prg_key_1 = Some("00000000000000000000000000000000".to_string());

    let err = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
        .expect_err("prod should reject zero PRG keys");
    assert!(err.to_string().contains("must be non-zero"));
}

#[test]
fn cli_flag_overrides_support_explicit_enable_and_disable() {
    let mut cli = CliArgs::default_for_tests();
    assert_eq!(cli.allow_synthetic_matrix_override(), None);
    assert_eq!(cli.disable_page_pir_override(), None);

    cli.allow_synthetic_matrix = true;
    assert_eq!(cli.allow_synthetic_matrix_override(), Some(true));

    cli = CliArgs::default_for_tests();
    cli.no_allow_synthetic_matrix = true;
    assert_eq!(cli.allow_synthetic_matrix_override(), Some(false));

    cli = CliArgs::default_for_tests();
    cli.disable_page_pir = true;
    assert_eq!(cli.disable_page_pir_override(), Some(true));

    cli = CliArgs::default_for_tests();
    cli.enable_page_pir = true;
    assert_eq!(cli.disable_page_pir_override(), Some(false));
}

#[cfg(feature = "cuda")]
#[test]
fn cli_gpu_preload_override_supports_explicit_enable_and_disable() {
    let mut cli = CliArgs::default_for_tests();
    assert_eq!(cli.gpu_preload_override(), None);

    cli.gpu_preload = true;
    assert_eq!(cli.gpu_preload_override(), Some(true));

    cli = CliArgs::default_for_tests();
    cli.no_gpu_preload = true;
    assert_eq!(cli.gpu_preload_override(), Some(false));
}

#[test]
fn cli_rejects_conflicting_allow_synthetic_flags() {
    let result = CliArgs::try_parse_from([
        "server",
        "--allow-synthetic-matrix",
        "--no-allow-synthetic-matrix",
    ]);
    assert!(result.is_err());
}

#[test]
fn cli_rejects_conflicting_page_pir_flags() {
    let result = CliArgs::try_parse_from(["server", "--disable-page-pir", "--enable-page-pir"]);
    assert!(result.is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn cli_rejects_conflicting_gpu_preload_flags() {
    let result = CliArgs::try_parse_from(["server", "--gpu-preload", "--no-gpu-preload"]);
    assert!(result.is_err());
}

#[test]
fn load_matrix_from_file_rejects_unaligned_size() {
    let path = unique_temp_path("morphogen_server_unaligned_matrix.bin");
    fs::write(&path, [0u8; 3]).expect("write temp matrix file");

    let err = match load_matrix_from_file(&path, 2, 2) {
        Ok(_) => panic!("file size must be aligned"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("divisible by row_size_bytes"));

    let _ = fs::remove_file(path);
}

#[test]
fn ctrl_c_failure_logging_policy_matches_expected_thresholds() {
    assert!(should_log_ctrl_c_failure(1));
    assert!(should_log_ctrl_c_failure(2));
    assert!(should_log_ctrl_c_failure(3));
    assert!(!should_log_ctrl_c_failure(4));
    assert!(should_log_ctrl_c_failure(10));
    assert!(should_log_ctrl_c_failure(20));
    assert!(!should_log_ctrl_c_failure(21));
}

#[test]
fn ctrl_c_failure_forced_shutdown_policy_matches_expected_thresholds() {
    assert!(!should_force_shutdown_after_ctrl_c_failures(0));
    assert!(!should_force_shutdown_after_ctrl_c_failures(
        CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN - 1
    ));
    assert!(should_force_shutdown_after_ctrl_c_failures(
        CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN
    ));

    #[cfg(unix)]
    assert!(!ctrl_c_failures_force_shutdown());
    #[cfg(not(unix))]
    assert!(ctrl_c_failures_force_shutdown());
}

#[test]
fn ctrl_c_retry_delay_backoff_is_capped() {
    assert_eq!(
        next_ctrl_c_retry_delay_ms(CTRL_C_INITIAL_RETRY_DELAY_MS),
        CTRL_C_INITIAL_RETRY_DELAY_MS * 2
    );
    assert_eq!(
        next_ctrl_c_retry_delay_ms(CTRL_C_MAX_RETRY_DELAY_MS),
        CTRL_C_MAX_RETRY_DELAY_MS
    );
    assert_eq!(
        next_ctrl_c_retry_delay_ms(CTRL_C_MAX_RETRY_DELAY_MS / 2),
        CTRL_C_MAX_RETRY_DELAY_MS
    );
}

#[test]
fn parse_env_usize_any_prefers_prefixed_and_falls_back_to_legacy() {
    let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_TEST";
    let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_TEST";
    let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_TEST";
    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);

    std::env::set_var(key_legacy, "19");
    let only_legacy =
        parse_env_usize_any(&[key_a, key_b, key_legacy]).expect("legacy parse should succeed");
    assert_eq!(only_legacy, Some(19));

    std::env::set_var(key_b, "23");
    let with_prefixed_b =
        parse_env_usize_any(&[key_a, key_b, key_legacy]).expect("prefixed parse should succeed");
    assert_eq!(with_prefixed_b, Some(23));

    std::env::remove_var(key_b);
    std::env::set_var(key_a, "29");
    let with_prefixed_a =
        parse_env_usize_any(&[key_a, key_b, key_legacy]).expect("prefixed parse should succeed");
    assert_eq!(with_prefixed_a, Some(29));

    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);
}

#[test]
fn parse_env_usize_any_rejects_conflicting_prefixed_values() {
    let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_CONFLICT";
    let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_CONFLICT";
    let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_CONFLICT";
    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);

    std::env::set_var(key_a, "21");
    std::env::set_var(key_b, "34");
    let err = parse_env_usize_any(&[key_a, key_b, key_legacy])
        .expect_err("conflicting prefixed values should fail startup");
    let err_text = err.to_string();
    assert!(err_text.contains("conflicting"));
    assert!(err_text.contains(key_a));
    assert!(err_text.contains(key_b));

    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);
}

#[test]
fn parse_env_usize_any_prefixed_wins_when_legacy_conflicts() {
    let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_LEGACY_CONFLICT";
    let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_LEGACY_CONFLICT";
    let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_LEGACY_CONFLICT";
    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);

    std::env::set_var(key_b, "31");
    std::env::set_var(key_legacy, "19");
    let parsed = parse_env_usize_any(&[key_a, key_b, key_legacy])
        .expect("prefixed value should take precedence over legacy");
    assert_eq!(parsed, Some(31));

    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);
}

#[test]
fn parse_env_usize_any_ignores_malformed_legacy_when_prefixed_present() {
    let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_MALFORMED_LEGACY";
    let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_MALFORMED_LEGACY";
    let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_MALFORMED_LEGACY";
    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);

    std::env::set_var(key_a, "29");
    std::env::set_var(key_legacy, "not-a-number");
    let parsed = parse_env_usize_any(&[key_a, key_b, key_legacy])
        .expect("prefixed value should win when legacy fallback is malformed");
    assert_eq!(parsed, Some(29));

    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);
}

#[test]
fn parse_env_usize_any_rejects_malformed_preferred_even_with_legacy() {
    let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_MALFORMED_PREFERRED";
    let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_MALFORMED_PREFERRED";
    let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_MALFORMED_PREFERRED";
    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);

    std::env::set_var(key_a, "not-a-number");
    std::env::set_var(key_legacy, "19");
    let err = parse_env_usize_any(&[key_a, key_b, key_legacy])
        .expect_err("malformed preferred key should fail startup");
    assert!(err.to_string().contains(key_a));

    std::env::remove_var(key_a);
    std::env::remove_var(key_b);
    std::env::remove_var(key_legacy);
}

fn expected_ctrl_c_retry_durations(retry_count: u32) -> Vec<Duration> {
    let mut delay_ms = CTRL_C_INITIAL_RETRY_DELAY_MS;
    let mut durations = Vec::with_capacity(retry_count as usize);

    for _ in 0..retry_count {
        durations.push(Duration::from_millis(delay_ms));
        delay_ms = next_ctrl_c_retry_delay_ms(delay_ms);
    }

    durations
}

#[tokio::test]
async fn wait_for_ctrl_c_signal_with_retries_until_success() {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let attempts = Arc::new(AtomicUsize::new(0));
    let sleep_durations = Arc::new(Mutex::new(Vec::new()));
    let outcomes = Arc::new(Mutex::new(VecDeque::from(vec![
        Err(std::io::Error::other("transient ctrl-c stream failure 1")),
        Err(std::io::Error::other("transient ctrl-c stream failure 2")),
        Ok(()),
    ])));

    wait_for_ctrl_c_signal_with(
        {
            let attempts = Arc::clone(&attempts);
            let outcomes = Arc::clone(&outcomes);
            move || {
                attempts.fetch_add(1, Ordering::Relaxed);
                let next = outcomes
                    .lock()
                    .expect("lock outcomes")
                    .pop_front()
                    .expect("must provide enough outcomes");
                async move { next }
            }
        },
        {
            let sleep_durations = Arc::clone(&sleep_durations);
            move |duration| {
                let sleep_durations = Arc::clone(&sleep_durations);
                async move {
                    sleep_durations
                        .lock()
                        .expect("lock sleep_durations")
                        .push(duration);
                }
            }
        },
    )
    .await;

    assert_eq!(attempts.load(Ordering::Relaxed), 3);
    assert_eq!(
        sleep_durations
            .lock()
            .expect("lock sleep_durations")
            .clone(),
        expected_ctrl_c_retry_durations(2)
    );
}

#[cfg(not(unix))]
#[tokio::test]
async fn wait_for_ctrl_c_signal_with_forces_shutdown_after_persistent_failures() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let attempts = Arc::new(AtomicUsize::new(0));
    let sleep_durations = Arc::new(Mutex::new(Vec::new()));

    wait_for_ctrl_c_signal_with(
        {
            let attempts = Arc::clone(&attempts);
            move || {
                attempts.fetch_add(1, Ordering::Relaxed);
                async { Err(std::io::Error::other("persistent ctrl-c stream failure")) }
            }
        },
        {
            let sleep_durations = Arc::clone(&sleep_durations);
            move |duration| {
                let sleep_durations = Arc::clone(&sleep_durations);
                async move {
                    sleep_durations
                        .lock()
                        .expect("lock sleep_durations")
                        .push(duration);
                }
            }
        },
    )
    .await;

    assert_eq!(
        attempts.load(Ordering::Relaxed) as u32,
        CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN
    );
    assert_eq!(
        sleep_durations
            .lock()
            .expect("lock sleep_durations")
            .clone(),
        expected_ctrl_c_retry_durations(CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN - 1)
    );
}

#[cfg(unix)]
#[tokio::test]
async fn wait_for_ctrl_c_signal_with_disables_waiter_after_persistent_failures() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let attempts = Arc::new(AtomicUsize::new(0));
    let sleep_durations = Arc::new(Mutex::new(Vec::new()));
    let waiter = wait_for_ctrl_c_signal_with(
        {
            let attempts = Arc::clone(&attempts);
            move || {
                attempts.fetch_add(1, Ordering::Relaxed);
                async { Err(std::io::Error::other("persistent ctrl-c stream failure")) }
            }
        },
        {
            let sleep_durations = Arc::clone(&sleep_durations);
            move |duration| {
                let sleep_durations = Arc::clone(&sleep_durations);
                async move {
                    sleep_durations
                        .lock()
                        .expect("lock sleep_durations")
                        .push(duration);
                }
            }
        },
    );

    let timeout = tokio::time::timeout(Duration::from_millis(25), waiter).await;
    assert!(
        timeout.is_err(),
        "ctrl+c waiter should stay pending on unix"
    );
    assert_eq!(
        attempts.load(Ordering::Relaxed) as u32,
        CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN
    );
    assert_eq!(
        sleep_durations
            .lock()
            .expect("lock sleep_durations")
            .clone(),
        expected_ctrl_c_retry_durations(CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN - 1)
    );
}

#[cfg(unix)]
#[tokio::test]
async fn drive_shutdown_from_futures_sets_channel_on_ctrl_c() {
    let (tx, rx) = watch::channel(false);
    drive_shutdown_from_futures(async {}, std::future::pending::<()>(), tx).await;
    assert!(*rx.borrow());
}

#[cfg(unix)]
#[tokio::test]
async fn drive_shutdown_from_futures_sets_channel_on_sigterm() {
    let (tx, rx) = watch::channel(false);
    drive_shutdown_from_futures(std::future::pending::<()>(), async {}, tx).await;
    assert!(*rx.borrow());
}

#[cfg(not(unix))]
#[tokio::test]
async fn drive_shutdown_from_futures_sets_channel_on_ctrl_c() {
    let (tx, rx) = watch::channel(false);
    drive_shutdown_from_futures(async {}, tx).await;
    assert!(*rx.borrow());
}

fn unique_temp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}", nanos, name))
}
