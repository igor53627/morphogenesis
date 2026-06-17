//! Env-var and CLI string parsing helpers for server startup config.
//!
//! Extracted from `bin/server/main.rs` in TASK-55.2. Pure leaf functions that
//! parse strings/env-vars into typed values, returning [`super::StartupError`]
//! on invalid input. Referenced by `RuntimeConfig`/`EnvConfig` resolution and
//! `validate_server_config`.

use morphogen_server::Environment;

use super::config::StartupError;

pub(super) fn pick3<T>(cli: Option<T>, env: Option<T>, file: Option<T>) -> Option<T> {
    cli.or(env).or(file)
}

pub(super) fn parse_environment(raw: &str) -> Result<Environment, StartupError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "dev" => Ok(Environment::Dev),
        "test" => Ok(Environment::Test),
        "prod" | "production" => Ok(Environment::Prod),
        _ => Err(StartupError::new(format!(
            "invalid environment '{}', expected dev|test|prod",
            raw
        ))),
    }
}

pub(super) fn parse_u64_triplet(raw: &str, field: &str) -> Result<[u64; 3], StartupError> {
    let parts: Vec<&str> = raw.split(',').map(|p| p.trim()).collect();
    if parts.len() != 3 {
        return Err(StartupError::new(format!(
            "{} must contain exactly 3 comma-separated values",
            field
        )));
    }

    let mut out = [0u64; 3];
    for (i, part) in parts.iter().enumerate() {
        out[i] = parse_u64_value(part, field)?;
    }
    Ok(out)
}

pub(super) fn parse_u64_value(raw: &str, field: &str) -> Result<u64, StartupError> {
    if let Some(hex) = raw.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
            .map_err(|e| StartupError::new(format!("invalid {} value '{}': {}", field, raw, e)))
    } else {
        raw.parse::<u64>()
            .map_err(|e| StartupError::new(format!("invalid {} value '{}': {}", field, raw, e)))
    }
}

pub(super) fn parse_fixed_hex<const N: usize>(
    raw: &str,
    field: &str,
) -> Result<[u8; N], StartupError> {
    let hex = raw.strip_prefix("0x").unwrap_or(raw);
    if hex.len() != N * 2 {
        return Err(StartupError::new(format!(
            "{} must be {} bytes ({} hex chars), got {} hex chars",
            field,
            N,
            N * 2,
            hex.len()
        )));
    }

    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(StartupError::new(format!(
            "{} contains non-hex characters",
            field
        )));
    }

    let mut out = [0u8; N];
    hex::decode_to_slice(hex, &mut out)
        .map_err(|e| StartupError::new(format!("failed to decode {} as hex: {}", field, e)))?;
    Ok(out)
}

pub(super) fn env_var(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

pub(super) fn parse_env_usize(key: &str) -> Result<Option<usize>, StartupError> {
    let Some(raw) = env_var(key) else {
        return Ok(None);
    };
    raw.parse::<usize>()
        .map(Some)
        .map_err(|e| StartupError::new(format!("invalid {} value '{}': {}", key, raw, e)))
}

pub(super) fn parse_env_usize_any(keys: &[&str]) -> Result<Option<usize>, StartupError> {
    let Some((legacy_key, preferred_keys)) = keys.split_last() else {
        return Ok(None);
    };

    if preferred_keys.is_empty() {
        return parse_env_usize(legacy_key);
    }

    let mut selected: Option<(&str, usize)> = None;
    for &key in preferred_keys {
        let Some(raw) = env_var(key) else {
            continue;
        };

        let value = raw
            .parse::<usize>()
            .map_err(|e| StartupError::new(format!("invalid {} value '{}': {}", key, raw, e)))?;

        if let Some((selected_key, selected_value)) = selected {
            if value != selected_value {
                return Err(StartupError::new(format!(
                    "conflicting {} ({}) and {} ({}) values; set only one or align them",
                    selected_key, selected_value, key, value
                )));
            }
        } else {
            selected = Some((key, value));
        }
    }

    if let Some((_, value)) = selected {
        return Ok(Some(value));
    }

    parse_env_usize(legacy_key)
}

pub(super) fn parse_env_u64(key: &str) -> Result<Option<u64>, StartupError> {
    let Some(raw) = env_var(key) else {
        return Ok(None);
    };
    parse_u64_value(&raw, key).map(Some)
}

pub(super) fn parse_env_bool(key: &str) -> Result<Option<bool>, StartupError> {
    let Some(raw) = env_var(key) else {
        return Ok(None);
    };

    let value = match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => {
            return Err(StartupError::new(format!(
                "invalid {} value '{}': expected true/false",
                key, raw
            )));
        }
    };

    Ok(Some(value))
}

pub(super) fn validate_admin_mtls_proxy_trust(
    allowed_subjects: &[String],
    trust_proxy_headers: bool,
) -> Result<(), StartupError> {
    if !allowed_subjects.is_empty() && !trust_proxy_headers {
        return Err(StartupError::new(
            "MORPHOGEN_ADMIN_MTLS_ALLOWED_SUBJECTS requires MORPHOGEN_ADMIN_TRUST_PROXY_HEADERS=true",
        ));
    }
    Ok(())
}

pub(super) fn parse_admin_mtls_subject_header(
    raw: Option<String>,
) -> Result<axum::http::HeaderName, StartupError> {
    let raw = raw.unwrap_or_else(|| super::DEFAULT_ADMIN_MTLS_SUBJECT_HEADER.to_string());
    axum::http::HeaderName::from_bytes(raw.as_bytes()).map_err(|_| {
        StartupError::new(format!(
            "invalid MORPHOGEN_ADMIN_MTLS_SUBJECT_HEADER '{}'",
            raw
        ))
    })
}
