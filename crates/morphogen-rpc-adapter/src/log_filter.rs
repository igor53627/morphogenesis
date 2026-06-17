//! Log filter parsing and matching for `eth_getLogs` / `eth_newFilter`.
//!
//! Extracted from `block_cache.rs` in TASK-54.18. Pure filter logic: parse a
//! JSON-RPC filter object into a [`LogFilter`] (resolving block tags against
//! caller-supplied heights) and match a log entry against a filter.
//!
//! Top-level module (sibling of `block_cache`) so it can be imported directly
//! as `crate::log_filter::*`. The hex block-number parser
//! ([`crate::block_cache::parse_hex_block_number`]) stays in `block_cache.rs`
//! because the poller also uses it.
//!
//! `block_cache.rs` re-exports the public items from here via
//! `pub use crate::log_filter::*;` so external callers' existing
//! `block_cache::LogFilter` / `block_cache::parse_log_filter_object` /
//! `block_cache::log_matches_filter` paths keep resolving.

use serde_json::Value;

pub struct LogFilter {
    pub from_block: u64,
    pub to_block: u64,
    /// If Some, only match logs from these addresses (lowercased hex with 0x prefix).
    pub addresses: Option<Vec<String>>,
    /// Topic filters by position. Each position is a filter.
    pub topics: Vec<TopicFilter>,
}

/// A single topic position filter.
pub enum TopicFilter {
    /// null = match any topic at this position.
    Any,
    /// Match a single exact topic hash (lowercased hex with 0x prefix).
    Exact(String),
    /// Match any one of these topic hashes (lowercased hex with 0x prefix).
    OneOf(Vec<String>),
}

/// Parse a JSON filter object into a LogFilter, resolving block tags.
/// Returns Err with a human-readable message on invalid input.
pub fn parse_log_filter_object(
    filter_obj: &Value,
    latest: u64,
    safe: Option<u64>,
    finalized: Option<u64>,
) -> Result<LogFilter, String> {
    let from_block = match filter_obj.get("fromBlock").and_then(|v| v.as_str()) {
        Some("latest") | Some("pending") | None => latest,
        Some("earliest") => 0,
        Some("safe") => safe.ok_or_else(|| {
            "\"safe\" block tag is unavailable; upstream finality marker could not be resolved"
                .to_string()
        })?,
        Some("finalized") => finalized.ok_or_else(|| {
            "\"finalized\" block tag is unavailable; upstream finality marker could not be resolved"
                .to_string()
        })?,
        Some(hex) => crate::block_cache::parse_hex_block_number(hex)
            .ok_or_else(|| format!("invalid fromBlock: {}", hex))?,
    };

    let to_block = match filter_obj.get("toBlock").and_then(|v| v.as_str()) {
        Some("latest") | Some("pending") | None => latest,
        Some("earliest") => 0,
        Some("safe") => safe.ok_or_else(|| {
            "\"safe\" block tag is unavailable; upstream finality marker could not be resolved"
                .to_string()
        })?,
        Some("finalized") => finalized.ok_or_else(|| {
            "\"finalized\" block tag is unavailable; upstream finality marker could not be resolved"
                .to_string()
        })?,
        Some(hex) => crate::block_cache::parse_hex_block_number(hex)
            .ok_or_else(|| format!("invalid toBlock: {}", hex))?,
    };

    if from_block > to_block {
        return Err(format!(
            "invalid block range: fromBlock ({}) > toBlock ({})",
            from_block, to_block
        ));
    }

    let addresses = match filter_obj.get("address") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(vec![s.to_lowercase()]),
        Some(Value::Array(arr)) => {
            let mut addrs = Vec::with_capacity(arr.len());
            for v in arr {
                match v.as_str() {
                    Some(s) => addrs.push(s.to_lowercase()),
                    None => {
                        return Err(format!(
                            "invalid address in array: expected string, got {}",
                            v
                        ))
                    }
                }
            }
            Some(addrs) // empty list = match nothing
        }
        Some(other) => {
            return Err(format!(
                "invalid address field: expected string or array, got {}",
                other
            ))
        }
    };

    let topics = match filter_obj.get("topics") {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(arr)) => {
            let mut result = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::Null => result.push(TopicFilter::Any),
                    Value::String(s) => result.push(TopicFilter::Exact(s.to_lowercase())),
                    Value::Array(alts) => {
                        let mut options = Vec::with_capacity(alts.len());
                        for v in alts {
                            match v.as_str() {
                                Some(s) => options.push(s.to_lowercase()),
                                None => {
                                    return Err(format!(
                                    "invalid topic in alternatives array: expected string, got {}",
                                    v
                                ))
                                }
                            }
                        }
                        if options.is_empty() {
                            return Err("empty topic alternatives array is not valid".to_string());
                        }
                        result.push(TopicFilter::OneOf(options));
                    }
                    other => {
                        return Err(format!(
                            "invalid topic filter: expected null, string, or array, got {}",
                            other
                        ))
                    }
                }
            }
            result
        }
        Some(other) => {
            return Err(format!(
                "invalid topics field: expected array, got {}",
                other
            ))
        }
    };

    Ok(LogFilter {
        from_block,
        to_block,
        addresses,
        topics,
    })
}

/// Check whether a log entry matches the given filter per EIP-1474 semantics.
pub fn log_matches_filter(log: &Value, filter: &LogFilter) -> bool {
    // Address filter
    if let Some(ref addrs) = filter.addresses {
        let log_addr = log
            .get("address")
            .and_then(|a| a.as_str())
            .unwrap_or("")
            .to_lowercase();
        if !addrs.iter().any(|a| a == &log_addr) {
            return false;
        }
    }

    // Topic filters
    let log_topics = log.get("topics").and_then(|t| t.as_array());
    for (i, topic_filter) in filter.topics.iter().enumerate() {
        match topic_filter {
            TopicFilter::Any => {
                // Wildcard still implies the position exists per EIP-1474
                if log_topics.is_none_or(|t| t.get(i).is_none()) {
                    return false;
                }
            }
            TopicFilter::Exact(expected) => {
                let actual = log_topics
                    .and_then(|t| t.get(i))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                if &actual != expected {
                    return false;
                }
            }
            TopicFilter::OneOf(options) => {
                let actual = log_topics
                    .and_then(|t| t.get(i))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                if !options.iter().any(|o| o == &actual) {
                    return false;
                }
            }
        }
    }

    true
}
