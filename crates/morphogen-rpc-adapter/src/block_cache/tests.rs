use super::*;
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn observe_max_in_flight(max_in_flight: &AtomicUsize, concurrent_now: usize) {
    loop {
        let previous = max_in_flight.load(Ordering::SeqCst);
        if concurrent_now <= previous {
            break;
        }
        if max_in_flight
            .compare_exchange(previous, concurrent_now, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            break;
        }
    }
}

#[test]
fn block_cache_insert_and_get() {
    let mut cache = BlockCache::new();
    let hash = [0xAA; 32];
    let blk_hash = [0x11; 32];
    let tx = serde_json::json!({"hash": "0xaa", "value": "0x1"});
    let receipt = serde_json::json!({"transactionHash": "0xaa", "status": "0x1"});

    cache.insert_block(
        100,
        blk_hash,
        vec![(hash, tx.clone())],
        vec![(hash, receipt.clone())],
    );

    assert_eq!(cache.latest_block(), 100);
    assert_eq!(cache.get_transaction(&hash), Some(&tx));
    assert_eq!(cache.get_receipt(&hash), Some(&receipt));
    assert_eq!(cache.latest_block_hash(), Some((100, blk_hash)));
}

#[test]
fn block_cache_eviction() {
    let mut cache = BlockCache::new();

    // Insert MAX_CACHED_BLOCKS + 1 blocks with both txs and receipts
    for i in 0..=(MAX_CACHED_BLOCKS as u64) {
        let mut hash = [0u8; 32];
        hash[0] = i as u8;
        hash[1] = (i >> 8) as u8;
        let mut blk_hash = [0u8; 32];
        blk_hash[0] = i as u8;
        let tx = serde_json::json!({"block": i});
        let receipt = serde_json::json!({"status": "0x1"});
        cache.insert_block(i, blk_hash, vec![(hash, tx)], vec![(hash, receipt)]);
    }

    // Oldest block's tx and receipt should be evicted
    let old_hash = [0u8; 32]; // block 0's hash
    assert_eq!(cache.get_transaction(&old_hash), None);
    assert_eq!(cache.get_receipt(&old_hash), None);

    // Latest block's tx and receipt should still be there
    let mut new_hash = [0u8; 32];
    new_hash[0] = MAX_CACHED_BLOCKS as u8;
    assert!(cache.get_transaction(&new_hash).is_some());
    assert!(cache.get_receipt(&new_hash).is_some());

    assert_eq!(cache.cached_blocks.len(), MAX_CACHED_BLOCKS);
}

#[test]
fn block_cache_invalidate_reorg() {
    let mut cache = BlockCache::new();
    let hash_a = [0xAA; 32];
    let hash_b = [0xBB; 32];
    let hash_c = [0xCC; 32];

    cache.insert_block(
        100,
        [0x01; 32],
        vec![(hash_a, serde_json::json!({}))],
        vec![(hash_a, serde_json::json!({}))],
    );
    cache.insert_block(
        101,
        [0x02; 32],
        vec![(hash_b, serde_json::json!({}))],
        vec![(hash_b, serde_json::json!({}))],
    );
    cache.insert_block(
        102,
        [0x03; 32],
        vec![(hash_c, serde_json::json!({}))],
        vec![(hash_c, serde_json::json!({}))],
    );

    assert_eq!(cache.latest_block(), 102);

    // Simulate reorg at block 101 — invalidate 101 and 102
    cache.invalidate_from(101);

    assert_eq!(cache.latest_block(), 100);
    assert!(cache.get_transaction(&hash_a).is_some());
    assert_eq!(cache.get_transaction(&hash_b), None);
    assert_eq!(cache.get_transaction(&hash_c), None);
    assert_eq!(cache.get_receipt(&hash_b), None);
    assert_eq!(cache.get_receipt(&hash_c), None);
}

#[test]
fn block_cache_miss() {
    let cache = BlockCache::new();
    let hash = [0xFF; 32];
    assert_eq!(cache.get_transaction(&hash), None);
    assert_eq!(cache.get_receipt(&hash), None);
}

#[test]
fn parse_tx_hash_valid() {
    let hash_str = "0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060";
    let hash = parse_tx_hash(hash_str).unwrap();
    assert_eq!(hash[0], 0x5c);
    assert_eq!(hash[31], 0x60);
}

#[test]
fn parse_tx_hash_invalid() {
    assert!(parse_tx_hash("0xshort").is_none());
    assert!(parse_tx_hash("not_hex").is_none());
    // Correct length (64 chars) but invalid hex characters
    let bad_hex = format!("0x{}", "g".repeat(64));
    assert_eq!(bad_hex.strip_prefix("0x").unwrap().len(), 64);
    assert!(parse_tx_hash(&bad_hex).is_none());
}

#[test]
fn parse_hex_block_number_valid() {
    assert_eq!(parse_hex_block_number("0x1"), Some(1));
    assert_eq!(parse_hex_block_number("0xff"), Some(255));
    assert_eq!(parse_hex_block_number("0x13b6340"), Some(0x13b6340));
}

#[test]
fn parse_hex_block_number_invalid() {
    assert_eq!(parse_hex_block_number("0xZZZZ"), None);
}

// --- Log filter tests ---

fn make_log(address: &str, topics: &[&str]) -> Value {
    serde_json::json!({
        "address": address,
        "topics": topics,
        "data": "0x",
        "blockNumber": "0x64",
        "logIndex": "0x0"
    })
}

fn make_receipt_with_logs(tx_hash: &str, logs: Vec<Value>) -> Value {
    serde_json::json!({
        "transactionHash": tx_hash,
        "status": "0x1",
        "logs": logs
    })
}

#[test]
fn log_filter_address_single() {
    let log = make_log("0xabc123", &["0xtopic1"]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: Some(vec!["0xabc123".to_string()]),
        topics: vec![],
    };
    assert!(log_matches_filter(&log, &filter));

    let filter_miss = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: Some(vec!["0xdef456".to_string()]),
        topics: vec![],
    };
    assert!(!log_matches_filter(&log, &filter_miss));
}

#[test]
fn log_filter_address_list() {
    let log = make_log("0xabc123", &[]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: Some(vec!["0xdef456".to_string(), "0xabc123".to_string()]),
        topics: vec![],
    };
    assert!(log_matches_filter(&log, &filter));
}

#[test]
fn log_filter_address_wildcard() {
    let log = make_log("0xabc123", &[]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![],
    };
    assert!(log_matches_filter(&log, &filter));
}

#[test]
fn log_filter_address_case_insensitive() {
    let log = make_log("0xAbC123", &[]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: Some(vec!["0xabc123".to_string()]),
        topics: vec![],
    };
    assert!(log_matches_filter(&log, &filter));
}

#[test]
fn log_filter_topic_exact() {
    let log = make_log("0xabc", &["0xdead", "0xbeef"]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![TopicFilter::Exact("0xdead".to_string())],
    };
    assert!(log_matches_filter(&log, &filter));

    let filter_miss = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![TopicFilter::Exact("0xbeef".to_string())],
    };
    assert!(!log_matches_filter(&log, &filter_miss));
}

#[test]
fn log_filter_topic_any() {
    let log = make_log("0xabc", &["0xdead", "0xbeef"]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![TopicFilter::Any, TopicFilter::Exact("0xbeef".to_string())],
    };
    assert!(log_matches_filter(&log, &filter));
}

#[test]
fn log_filter_topic_one_of() {
    let log = make_log("0xabc", &["0xdead"]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![TopicFilter::OneOf(vec![
            "0xcafe".to_string(),
            "0xdead".to_string(),
        ])],
    };
    assert!(log_matches_filter(&log, &filter));

    let filter_miss = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![TopicFilter::OneOf(vec![
            "0xcafe".to_string(),
            "0xbabe".to_string(),
        ])],
    };
    assert!(!log_matches_filter(&log, &filter_miss));
}

#[test]
fn log_filter_topic_position_out_of_range() {
    // Log has only 1 topic, filter asks for topic at position 1
    let log = make_log("0xabc", &["0xdead"]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![TopicFilter::Any, TopicFilter::Exact("0xbeef".to_string())],
    };
    assert!(!log_matches_filter(&log, &filter));
}

#[test]
fn log_filter_mixed_address_and_topics() {
    let log = make_log("0xabc", &["0xdead", "0xbeef"]);
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: Some(vec!["0xabc".to_string()]),
        topics: vec![
            TopicFilter::Exact("0xdead".to_string()),
            TopicFilter::Exact("0xbeef".to_string()),
        ],
    };
    assert!(log_matches_filter(&log, &filter));

    // Wrong address
    let filter_wrong_addr = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: Some(vec!["0xdef".to_string()]),
        topics: vec![TopicFilter::Exact("0xdead".to_string())],
    };
    assert!(!log_matches_filter(&log, &filter_wrong_addr));
}

#[test]
fn get_logs_from_cache() {
    let mut cache = BlockCache::new();

    let log1 = make_log("0xaaa", &["0xevent1"]);
    let log2 = make_log("0xbbb", &["0xevent2"]);
    let log3 = make_log("0xaaa", &["0xevent3"]);

    let r1 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log1.clone()],
    );
    let r2 = make_receipt_with_logs(
        "0xbb00000000000000000000000000000000000000000000000000000000000000",
        vec![log2.clone(), log3.clone()],
    );

    let h1 = [0xAA; 32];
    let h2 = [0xBB; 32];
    cache.insert_block(100, [0x01; 32], vec![], vec![(h1, r1)]);
    cache.insert_block(101, [0x02; 32], vec![], vec![(h2, r2)]);

    // All logs (no filter)
    let filter_all = LogFilter {
        from_block: 100,
        to_block: 101,
        addresses: None,
        topics: vec![],
    };
    assert_eq!(cache.get_logs(&filter_all).len(), 3);

    // Filter by address
    let filter_aaa = LogFilter {
        from_block: 100,
        to_block: 101,
        addresses: Some(vec!["0xaaa".to_string()]),
        topics: vec![],
    };
    let results = cache.get_logs(&filter_aaa);
    assert_eq!(results.len(), 2);

    // Filter by single block
    let filter_block = LogFilter {
        from_block: 101,
        to_block: 101,
        addresses: None,
        topics: vec![],
    };
    assert_eq!(cache.get_logs(&filter_block).len(), 2);
}

#[test]
fn has_block_range_checks() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);
    cache.insert_block(101, [0x02; 32], vec![], vec![]);
    cache.insert_block(102, [0x03; 32], vec![], vec![]);

    assert!(cache.has_block_range(100, 102));
    assert!(cache.has_block_range(101, 101));
    assert!(!cache.has_block_range(99, 102));
    assert!(!cache.has_block_range(100, 103));
}

#[test]
fn get_logs_empty_and_no_match() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);

    let filter = LogFilter {
        from_block: 100,
        to_block: 100,
        addresses: Some(vec!["0xnothere".to_string()]),
        topics: vec![],
    };
    assert!(cache.get_logs(&filter).is_empty());
}

#[test]
fn logs_evicted_with_block() {
    let mut cache = BlockCache::new();
    for i in 0..=(MAX_CACHED_BLOCKS as u64) {
        let log = make_log(&format!("0x{:x}", i), &[]);
        let receipt = make_receipt_with_logs(&format!("0x{:0>64x}", i), vec![log]);
        let mut hash = [0u8; 32];
        hash[0] = i as u8;
        cache.insert_block(i, [i as u8; 32], vec![], vec![(hash, receipt)]);
    }

    // Block 0 should be evicted
    assert!(!cache.logs.contains_key(&0));
    // Latest block should still have logs
    assert!(cache.logs.contains_key(&(MAX_CACHED_BLOCKS as u64)));
}

#[test]
fn logs_cleaned_on_invalidate() {
    let mut cache = BlockCache::new();
    let log = make_log("0xabc", &[]);
    let receipt = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log],
    );
    cache.insert_block(100, [0x01; 32], vec![], vec![([0xAA; 32], receipt)]);

    assert!(cache.logs.contains_key(&100));
    cache.invalidate_from(100);
    assert!(!cache.logs.contains_key(&100));
}

// --- Filter tests ---

#[test]
fn filter_log_get_changes() {
    let mut cache = BlockCache::new();

    // Insert initial block
    let log1 = make_log("0xaaa", &["0xevent1"]);
    let r1 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log1],
    );
    cache.insert_block(100, [0x01; 32], vec![], vec![([0xAA; 32], r1)]);

    // Create log filter — cursor starts at from_block - 1 (99)
    let filter = LogFilter {
        from_block: 100,
        to_block: u64::MAX,
        addresses: Some(vec!["0xaaa".to_string()]),
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    // First poll returns block 100's log (from_block is inclusive)
    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["address"].as_str().unwrap(), "0xaaa");

    // Polling again with no new blocks returns empty
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());

    // Insert a new block with a matching log
    let log2 = make_log("0xaaa", &["0xevent2"]);
    let r2 = make_receipt_with_logs(
        "0xbb00000000000000000000000000000000000000000000000000000000000000",
        vec![log2],
    );
    cache.insert_block(101, [0x02; 32], vec![], vec![([0xBB; 32], r2)]);

    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["address"].as_str().unwrap(), "0xaaa");

    // Polling again with no new blocks returns empty
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());
}

#[test]
fn filter_log_get_changes_includes_block_zero_for_from_block_zero() {
    let mut cache = BlockCache::new();

    let log0 = make_log("0xaaa", &["0xevent0"]);
    let r0 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log0],
    );
    cache.insert_block(0, [0x01; 32], vec![], vec![([0xAA; 32], r0)]);

    let filter = LogFilter {
        from_block: 0,
        to_block: u64::MAX,
        addresses: Some(vec!["0xaaa".to_string()]),
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["address"].as_str().unwrap(), "0xaaa");

    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());
}

#[test]
fn filter_log_get_changes_includes_block_zero_after_empty_initial_poll() {
    let mut cache = BlockCache::new();

    let filter = LogFilter {
        from_block: 0,
        to_block: u64::MAX,
        addresses: Some(vec!["0xaaa".to_string()]),
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    // First poll happens before block 0 is available.
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());

    let log0 = make_log("0xaaa", &["0xevent0"]);
    let r0 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log0],
    );
    cache.insert_block(0, [0x01; 32], vec![], vec![([0xAA; 32], r0)]);

    // Polling again should still return block 0.
    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["address"].as_str().unwrap(), "0xaaa");
}

#[test]
fn filter_log_get_changes_includes_block_zero_after_latest_advances_before_cache_warmup() {
    let mut cache = BlockCache::new();

    let filter = LogFilter {
        from_block: 0,
        to_block: u64::MAX,
        addresses: Some(vec!["0xaaa".to_string()]),
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    // Simulate startup state where head height is known before block 0 is cached.
    cache.latest_block = 5;

    // Initial poll should not consume the block-0 sentinel cursor.
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());

    let log0 = make_log("0xaaa", &["0xevent0"]);
    let r0 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log0],
    );
    cache.insert_block(0, [0x01; 32], vec![], vec![([0xAA; 32], r0)]);

    // Polling again should still return block 0.
    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["address"].as_str().unwrap(), "0xaaa");
}

#[test]
fn filter_log_get_changes_returns_higher_blocks_while_block_zero_pending() {
    let mut cache = BlockCache::new();

    let filter = LogFilter {
        from_block: 0,
        to_block: u64::MAX,
        addresses: Some(vec!["0xaaa".to_string()]),
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    let log1 = make_log("0xaaa", &["0xevent1"]);
    let r1 = make_receipt_with_logs(
        "0xbb00000000000000000000000000000000000000000000000000000000000000",
        vec![log1],
    );
    cache.insert_block(1, [0x02; 32], vec![], vec![([0xBB; 32], r1)]);

    // Polling should still return block 1 logs even though block 0 is absent.
    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["topics"][0].as_str().unwrap(), "0xevent1");

    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());

    let log0 = make_log("0xaaa", &["0xevent0"]);
    let r0 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log0],
    );
    cache.insert_block(0, [0x01; 32], vec![], vec![([0xAA; 32], r0)]);

    // Once block 0 arrives late, it should still be emitted exactly once.
    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["topics"][0].as_str().unwrap(), "0xevent0");

    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());
}

#[test]
fn filter_log_get_changes_keeps_block_zero_before_newer_logs_when_returned_together() {
    let mut cache = BlockCache::new();

    let filter = LogFilter {
        from_block: 0,
        to_block: u64::MAX,
        addresses: Some(vec!["0xaaa".to_string()]),
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    let log1 = make_log("0xaaa", &["0xevent1"]);
    let r1 = make_receipt_with_logs(
        "0xbb00000000000000000000000000000000000000000000000000000000000000",
        vec![log1],
    );
    cache.insert_block(1, [0x02; 32], vec![], vec![([0xBB; 32], r1)]);

    // Consume block 1 so the cursor advances while block 0 is still pending.
    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["topics"][0].as_str().unwrap(), "0xevent1");

    let log2 = make_log("0xaaa", &["0xevent2"]);
    let r2 = make_receipt_with_logs(
        "0xcc00000000000000000000000000000000000000000000000000000000000000",
        vec![log2],
    );
    cache.insert_block(2, [0x03; 32], vec![], vec![([0xCC; 32], r2)]);

    let log0 = make_log("0xaaa", &["0xevent0"]);
    let r0 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log0],
    );
    cache.insert_block(0, [0x01; 32], vec![], vec![([0xAA; 32], r0)]);

    // When late block 0 and newer logs are emitted together, keep chronological order.
    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 2);
    assert_eq!(changes[0]["topics"][0].as_str().unwrap(), "0xevent0");
    assert_eq!(changes[1]["topics"][0].as_str().unwrap(), "0xevent2");
}

#[test]
fn filter_block_get_changes() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);

    let id = cache.create_block_filter();

    // No new blocks
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());

    // Add two blocks
    cache.insert_block(101, [0x02; 32], vec![], vec![]);
    cache.insert_block(102, [0x03; 32], vec![], vec![]);

    let changes = cache.get_filter_changes(&id).unwrap();
    assert_eq!(changes.len(), 2);
    // Verify they're hex-encoded block hashes
    assert!(changes[0].as_str().unwrap().starts_with("0x"));
    assert!(changes[1].as_str().unwrap().starts_with("0x"));
}

#[test]
fn filter_pending_tx_returns_empty() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);

    let id = cache.create_pending_tx_filter();
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());

    // Even after new blocks, still empty (no mempool)
    cache.insert_block(101, [0x02; 32], vec![], vec![]);
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());
}

#[test]
fn filter_uninstall() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);

    let id = cache.create_block_filter();
    assert!(cache.uninstall_filter(&id));
    assert!(!cache.uninstall_filter(&id)); // already removed
    assert!(cache.get_filter_changes(&id).is_none()); // not found
}

#[test]
fn filter_expired_cleanup() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);

    let id = cache.create_block_filter();

    // Manually set last_accessed far in the past
    cache.filters.get_mut(&id).unwrap().last_accessed =
        Instant::now() - std::time::Duration::from_secs(FILTER_EXPIRY_SECS + 1);

    // Next filter creation triggers cleanup
    let _id2 = cache.create_block_filter();
    assert!(cache.get_filter_changes(&id).is_none());
}

#[test]
fn filter_get_filter_logs() {
    let mut cache = BlockCache::new();

    let log1 = make_log("0xaaa", &["0xevent1"]);
    let r1 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log1],
    );
    cache.insert_block(100, [0x01; 32], vec![], vec![([0xAA; 32], r1)]);

    let log2 = make_log("0xaaa", &["0xevent2"]);
    let r2 = make_receipt_with_logs(
        "0xbb00000000000000000000000000000000000000000000000000000000000000",
        vec![log2],
    );
    cache.insert_block(101, [0x02; 32], vec![], vec![([0xBB; 32], r2)]);

    // Log filter from block 100
    let filter = LogFilter {
        from_block: 100,
        to_block: u64::MAX,
        addresses: Some(vec!["0xaaa".to_string()]),
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    // get_filter_logs returns all matching logs from from_block to latest
    let result = cache.get_filter_logs(&id).unwrap().unwrap();
    assert_eq!(result.len(), 2);

    // Block filter returns Some(None) — not a log filter
    let block_id = cache.create_block_filter();
    let result = cache.get_filter_logs(&block_id);
    assert!(result.unwrap().is_none());

    // Unknown filter returns None
    assert!(cache.get_filter_logs("0xdeadbeef").is_none());
}

#[test]
fn parse_log_filter_object_basic() {
    let obj = serde_json::json!({
        "fromBlock": "0x64",
        "toBlock": "0x65",
        "address": "0xABC",
        "topics": ["0xDEAD", null, ["0xA", "0xB"]]
    });
    let filter = parse_log_filter_object(&obj, 200, None, None).unwrap();
    assert_eq!(filter.from_block, 0x64);
    assert_eq!(filter.to_block, 0x65);
    assert_eq!(filter.addresses, Some(vec!["0xabc".to_string()]));
    assert_eq!(filter.topics.len(), 3);
}

#[test]
fn parse_log_filter_object_rejects_reversed_range() {
    let obj = serde_json::json!({
        "fromBlock": "0x100",
        "toBlock": "0x50"
    });
    assert!(parse_log_filter_object(&obj, 200, None, None).is_err());
}

#[test]
fn parse_log_filter_object_defaults_to_latest() {
    let obj = serde_json::json!({});
    let filter = parse_log_filter_object(&obj, 500, None, None).unwrap();
    assert_eq!(filter.from_block, 500);
    assert_eq!(filter.to_block, 500);
}

#[test]
fn filter_ids_are_unique() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);

    let id1 = cache.create_block_filter();
    let id2 = cache.create_log_filter(LogFilter {
        from_block: 100,
        to_block: 100,
        addresses: None,
        topics: vec![],
    });
    let id3 = cache.create_pending_tx_filter();

    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);
}

#[test]
fn has_block_range_reversed() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);
    cache.insert_block(101, [0x02; 32], vec![], vec![]);
    assert!(!cache.has_block_range(101, 100));
}

#[test]
fn topic_any_requires_position_exists() {
    // Filter [null] should NOT match a log with empty topics
    let log_no_topics = serde_json::json!({"address": "0xabc", "topics": [], "data": "0x"});
    let filter = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![TopicFilter::Any],
    };
    assert!(!log_matches_filter(&log_no_topics, &filter));

    // Filter [null, null] should NOT match a log with only 1 topic
    let log_one_topic = make_log("0xabc", &["0xdead"]);
    let filter2 = LogFilter {
        from_block: 0,
        to_block: 0,
        addresses: None,
        topics: vec![TopicFilter::Any, TopicFilter::Any],
    };
    assert!(!log_matches_filter(&log_one_topic, &filter2));

    // Filter [null] SHOULD match a log with 1 topic
    assert!(log_matches_filter(&log_one_topic, &filter));
}

#[test]
fn filter_changes_respects_to_block() {
    let mut cache = BlockCache::new();
    cache.insert_block(100, [0x01; 32], vec![], vec![]);
    cache.insert_block(101, [0x02; 32], vec![], vec![]);

    // Create a log filter with to_block=100 when latest is already 101
    let filter = LogFilter {
        from_block: 99,
        to_block: 100,
        addresses: None,
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    // get_filter_changes should return empty (cursor was set to 101, to_block is 100)
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());

    // Adding more blocks shouldn't change anything — to_block already passed
    cache.insert_block(102, [0x03; 32], vec![], vec![]);
    let changes = cache.get_filter_changes(&id).unwrap();
    assert!(changes.is_empty());
}

#[test]
fn filter_logs_respects_to_block() {
    let mut cache = BlockCache::new();
    let log1 = make_log("0xaaa", &[]);
    let log2 = make_log("0xaaa", &[]);
    let r1 = make_receipt_with_logs(
        "0xaa00000000000000000000000000000000000000000000000000000000000000",
        vec![log1],
    );
    let r2 = make_receipt_with_logs(
        "0xbb00000000000000000000000000000000000000000000000000000000000000",
        vec![log2],
    );
    cache.insert_block(100, [0x01; 32], vec![], vec![([0xAA; 32], r1)]);
    cache.insert_block(101, [0x02; 32], vec![], vec![([0xBB; 32], r2)]);

    let filter = LogFilter {
        from_block: 100,
        to_block: 100,
        addresses: None,
        topics: vec![],
    };
    let id = cache.create_log_filter(filter);

    // get_filter_logs should only return logs up to to_block=100
    let result = cache.get_filter_logs(&id).unwrap().unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn parse_filter_rejects_invalid_address_type() {
    let obj = serde_json::json!({"address": 42});
    assert!(parse_log_filter_object(&obj, 100, None, None).is_err());
}

#[test]
fn parse_filter_rejects_invalid_topic_type() {
    let obj = serde_json::json!({"topics": [42]});
    assert!(parse_log_filter_object(&obj, 100, None, None).is_err());
}

#[test]
fn parse_filter_rejects_empty_topic_alternatives() {
    let obj = serde_json::json!({"topics": [[]]});
    assert!(parse_log_filter_object(&obj, 100, None, None).is_err());
}

#[test]
fn empty_address_array_matches_nothing() {
    let obj = serde_json::json!({"address": []});
    let filter = parse_log_filter_object(&obj, 100, None, None).unwrap();
    let log = make_log("0xabc", &[]);
    assert!(!log_matches_filter(&log, &filter));
}

#[test]
fn parse_filter_supports_safe_finalized() {
    let obj = serde_json::json!({"fromBlock": "safe"});
    let filter =
        parse_log_filter_object(&obj, 100, Some(95), Some(90)).expect("safe should be accepted");
    assert_eq!(filter.from_block, 95);

    let obj = serde_json::json!({"fromBlock": "earliest", "toBlock": "finalized"});
    let filter = parse_log_filter_object(&obj, 100, Some(95), Some(90))
        .expect("finalized should be accepted");
    assert_eq!(filter.to_block, 90);
}

#[test]
fn parse_filter_rejects_safe_finalized_when_unresolved() {
    let obj = serde_json::json!({"fromBlock": "safe"});
    assert!(parse_log_filter_object(&obj, 100, None, Some(90)).is_err());

    let obj = serde_json::json!({"toBlock": "finalized"});
    assert!(parse_log_filter_object(&obj, 100, Some(95), None).is_err());
}

#[tokio::test]
async fn fetch_receipts_fallback_is_bounded_and_preserves_order() {
    let mut tx_hashes = Vec::new();
    for i in 0..12u8 {
        let mut hash = [0u8; 32];
        hash[31] = i;
        tx_hashes.push(hash);
    }

    let in_flight_requests = Arc::new(AtomicUsize::new(0));
    let max_in_flight_requests = Arc::new(AtomicUsize::new(0));
    let fetcher: Arc<dyn Fn([u8; 32]) -> ReceiptFetchFuture + Send + Sync> = Arc::new({
        let in_flight_requests = in_flight_requests.clone();
        let max_in_flight_requests = max_in_flight_requests.clone();
        move |hash: [u8; 32]| {
            let in_flight_requests = in_flight_requests.clone();
            let max_in_flight_requests = max_in_flight_requests.clone();
            Box::pin(async move {
                let concurrent_now = in_flight_requests.fetch_add(1, Ordering::SeqCst) + 1;
                observe_max_in_flight(&max_in_flight_requests, concurrent_now);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                in_flight_requests.fetch_sub(1, Ordering::SeqCst);
                Ok(json!({
                    "transactionHash": format!("0x{}", hex::encode(hash)),
                    "status": "0x1",
                    "logs": [],
                }))
            })
        }
    });
    let receipts = fetch_receipts_fallback_bounded(&tx_hashes, fetcher).await;

    assert_eq!(receipts.len(), tx_hashes.len());
    let returned_hashes: Vec<[u8; 32]> = receipts.iter().map(|(hash, _)| *hash).collect();
    assert_eq!(returned_hashes, tx_hashes);

    let observed_max = max_in_flight_requests.load(Ordering::SeqCst);
    assert!(
        observed_max > 1,
        "expected concurrent receipt fallback requests, observed {}",
        observed_max
    );
    assert!(
        observed_max <= RECEIPT_FALLBACK_MAX_IN_FLIGHT,
        "expected at most {} concurrent requests, observed {}",
        RECEIPT_FALLBACK_MAX_IN_FLIGHT,
        observed_max
    );
}

#[tokio::test]
async fn fetch_receipts_fallback_skips_nulls_and_errors_preserving_order() {
    let mut tx_hashes = Vec::new();
    for i in 0..8u8 {
        let mut hash = [0u8; 32];
        hash[31] = i;
        tx_hashes.push(hash);
    }

    let in_flight_requests = Arc::new(AtomicUsize::new(0));
    let max_in_flight_requests = Arc::new(AtomicUsize::new(0));
    let fetcher: Arc<dyn Fn([u8; 32]) -> ReceiptFetchFuture + Send + Sync> = Arc::new({
        let in_flight_requests = in_flight_requests.clone();
        let max_in_flight_requests = max_in_flight_requests.clone();
        move |hash: [u8; 32]| {
            let in_flight_requests = in_flight_requests.clone();
            let max_in_flight_requests = max_in_flight_requests.clone();
            Box::pin(async move {
                let concurrent_now = in_flight_requests.fetch_add(1, Ordering::SeqCst) + 1;
                observe_max_in_flight(&max_in_flight_requests, concurrent_now);

                let delay_ms = 5 + ((hash[31] as u64) % 3) * 10;
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                in_flight_requests.fetch_sub(1, Ordering::SeqCst);

                match hash[31] {
                    1 | 5 => Ok(Value::Null),
                    3 | 6 => Err(format!("simulated upstream error for {}", hash[31])),
                    _ => Ok(json!({
                        "transactionHash": format!("0x{}", hex::encode(hash)),
                        "status": "0x1",
                        "logs": [],
                    })),
                }
            })
        }
    });

    let receipts = fetch_receipts_fallback_bounded(&tx_hashes, fetcher).await;
    let returned_hashes: Vec<[u8; 32]> = receipts.iter().map(|(hash, _)| *hash).collect();
    let expected_hashes: Vec<[u8; 32]> = tx_hashes
        .iter()
        .copied()
        .filter(|hash| !matches!(hash[31], 1 | 3 | 5 | 6))
        .collect();
    assert_eq!(returned_hashes, expected_hashes);

    let observed_max = max_in_flight_requests.load(Ordering::SeqCst);
    assert!(
        observed_max > 1,
        "expected concurrent receipt fallback requests, observed {}",
        observed_max
    );
    assert!(
        observed_max <= RECEIPT_FALLBACK_MAX_IN_FLIGHT,
        "expected at most {} concurrent requests, observed {}",
        RECEIPT_FALLBACK_MAX_IN_FLIGHT,
        observed_max
    );
}
