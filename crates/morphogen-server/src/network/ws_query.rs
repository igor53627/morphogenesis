//! WebSocket query message handlers (single + batch) and the session loop.
//!
//! Extracted from `network/api.rs` in TASK-54.16. This module owns the
//! per-message processing for `/ws/query`: JSON-RPC style single and batch
//! PIR query execution over a WebSocket, with size/queue caps and a
//! pipelined (one in-flight + queued) session loop.
//!
//! Private to the `network` module — NOT part of the crate's public API.
//! Referenced from `network/api.rs::ws_query_handler` via
//! `use super::ws_query::handle_ws_query;`.
//!
//! Deps (resolved via globs):
//! - [`super::api`] (`*`): `AppState`, `WsQueryError`, `MAX_BATCH_SIZE`,
//!   `MAX_WS_MESSAGE_BYTES`, `MAX_WS_QUEUED_MESSAGES`
//! - [`super::dto`] (`*` via api re-export): `QueryRequest`, `QueryResponse`,
//!   `BatchQueryRequest`, `BatchQueryResponse`, `BatchQueryResult`
//! - [`super::scan_helpers`] (`*`): `payload_array_into_vec`,
//!   `apply_delta_entries_to_payloads`

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};

use super::api::*;
use super::scan_helpers::*;

pub(crate) const WS_INTERNAL_ERROR: &str =
    r#"{"error":"internal server error","code":"internal_error"}"#;

pub(crate) fn ws_error_json(error: &str, code: &str) -> String {
    serde_json::to_string(&WsQueryError {
        error: error.to_string(),
        code: code.to_string(),
    })
    .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string())
}

pub(crate) async fn handle_ws_single_query(state: &AppState, request: QueryRequest) -> String {
    use crate::scan::scan_consistent;
    use morphogen_dpf::AesDpfKey;

    if request.keys.len() != 3 {
        return ws_error_json("expected exactly 3 keys", "bad_request");
    }

    let keys_result = (
        AesDpfKey::from_bytes(&request.keys[0]),
        AesDpfKey::from_bytes(&request.keys[1]),
        AesDpfKey::from_bytes(&request.keys[2]),
    );

    match keys_result {
        (Ok(k0), Ok(k1), Ok(k2)) => {
            let keys = [k0, k1, k2];
            let global = Arc::clone(&state.global);
            let row_size_bytes = state.row_size_bytes;
            let scan_result = match tokio::task::spawn_blocking(move || {
                let pending = global.load_pending();
                scan_consistent(global.as_ref(), pending.as_ref(), &keys, row_size_bytes)
            })
            .await
            {
                Ok(result) => result,
                Err(_) => return ws_error_json("internal error", "internal_error"),
            };

            match scan_result {
                Ok((results, epoch_id)) => serde_json::to_string(&QueryResponse {
                    epoch_id,
                    payloads: payload_array_into_vec(results),
                })
                .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string()),
                Err(e) => {
                    use crate::scan::ScanError;
                    let code = match e {
                        ScanError::TooManyRetries { .. } => "too_many_retries",
                        ScanError::LockPoisoned => "internal_error",
                        ScanError::MatrixNotAligned { .. } => "internal_error",
                        ScanError::ChunkNotAligned { .. } => "internal_error",
                    };
                    ws_error_json(&format!("scan error: {}", e), code)
                }
            }
        }
        _ => ws_error_json("invalid key format", "bad_request"),
    }
}

pub(crate) async fn handle_ws_batch_query(state: &AppState, request: BatchQueryRequest) -> String {
    use crate::scan::scan_consistent;
    use morphogen_dpf::AesDpfKey;

    let n = request.queries.len();
    if n == 0 || n > MAX_BATCH_SIZE {
        return ws_error_json("batch size must be 1..=32", "bad_request");
    }

    let mut all_keys: Vec<[AesDpfKey; 3]> = Vec::with_capacity(n);
    for query in &request.queries {
        if query.keys.len() != 3 {
            return ws_error_json("each query must have exactly 3 keys", "bad_request");
        }
        match (
            AesDpfKey::from_bytes(&query.keys[0]),
            AesDpfKey::from_bytes(&query.keys[1]),
            AesDpfKey::from_bytes(&query.keys[2]),
        ) {
            (Ok(k0), Ok(k1), Ok(k2)) => all_keys.push([k0, k1, k2]),
            _ => return ws_error_json("invalid key format", "bad_request"),
        }
    }

    // Try consistent snapshot for all queries
    let snapshot1 = state.global.load();
    let epoch1 = snapshot1.epoch_id;
    let pending = state.global.load_pending();
    let snapshot_result = pending.snapshot_with_epoch();
    let (pending_epoch, entries) = match snapshot_result {
        Ok(r) => r,
        Err(_) => return ws_error_json("internal error", "internal_error"),
    };
    let snapshot2 = state.global.load();

    if snapshot2.epoch_id != epoch1 || pending_epoch != epoch1 {
        // Fallback: per-query scan_consistent, tracking epoch consistency
        let mut results = Vec::with_capacity(n);
        let mut batch_epoch: Option<u64> = None;
        let global = Arc::clone(&state.global);
        let row_size_bytes = state.row_size_bytes;
        for keys in &all_keys {
            let keys = keys.clone();
            let global_for_scan = Arc::clone(&global);
            let scan_result = match tokio::task::spawn_blocking(move || {
                let pending_for_scan = global_for_scan.load_pending();
                scan_consistent(
                    global_for_scan.as_ref(),
                    pending_for_scan.as_ref(),
                    &keys,
                    row_size_bytes,
                )
            })
            .await
            {
                Ok(result) => result,
                Err(_) => return ws_error_json("internal error", "internal_error"),
            };

            match scan_result {
                Ok((payloads, query_epoch)) => {
                    match batch_epoch {
                        None => batch_epoch = Some(query_epoch),
                        Some(e) if e != query_epoch => {
                            return ws_error_json(
                                "epoch changed during batch scan",
                                "too_many_retries",
                            );
                        }
                        _ => {}
                    }
                    results.push(BatchQueryResult {
                        payloads: payload_array_into_vec(payloads),
                    });
                }
                Err(e) => {
                    use crate::scan::ScanError;
                    let code = match e {
                        ScanError::TooManyRetries { .. } => "too_many_retries",
                        ScanError::LockPoisoned
                        | ScanError::MatrixNotAligned { .. }
                        | ScanError::ChunkNotAligned { .. } => "internal_error",
                    };
                    return ws_error_json(&format!("scan error: {}", e), code);
                }
            }
        }
        let epoch_id = batch_epoch.unwrap_or_else(|| state.global.load().epoch_id);
        return serde_json::to_string(&BatchQueryResponse { epoch_id, results })
            .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string());
    }

    let matrix = Arc::clone(&snapshot1.matrix);
    let row_size_bytes = state.row_size_bytes;
    let results = match tokio::task::spawn_blocking(move || {
        let mut results = Vec::with_capacity(n);
        for keys in &all_keys {
            let mut payloads = crate::scan::scan_main_matrix(matrix.as_ref(), keys, row_size_bytes);
            apply_delta_entries_to_payloads(&mut payloads, keys, &entries).map_err(|_| ())?;
            results.push(BatchQueryResult {
                payloads: payload_array_into_vec(payloads),
            });
        }
        Ok(results)
    })
    .await
    {
        Ok(Ok(results)) => results,
        Ok(Err(())) => return ws_error_json("delta length mismatch", "internal_error"),
        Err(_) => return ws_error_json("internal error", "internal_error"),
    };

    serde_json::to_string(&BatchQueryResponse {
        epoch_id: epoch1,
        results,
    })
    .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string())
}

pub(crate) async fn handle_ws_query(mut socket: WebSocket, state: Arc<AppState>) {
    use std::collections::VecDeque;

    let mut queued_text_messages: VecDeque<String> = VecDeque::new();

    loop {
        let text = if let Some(queued) = queued_text_messages.pop_front() {
            queued
        } else {
            let Some(Ok(msg)) = socket.recv().await else {
                break;
            };
            match msg {
                Message::Text(text) => text.to_string(),
                Message::Close(_) => break,
                _ => continue,
            }
        };

        if text.len() > MAX_WS_MESSAGE_BYTES {
            let error = ws_error_json("message too large", "message_too_large");
            let _ = socket.send(Message::Text(error.into())).await;
            continue;
        }

        let handler_state = Arc::clone(&state);
        let request_text = text;
        let mut response_task = tokio::spawn(async move {
            // Try batch request first (has "queries" field), fall back to single query.
            if let Ok(batch) = serde_json::from_str::<BatchQueryRequest>(&request_text) {
                handle_ws_batch_query(&handler_state, batch).await
            } else {
                match serde_json::from_str::<QueryRequest>(&request_text) {
                    Ok(request) => handle_ws_single_query(&handler_state, request).await,
                    Err(e) => ws_error_json(&e.to_string(), "bad_request"),
                }
            }
        });

        let response = loop {
            tokio::select! {
                result = &mut response_task => {
                    break match result {
                        Ok(response) => response,
                        Err(_) => ws_error_json("internal error", "internal_error"),
                    };
                }
                incoming = socket.recv() => {
                    match incoming {
                        Some(Ok(Message::Text(next_text))) => {
                            let next_text = next_text.to_string();
                            if next_text.len() > MAX_WS_MESSAGE_BYTES {
                                let error = ws_error_json("message too large", "message_too_large");
                                let _ = socket.send(Message::Text(error.into())).await;
                                continue;
                            }
                            if queued_text_messages.len() >= MAX_WS_QUEUED_MESSAGES {
                                let error = ws_error_json(
                                    "too many queued websocket messages",
                                    "too_many_messages",
                                );
                                let _ = socket.send(Message::Text(error.into())).await;
                                continue;
                            }
                            queued_text_messages.push_back(next_text);
                        }
                        Some(Ok(Message::Close(_))) | None | Some(Err(_)) => {
                            response_task.abort();
                            return;
                        }
                        Some(Ok(_)) => {}
                    }
                }
            }
        };

        if socket.send(Message::Text(response.into())).await.is_err() {
            break;
        }
    }
}
