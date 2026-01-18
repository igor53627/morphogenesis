//! Integration tests for the network API.

#![cfg(feature = "network")]

use std::sync::Arc;

use morphogen_server::network::{create_router, AppState, EpochMetadata};
use tokio::sync::watch;

fn test_state() -> Arc<AppState> {
    let initial = EpochMetadata {
        epoch_id: 42,
        num_rows: 100_000,
        seeds: [0x1234, 0x5678, 0x9ABC],
        block_number: 12345678,
        state_root: [0xAB; 32],
    };
    let (_tx, rx) = watch::channel(initial);
    Arc::new(AppState {
        epoch_id: 42,
        num_rows: 100_000,
        seeds: [0x1234, 0x5678, 0x9ABC],
        block_number: 12345678,
        state_root: [0xAB; 32],
        epoch_rx: rx,
    })
}

mod health {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn health_returns_ok_status() {
        let app = create_router(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_returns_epoch_info() {
        let app = create_router(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["status"], "ok");
        assert_eq!(json["epoch_id"], 42);
        assert_eq!(json["block_number"], 12345678);
    }
}

mod epoch {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn epoch_returns_metadata() {
        let app = create_router(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/epoch")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["epoch_id"], 42);
        assert_eq!(json["num_rows"], 100_000);
        assert_eq!(json["block_number"], 12345678);
        assert!(json["state_root"].as_str().unwrap().starts_with("0x"));
    }

    #[tokio::test]
    async fn epoch_returns_seeds_array() {
        let app = create_router(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/epoch")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let seeds = json["seeds"].as_array().unwrap();
        assert_eq!(seeds.len(), 3);
        assert_eq!(seeds[0], 0x1234);
        assert_eq!(seeds[1], 0x5678);
        assert_eq!(seeds[2], 0x9ABC);
    }
}

mod query {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn query_accepts_three_keys() {
        let app = create_router(test_state());

        let body = r#"{"keys":["0xaabbccdd","0x11223344","0x55667788"]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/query")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn query_rejects_wrong_key_count() {
        let app = create_router(test_state());

        let body = r#"{"keys":["0xaabbccdd","0x11223344"]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/query")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn query_returns_three_payloads() {
        let app = create_router(test_state());

        let body = r#"{"keys":["0xaabbccdd","0x11223344","0x55667788"]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/query")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["epoch_id"], 42);
        let payloads = json["payloads"].as_array().unwrap();
        assert_eq!(payloads.len(), 3);
    }

    #[tokio::test]
    async fn query_returns_epoch_id_matching_state() {
        let initial = EpochMetadata {
            epoch_id: 999,
            num_rows: 1000,
            seeds: [1, 2, 3],
            block_number: 100,
            state_root: [0; 32],
        };
        let (_tx, rx) = watch::channel(initial);
        let state = Arc::new(AppState {
            epoch_id: 999,
            num_rows: 1000,
            seeds: [1, 2, 3],
            block_number: 100,
            state_root: [0; 32],
            epoch_rx: rx,
        });
        let app = create_router(state);

        let body = r#"{"keys":["0xaa","0xbb","0xcc"]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/query")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["epoch_id"], 999);
    }
}

mod websocket_epoch {
    use super::*;
    use futures_util::StreamExt;
    use std::future::IntoFuture;
    use std::net::{Ipv4Addr, SocketAddr};
    use tokio_tungstenite::tungstenite;

    fn test_state_with_watch() -> (Arc<AppState>, watch::Sender<EpochMetadata>) {
        let initial = EpochMetadata {
            epoch_id: 42,
            num_rows: 100_000,
            seeds: [0x1234, 0x5678, 0x9ABC],
            block_number: 12345678,
            state_root: [0xAB; 32],
        };
        let (tx, rx) = watch::channel(initial);
        let state = Arc::new(AppState {
            epoch_id: 42,
            num_rows: 100_000,
            seeds: [0x1234, 0x5678, 0x9ABC],
            block_number: 12345678,
            state_root: [0xAB; 32],
            epoch_rx: rx,
        });
        (state, tx)
    }

    #[tokio::test]
    async fn ws_epoch_connects_successfully() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let (state, _tx) = test_state_with_watch();
        let app = create_router(state);

        tokio::spawn(axum::serve(listener, app).into_future());

        let (socket, response) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/epoch"))
            .await
            .unwrap();

        assert_eq!(response.status(), 101);
        drop(socket);
    }

    #[tokio::test]
    async fn ws_epoch_receives_initial_metadata() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let (state, _tx) = test_state_with_watch();
        let app = create_router(state);

        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/epoch"))
            .await
            .unwrap();

        let msg = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
            .await
            .expect("timeout waiting for message")
            .expect("stream ended")
            .expect("websocket error");

        let text = match msg {
            tungstenite::Message::Text(t) => t,
            other => panic!("expected text message, got {:?}", other),
        };

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["epoch_id"], 42);
        assert_eq!(json["num_rows"], 100_000);
        assert_eq!(json["block_number"], 12345678);
        assert!(json["state_root"].as_str().unwrap().starts_with("0x"));
    }

    #[tokio::test]
    async fn ws_epoch_metadata_contains_seeds() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let (state, _tx) = test_state_with_watch();
        let app = create_router(state);

        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/epoch"))
            .await
            .unwrap();

        let msg = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        let text = match msg {
            tungstenite::Message::Text(t) => t,
            other => panic!("expected text message, got {:?}", other),
        };

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let seeds = json["seeds"].as_array().unwrap();
        assert_eq!(seeds.len(), 3);
        assert_eq!(seeds[0], 0x1234);
    }

    #[tokio::test]
    async fn ws_epoch_pushes_on_change() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let (state, tx) = test_state_with_watch();
        let app = create_router(state);

        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/epoch"))
            .await
            .unwrap();

        let msg1 = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let text1 = match msg1 {
            tungstenite::Message::Text(t) => t,
            other => panic!("expected text, got {:?}", other),
        };
        let json1: serde_json::Value = serde_json::from_str(&text1).unwrap();
        assert_eq!(json1["epoch_id"], 42);

        tx.send(EpochMetadata {
            epoch_id: 43,
            num_rows: 100_000,
            seeds: [0x1234, 0x5678, 0x9ABC],
            block_number: 12345690,
            state_root: [0xCD; 32],
        })
        .unwrap();

        let msg2 = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
            .await
            .expect("timeout waiting for epoch update")
            .unwrap()
            .unwrap();
        let text2 = match msg2 {
            tungstenite::Message::Text(t) => t,
            other => panic!("expected text, got {:?}", other),
        };
        let json2: serde_json::Value = serde_json::from_str(&text2).unwrap();
        assert_eq!(json2["epoch_id"], 43);
        assert_eq!(json2["block_number"], 12345690);
    }

    #[tokio::test]
    async fn ws_epoch_multiple_updates() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let (state, tx) = test_state_with_watch();
        let app = create_router(state);

        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/epoch"))
            .await
            .unwrap();

        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
            .await
            .unwrap();

        for epoch_id in 43..=45 {
            tx.send(EpochMetadata {
                epoch_id,
                num_rows: 100_000,
                seeds: [0x1234, 0x5678, 0x9ABC],
                block_number: 12345678 + epoch_id,
                state_root: [epoch_id as u8; 32],
            })
            .unwrap();

            let msg = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
                .await
                .expect("timeout")
                .unwrap()
                .unwrap();
            let text = match msg {
                tungstenite::Message::Text(t) => t,
                other => panic!("expected text, got {:?}", other),
            };
            let json: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(json["epoch_id"], epoch_id);
        }
    }
}

mod websocket_query {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use std::future::IntoFuture;
    use std::net::{Ipv4Addr, SocketAddr};
    use tokio_tungstenite::tungstenite;

    #[tokio::test]
    async fn ws_query_connects_successfully() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let app = create_router(test_state());

        tokio::spawn(axum::serve(listener, app).into_future());

        let (socket, response) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/query"))
            .await
            .unwrap();

        assert_eq!(response.status(), 101);
        drop(socket);
    }

    #[tokio::test]
    async fn ws_query_processes_keys_and_returns_payloads() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let app = create_router(test_state());

        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/query"))
            .await
            .unwrap();

        let request = r#"{"keys":["0xaabbccdd","0x11223344","0x55667788"]}"#;
        socket
            .send(tungstenite::Message::text(request))
            .await
            .unwrap();

        let msg = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
            .await
            .expect("timeout")
            .expect("stream ended")
            .expect("websocket error");

        let text = match msg {
            tungstenite::Message::Text(t) => t,
            other => panic!("expected text message, got {:?}", other),
        };

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["epoch_id"], 42);
        let payloads = json["payloads"].as_array().unwrap();
        assert_eq!(payloads.len(), 3);
    }

    #[tokio::test]
    async fn ws_query_handles_multiple_requests() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let app = create_router(test_state());

        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/query"))
            .await
            .unwrap();

        for i in 0..3 {
            let request = r#"{"keys":["0xaa","0xbb","0xcc"]}"#;
            socket
                .send(tungstenite::Message::text(request))
                .await
                .unwrap();

            let msg = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
                .await
                .expect("timeout")
                .expect("stream ended")
                .expect("websocket error");

            let text = match msg {
                tungstenite::Message::Text(t) => t,
                other => panic!("expected text message, got {:?}", other),
            };

            let json: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(json["epoch_id"], 42, "request {i} failed");
        }
    }

    #[tokio::test]
    async fn ws_query_returns_error_for_wrong_key_count() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let app = create_router(test_state());

        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/query"))
            .await
            .unwrap();

        let request = r#"{"keys":["0xaa","0xbb"]}"#;
        socket
            .send(tungstenite::Message::text(request))
            .await
            .unwrap();

        let msg = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
            .await
            .expect("timeout")
            .expect("stream ended")
            .expect("websocket error");

        let text = match msg {
            tungstenite::Message::Text(t) => t,
            other => panic!("expected text message, got {:?}", other),
        };

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(json["error"].is_string());
    }
}
