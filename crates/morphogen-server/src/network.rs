//! Network layer for PIR server.
//!
//! Provides HTTP and WebSocket endpoints for:
//! - Health checks
//! - PIR queries (single and batch)
//! - Epoch metadata streaming

#[cfg(feature = "network")]
mod admin_auth;
#[cfg(feature = "network")]
pub mod api;
#[cfg(feature = "network")]
pub mod dto;
#[cfg(feature = "network")]
mod gpu_batch;
#[cfg(feature = "network")]
mod gpu_metrics;
#[cfg(feature = "network")]
mod gpu_scan;
#[cfg(feature = "network")]
mod scan_helpers;
#[cfg(feature = "network")]
mod serde_hex;
#[cfg(feature = "network")]
mod snapshot_config;
#[cfg(feature = "network")]
pub mod telemetry;

#[cfg(feature = "network")]
pub use api::*;
