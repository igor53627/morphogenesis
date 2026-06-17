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
mod gpu_batch;
#[cfg(feature = "network")]
mod gpu_metrics;
#[cfg(feature = "network")]
mod serde_hex;
#[cfg(feature = "network")]
pub mod telemetry;

#[cfg(feature = "network")]
pub use api::*;
