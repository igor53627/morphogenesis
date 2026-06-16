//! Binary entry point for `morphogen-rpc-adapter`.
//!
//! All logic lives in the [`morphogen_rpc_adapter`] library crate; this file is
//! a thin wrapper that hands control to [`morphogen_rpc_adapter::run`].
//!
//! See TASK-54.1: the library was extracted from this file so the handlers,
//! proxy, filter, and config code can be split into modules and the inline unit
//! tests can move to `tests/` in follow-up changes.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    morphogen_rpc_adapter::run().await
}
