//! GPU-accelerated DPF using ChaCha8 PRG for PIR.
//!
//! This crate provides a GPU-friendly DPF implementation that uses ChaCha8
//! instead of AES for the PRG. ChaCha is based on ARX (Add-Rotate-XOR) operations
//! which are efficient on GPUs without requiring special instructions like AES-NI.
//!
//! # Architecture
//!
//! The implementation uses a fused kernel approach:
//! 1. DPF evaluation generates 16-byte masks for each page
//! 2. Masks are AND'd with database pages
//! 3. Results are XOR-accumulated into the output
//!
//! All three operations happen in a single pass over the database,
//! minimizing memory bandwidth usage.

#[cfg(feature = "cuda")]
pub extern crate cudarc;

pub mod chacha_prg;
pub mod dpf;
pub mod kernel;

pub use chacha_prg::{ChaCha8Prg, PrgOutput};
pub use dpf::{ChaChaKey, ChaChaParams, GpuDpfError};
#[cfg(feature = "cuda")]
pub use kernel::GpuScanner;
