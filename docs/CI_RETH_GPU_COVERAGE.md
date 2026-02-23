# CI Coverage: `morphogen-gpu-dpf` and `reth-adapter`

## Jobs in `.github/workflows/ci.yml`

| Job | Runner | Coverage |
| --- | --- | --- |
| `gpu-dpf` | `ubuntu-latest` | Compiles and runs `morphogen-gpu-dpf` library + integration tests on the CPU path (`RUSTFLAGS='-Dwarnings' cargo check -p morphogen-gpu-dpf`, `RUSTFLAGS='-Dwarnings' cargo test -p morphogen-gpu-dpf --lib`, `RUSTFLAGS='-Dwarnings' cargo test -p morphogen-gpu-dpf --tests`). |
| `reth-adapter` | `ubuntu-latest` | Validates default build/tests and `reth` feature compile/test coverage (`RUSTFLAGS='-Dwarnings' cargo build -p reth-adapter`, `RUSTFLAGS='-Dwarnings' cargo test -p reth-adapter`, `RUSTFLAGS='-Dwarnings' cargo check -p reth-adapter --features reth`, `RUSTFLAGS='-Dwarnings' cargo test -p reth-adapter --features reth --lib`, `RUSTFLAGS='-Dwarnings' cargo test -p reth-adapter --features reth --tests`). |
| `gpu-dpf-cuda` | `self-hosted` (`linux`, `x64`, `gpu`, `cuda`) | Validates CUDA build/test path (`cargo check -p morphogen-gpu-dpf --features cuda`, `cargo test -p morphogen-gpu-dpf --features cuda --lib`, `cargo test -p morphogen-gpu-dpf --features cuda --tests`). Enabled only when `ENABLE_GPU_CI=true` and the event is explicitly allowlisted (`push`, `workflow_dispatch`, or internal `pull_request`). |

## Expected Runner Capabilities

### `ubuntu-latest` jobs
- Stable Rust toolchain.
- No CUDA dependency required.

### `gpu-dpf-cuda` self-hosted job
- Linux x86_64 runner with labels: `self-hosted`, `linux`, `x64`, `gpu`, `cuda`.
- NVIDIA driver available (`nvidia-smi` must succeed).
- CUDA toolkit installed with `nvcc` on `PATH`.
- Access to the repository cache is recommended to avoid long cold starts.
- Security gate: the job uses an explicit event allowlist and is skipped for forked pull requests; only trusted contexts (push/workflow dispatch/internal PRs) can execute on self-hosted GPU runners.

## Failure Triage Workflow

1. Identify the failing command in the job logs (all jobs run one cargo command per step).
2. Reproduce locally with the same command and strict warnings:
   - `RUSTFLAGS='-Dwarnings' cargo test -p morphogen-gpu-dpf --lib`
   - `RUSTFLAGS='-Dwarnings' cargo test -p morphogen-gpu-dpf --tests`
   - `RUSTFLAGS='-Dwarnings' cargo test -p reth-adapter`
   - `RUSTFLAGS='-Dwarnings' cargo check -p reth-adapter --features reth`
   - `RUSTFLAGS='-Dwarnings' cargo test -p reth-adapter --features reth --lib`
   - `RUSTFLAGS='-Dwarnings' cargo test -p reth-adapter --features reth --tests`
3. For `reth-adapter --features reth` failures, first check for upstream `reth` API drift, then patch call sites/imports.
4. For CUDA job failures, verify runner health first:
   - `nvidia-smi`
   - `nvcc --version`
5. If the failure is runner-specific (driver/toolkit mismatch), fix the runner image and rerun; if it is source-related, patch code and rerun.
