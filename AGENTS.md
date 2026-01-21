# Morphogenesis Agent Guidelines

## Development Workflow

### TDD (Test-Driven Development)
- Write tests FIRST before implementing features
- Red -> Green -> Refactor cycle
- Run tests frequently during development

### Progress Tracking
- Use `docs/KANBAN.md` to track all work
- Update KANBAN when starting/completing tasks
- Mark phases with `[x]` when done

### Version Control
- Use `jj` (Jujutsu) instead of `git` for all VCS operations
- Common commands:
  - `jj status` - show working copy changes
  - `jj log` - show commit history
  - `jj new -m "message"` - create new change
  - `jj describe -m "message"` - update current change description
  - `jj squash` - squash into parent
  - `jj metaedit --update-author` - fix author info

## Build & Test Commands

```bash
cargo build --package morphogen-server --features network
cargo test --package morphogen-server --features network
cargo fmt --package morphogen-server
cargo clippy --package morphogen-server --features network
```

## Project Structure

- `crates/morphogen-server/` - PIR server with HTTP/WebSocket API
- `crates/morphogen-core/` - Core types (DeltaBuffer, EpochSnapshot, Cuckoo)
- `crates/morphogen-storage/` - AlignedMatrix, ChunkedMatrix
- `crates/morphogen-dpf/` - DPF key generation and evaluation
- `crates/morphogen-client/` - Client library and fixtures
- `crates/reth-adapter/` - ETL tool for converting Reth DB to Cuckoo Matrix
- `docs/KANBAN.md` - Project task tracking
