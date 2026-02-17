# Morphogenesis Agent Guidelines

## Development Workflow

### TDD (Test-Driven Development)
- Write tests FIRST before implementing features
- Red -> Green -> Refactor cycle
- Run tests frequently during development

### Progress Tracking
- Use `backlog` CLI to track all work (`backlog task list`, `backlog task edit`, `backlog task view`)
- Update the relevant backlog task status/notes when starting or completing work
- Treat `docs/KANBAN.md` as historical archive only (do not use it for active tracking)

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
- `backlog/tasks/` - Active task tracking (managed via `backlog` CLI)
- `docs/KANBAN.md` - Historical task notes (archival)
