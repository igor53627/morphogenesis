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

### Pull Request review loop (MANDATORY — do not skip)
Every PR must complete the FULL review loop before merge. Roborev is one
input, not the whole loop.

1. **Run roborev 2x2** (`codex` + `claude-code` x `security` + `design`) on
   the pushed SHA (sequential enqueue to avoid the daemon race).
2. **Read the GitHub bot reviews**, not just roborev:
   - `gemini-code-assist[bot]` — inline comments (often the most actionable)
   - `coderabbitai[bot]`
   - `Cursor Bugbot`
   - any other installed reviewer bot
   Inspect with:
   ```bash
   gh api repos/igor53627/morphogenesis/pulls/<N>/reviews   --jq '.[] | {user,state,body}'
   gh api repos/igor53627/morphogenesis/pulls/<N>/comments  --jq '.[] | {user,path,line,body}'
   gh api repos/igor53627/morphogenesis/issues/<N>/comments --jq '.[].user.login'
   ```
3. **Triage every inline finding** (action / defer-with-reason / reject-with-reason).
   Do NOT merge with an unaddressed HIGH/MEDIUM bot finding unless you record
   an explicit reason in the PR body.
4. **Reply on the thread** for each actionable inline comment (`POST .../pulls/<N>/comments/<id>/replies`),
   linking the follow-up PR if the fix lands separately. Replies MUST go on
   the same PR that owns the comment (comment_id is scoped to its PR).
5. **If fixes are needed**, open a follow-up PR and re-run steps 1–4 on it.

Hard-won lesson (2026-06-17, TASK-54.1–54.7): running only roborev and
ignoring gemini-code-assist inline reviews let a real HIGH-severity bug
(`--refresh-interval 0` tight loop) ship through 7 PRs. Bot-review handling
is part of the merge gate.

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
