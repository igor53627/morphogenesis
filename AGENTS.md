# Morphogenesis Agent Guidelines

> **Shared workflow**: this repo extends the project-agnostic
> [**global agents-md template**](https://github.com/igor53627/agents-md)
> (TDD · `backlog` · `jj` · roborev PR review loop · close/compact discipline).
> The shared sections below are a copy of `~/pse/agents-md/AGENTS.md`
> (Option 2 from the template README) so this file stays self-contained
> for pi / contributors who don't have the template locally.
> When the template changes, re-apply via `diff`.

Morphogenesis is a 2-server DPF-based PIR system for Ethereum state.
See `README.md` for the full architecture overview.

<!-- BEGIN SHARED (from https://github.com/igor53627/agents-md ) -->

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
2. **Wait for the GitHub bot reviews to land** before triaging — they are
   asynchronous and inline findings can arrive minutes after the push.
   Poll the PR's reviews/comments (see step 3) until each expected bot
   (`gemini-code-assist[bot]`, `coderabbitai[bot]`, `Cursor Bugbot`,
   `greptile-apps[bot]`) has
   either posted or timed out. If a bot never responds, record that
   explicitly ("timed out / not installed") in the PR body so the absence
   is deliberate, not accidental.
3. **Read the GitHub bot reviews**, not just roborev:
   - `gemini-code-assist[bot]` — inline comments (often the most actionable)
   - `coderabbitai[bot]`
   - `Cursor Bugbot`
   - `greptile-apps[bot]` — static analysis that catches cfg-gated compile
     failures the others miss (see lessons below)
   - any other installed reviewer bot
   Inspect with:
   ```bash
   gh api repos/igor53627/morphogenesis/pulls/<N>/reviews   --jq '.[] | {user,state,body}'
   gh api repos/igor53627/morphogenesis/pulls/<N>/comments  --jq '.[] | {user,path,line,body}'
   gh api repos/igor53627/morphogenesis/issues/<N>/comments --jq '.[].user.login'
   ```
4. **Triage every inline finding** (action / defer-with-reason / reject-with-reason).
   Actionable HIGH/MEDIUM findings MUST be fixed in the CURRENT PR before
   merge — do NOT merge with an actionable HIGH/MEDIUM finding outstanding.
   A follow-up PR is acceptable only for (a) historical findings on already-
   merged code, or (b) findings you explicitly defer with a recorded
   rationale in both the PR body and the review thread reply.
5. **Reply on the thread** for each actionable inline comment
   (`POST .../pulls/<N>/comments/<id>/replies`), linking the follow-up PR if
   the fix lands separately. Replies MUST go on the **same PR that owns
   the comment** — `comment_id` is scoped to its PR:
   ```bash
   gh api -X POST repos/<OWNER>/<REPO>/pulls/<N>/comments/<comment_id>/replies \
     -f body="Fixed in this PR via ..." --jq '{created: (.created_at != null)}'
   ```
6. **Resolve the thread** after replying — GitHub "Resolve conversation"
   is GraphQL-only (no REST endpoint). Get the thread node_id, then:
   ```bash
   gh api graphql -f query='{ repository(owner: "<OWNER>", name: "<REPO>") {
     pullRequest(number: <N>) { reviewThreads(first: 30) { nodes {
       id isResolved comments(first: 1) { nodes { databaseId } } } } } } }' \
     --jq '.data.repository.pullRequest.reviewThreads.nodes[]
           | select(.isResolved == false)
           | {thread_id: .id, comment_id: .comments.nodes[0].databaseId}'
   # Then resolve each:
   gh api graphql -f query='mutation { resolveReviewThread(input:
     {threadId: "PRRT_..."}) { thread { isResolved } } }'
   ```
   Never leave a thread open after triage — even on merged PRs.
7. **Close the roborev job** once findings are addressed or explicitly
   deferred: `roborev close <job_id>` (alias `address`). Open reviews are
   not a backlog — they are state that must be resolved per-PR.
8. **Compact regularly.** After merging a series of PRs (or any time
   `roborev list --open --all-branches --limit 200` shows > 10 open
   reviews), run:
   ```bash
   roborev compact --all-branches --wait --limit 50 --timeout 15m
   ```
   This consolidates duplicates, drops false positives / already-fixed
   findings, and auto-closes the originals. The resulting consolidated job
   must itself be triaged (step 4) and closed (step 6). Do this across ALL
   branches (`--all-branches`) — reviews live on whatever feature branch
   the PR used, not just `main`.
9. **If fixes are needed**, open a follow-up PR and re-run steps 1–8 on it.

### Pre-merge checklist (run this BEFORE every `gh pr merge`)

```bash
# 1. CI is CLEAN (all required checks SUCCESS)
gh pr view <N> --json mergeStateStatus --jq '.mergeStateStatus'  # must print CLEAN

# 2. greptile has posted (it is the slowest bot — 5-10 min after push).
#    Check BOTH inline comments AND reviews (greptile may post a top-level
#    review with no inline comment):
gh api repos/<OWNER>/<REPO>/pulls/<N>/comments \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | length'
gh api repos/<OWNER>/<REPO>/pulls/<N>/reviews \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | length'
# If BOTH are 0 AND the PR was pushed <10 min ago: WAIT. Do not merge.
# greptile catches cfg-gated compile failures and AC/notes consistency
# issues that gemini + roborev miss. It caught 3 real issues (P1/P2) on
# PRs #50, #55, #56 that all shipped to main because I merged too early.

# 3. No unresolved review threads (paginate if >30 threads)
gh api graphql -f query='{ repository(owner: "<OWNER>", name: "<REPO>") {
  pullRequest(number: <N>) { reviewThreads(first: 50) {
    nodes { isResolved } pageInfo { hasNextPage } } } } }' \
  --jq 'if .data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage
        then "ERROR: >50 threads, paginate manually"
        else [.data.repository.pullRequest.reviewThreads.nodes[]
              | select(.isResolved == false)] | length
        end'  # must print 0

# 4. roborev jobs closed
roborev list --open --limit 200 --json --repo "$PWD" \
  --branch <BRANCH> | jq 'length'  # must print 0
```

Only after ALL FOUR pass: `gh pr merge <N> --merge --delete-branch`.

Hard-won lessons:
- 2026-06-17, TASK-54.1–54.7: running only roborev and ignoring
  gemini-code-assist inline reviews let a real HIGH-severity bug
  (`--refresh-interval 0` tight loop) ship through 7 PRs. Bot-review handling
  is part of the merge gate.
- 2026-06-17, TASK-54.8–54.10: left 64 roborev reviews open across the
  series because step 6/7 were missing from the loop. `roborev compact
  --all-branches` cut them to 15 and surfaced 7 verified findings that
  had been silently accumulating. Always close per-PR and compact when
  open count > 10.
- 2026-06-17, TASK-54.11 (morphogenesis): codex security and
  `gemini-code-assist` independently caught a behavior regression
  (`is_allowed_snapshot_host` subdomain-matching lost during a refactor).
  Fixed in the same PR rather than follow-up because of step 4.
- 2026-06-17, TASK-55.3 (morphogenesis): left a stray `#[derive]` on
  `init_gpu_resources` during a Python-script extraction. Non-cuda CI
  stayed green (cfg-gated). `greptile-apps[bot]` — the 5th reviewer bot —
  caught it via static analysis (P1). Always poll `pulls/<N>/comments`
  for ALL bot users, not just check for non-empty review summaries.
- 2026-06-17: across PRs #22–#50, left 23 review threads open after
  replying — the "Resolve conversation" GraphQL step was missing from
  the loop. Added as step 6. Never leave a thread open after triage.

<!-- END SHARED -->

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

### Project invariants (reviewers MUST preserve)

- **Privacy fail-closed (TASK-37)**: in `morphogen-rpc-adapter`, every upstream
  fallback for a private method must gate through
  `state::fail_closed_if_fallback_disabled`. In prod, `--fallback-to-upstream`
  requires `--allow-privacy-degraded-fallback` (enforced by
  `config::validate_privacy_fallback_config`).
- **URL redaction**: upstream URLs may carry credentials in userinfo/query.
  Log them only via `telemetry::sanitize_url_for_telemetry`. `reqwest::Error`
  debug output must be redacted (see `proxy::redact_reqwest_error`) before
  logging.
- **Dropped methods** (`eth_getProof` / `eth_sign` / `eth_signTransaction`):
  rejected at the adapter; never proxied to upstream.
- **Relay routing**: `eth_sendRawTransaction` → Flashbots Protect (or
  configured `--tx-relay`), never the public mempool.
- **Server bind**: the RPC adapter binds to `127.0.0.1` only by default.
