# Task-Commit Linking Best Practices

## Overview

Link tasks to commits for complete traceability between planning and implementation.

## When to Add References

### During Development
```bash
# After committing, add reference to the task
git commit -m "feat: implement feature"
COMMIT_SHA=$(git rev-parse --short HEAD)
backlog task edit TASK-X --ref "$COMMIT_SHA - Brief description"
```

### Adding roborev Job References
```bash
# Include roborev job ID for review traceability
backlog task edit TASK-X --ref "abc1234 - Initial impl (roborev job 21)"
```

### Multiple Commits Per Task
```bash
# Add multiple references as work progresses
backlog task edit TASK-6 \
  --ref "aa13844 - Initial implementation" \
  --ref "afa4322 - Security fixes (roborev job 21)" \
  --ref "9596c98 - Test refactoring (roborev job 22)"
```

## Searching for Commits

### Find All Commits for a Task
```bash
# View task details including references
backlog task view TASK-6

# Grep task file for references
grep "references:" -A 10 backlog/tasks/task-*.md
```

### Find Task for a Commit
```bash
# Search commit message for task ID
git log --grep="TASK-6"

# Or search backlog files for commit SHA
grep -r "9596c98" backlog/tasks/
```

### Using roborev
```bash
# View commit review
roborev show 9596c98

# See which task the review relates to
roborev show 21 --job | head -20
```

## Commit Message Convention

Include task ID in commit messages:

```bash
git commit -m "feat(rpc): add missing methods

Implements standard RPC passthroughs for wallet compatibility.

Addresses TASK-6

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

This enables:
- `git log --grep="TASK-6"` to find all commits
- Automatic linking in many tools
- Clear traceability in git history

## Automation Ideas

### Post-Commit Hook
```bash
# .git/hooks/post-commit
#!/bin/sh
# Extract TASK-ID from commit message
TASK_ID=$(git log -1 --pretty=%B | grep -o 'TASK-[0-9]*')
COMMIT_SHA=$(git rev-parse --short HEAD)

if [ -n "$TASK_ID" ]; then
  backlog task edit "$TASK_ID" --ref "$COMMIT_SHA - $(git log -1 --pretty=%s)"
fi
```

## Benefits

1. **Traceability**: Know exactly which commits implemented which tasks
2. **Review Context**: Link roborev findings back to tasks
3. **History**: Understand evolution of a feature
4. **Search**: Find related work quickly
5. **Documentation**: Self-documenting development process

## Example: TASK-6

```yaml
references:
  - aa13844 - Initial implementation
  - afa4322 - Security fixes (roborev job 21)
  - 9596c98 - Test refactoring (roborev job 22)
```

Shows the complete development arc:
1. Initial feature implementation
2. Security review and fixes
3. Code quality improvements

## Integration with roborev

roborev jobs automatically track:
- Commit SHA
- Review findings
- Addressed status

Link these to tasks for complete audit trail:
```
TASK-6 → Commits (aa13844, afa4322, 9596c98) → roborev Jobs (21, 22) → Findings → Fixes
```
