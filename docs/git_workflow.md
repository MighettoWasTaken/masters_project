# Git Workflow

This project uses a single shared repository with task-scoped feature branches and manual PR review by the team lead. This document covers branching rules, syncing strategy, and the review priority order.

---

## Branching model

```
main
 ├── task/17-1          (team lead)        — no dependencies, branch from main
 ├── task/17-2          (team lead)        — depends on 17.1, branch after 17.1 merges
 ├── task/17-4          (codegen)          — parallel track, branch from main immediately
 ├── task/17-3          (team lead)        — depends on 17.1 + 17.2
 ├── task/17-5          (CUDA eng)         — depends on 17.1 + 17.2
 ├── task/17-6          (CUDA eng)         — depends on 17.2 + 17.5
 ├── task/17-7          (CUDA eng)         — depends on 17.1 + 17.2 + 17.5
 ├── task/17-8          (CUDA + codegen)   — depends on 17.4 + 17.7
 ├── task/17-9          (VRAM eng)         — depends on 17.3 + 17.5 + 17.7
 ├── task/17-10         (team lead)        — depends on 17.1 + 17.3
 ├── task/17-11         (test eng)         — depends on full stack (17.5–17.10)
 └── task/17-12         (test eng)         — depends on 17.11
```

Branch naming: `task/17-X` or `task/17-X-Y` for sub-tasks (e.g. `task/17-5-1`).

---

## Core rules

### 1. Don't branch until your dependencies have merged

Never start a task branch until every task it depends on has merged to `main`. Use the dependency list at the top of each task file.

If your dependency is delayed, do design work, read the relevant headers, write tests — but don't write implementation code that assumes an interface that hasn't landed yet.

### 2. Branch from `main`, not from another dev's branch

Once the dependency task merges to `main`, branch from `main`:

```bash
git checkout main
git pull
git checkout -b task/17-5
```

The only exception is a sub-task file (`task/17-5-1`) that directly extends work in the parent branch before that branch has merged — in that case branch from the parent task branch and PR to it, not to `main`.

### 3. Sync by rebasing, not merging

When `main` moves forward (another task merges), update your branch with:

```bash
git fetch origin
git rebase origin/main
```

Do not `git merge main` into your feature branch. Merge commits from routine syncs make PR diffs noisy and harder to review. Rebase keeps the history linear and surfaces conflicts one commit at a time.

If a rebase produces conflicts, resolve them before pushing. If the conflicts are substantial (interface changed under you), flag it to the team lead before proceeding.

### 4. One PR per task file

Each PR corresponds to one task file. PRs that bundle multiple tasks will be sent back for splitting. Smaller PRs mean faster reviews, which means faster unblocking of downstream tasks.

For large tasks completed across multiple commits, one PR is still correct — just keep the individual commits clean and task-scoped.

### 5. PRs target `main`

All PRs go to `main`. The team lead reviews and merges. Do not merge your own PR.

---

## Review priority order

The team lead reviews in dependency order — blockers first:

| Priority | Task | Blocks |
|---|---|---|
| 1 | 17.1 | Everything |
| 2 | 17.2 | 17.3, 17.5, 17.6, 17.7 |
| 3 | 17.3 | 17.9, 17.10 |
| 3 | 17.4 | 17.8 |
| 3 | 17.5 | 17.6, 17.7, 17.9 |
| 4 | 17.7 | 17.8 |
| 4 | 17.9 | 17.11 |
| 4 | 17.10 | 17.11 |
| 5 | 17.8 | 17.11 |
| 6 | 17.11 | 17.12 |
| 7 | 17.12 | — |

A one-day delay on 17.1 or 17.2 cascades to the entire team. Prioritize those reviews above everything else.

---

## What to do while waiting for a dependency

- Read the task file and the relevant headers/source files in `main`
- Write the test file (tests can be written against the specified interface before the implementation lands)
- Write stub headers for your own new files
- Review the "Contract for downstream tasks" section of the dependency task file — if anything is unclear, raise it before the dependency PR closes, not after

---

## Interface contract violations

The leading cause of incompatible work is one dev assuming an interface that another dev changed. The mitigation:

- The "Contract for downstream tasks" section of each task file is binding. If you need to deviate from it during implementation, open a discussion before merging — downstream devs may already be building against the specified interface.
- If a merged interface needs to change, create a `task/17-X-Y` branch off `main`, update the contract in the task file, and PR it as a small targeted change. Downstream devs then rebase.

---

## Quick reference

```
# Start a new task (after dependencies merged)
git checkout main && git pull
git checkout -b task/17-X

# Sync when main moves
git fetch origin
git rebase origin/main

# Push and open PR
git push -u origin task/17-X
# Open PR on GitHub targeting main
# Title: [task 17.X] completed|partial: <summary>
```
