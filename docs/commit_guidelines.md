# Commit Message Guidelines

All commits to this repository follow a structured format. These guidelines exist so that task progress, design decisions, and file changes are traceable from the git log alone — without needing to cross-reference task files or chat history.

---

## Format

```
[task X.Y[.Z]] <status>: <one-line summary>

Task: <task file reference(s)>
Status: <Completed | Partial — <what was done>>
Files: <comma-separated list of modified files>

<body: design decisions, non-obvious choices, constraints discovered>

[New task file: taskX.Y.Z.md — <one-line reason>]
```

---

## Fields

### Subject line (first line)

```
[task 17.3] partial: PoolManager CUDA routing + assign_to_device
```

- **Tag** — `[task X.Y]` or `[task X.Y.Z]` at the start. Use the task number from the task file. If the commit touches multiple tasks, list them: `[task 17.1, 17.2]`.
- **Status keyword** — `completed` or `partial`. Lowercase.
- **Summary** — what was done, not what the task says to do. Max 72 characters total for the subject line.

### `Task:` line

Full filename(s) of the task file(s) this commit advances.

```
Task: task17.3.md
```

If the commit also creates a new sub-task file:

```
Task: task17.3.md, task17.3.1.md (new)
```

### `Status:` line

For completed tasks:
```
Status: Completed
```

For partial commits, describe concretely what progress was made:
```
Status: Partial — assign_to_device() and synchronize_cuda() implemented;
        pinned memory allocation in Network not yet done
```

### `Files:` line

List every file that was meaningfully changed. Use paths relative to the repo root. Separate with commas; wrap at 80 characters.

```
Files: src/cpp/include/hodgkin_huxley/network/pool_manager.hpp,
       src/cpp/src/network/pool_manager.cpp,
       src/cpp/include/hodgkin_huxley/regional_network.hpp
```

### Body

One short paragraph (3–6 lines) covering:
- **Design decisions** — choices that weren't obvious from the task spec.
- **Constraints discovered** — anything that required deviating from the task file.
- **Deferred items** — things the task file asks for that are deliberately left for a follow-on commit.

Skip the body only if the subject line fully describes the commit (e.g. a one-liner fix or stub file creation).

### Completing a task — required housekeeping

When a task is fully done, two additional changes must be included in the same commit:

1. **Move the task file to `completed/`:**
   ```
   git mv task17.1.md completed/task17.1.md
   ```

2. **Check it off in `task17.md`** (or the relevant parent task file). Change the row's checkbox from `[ ]` to `[x]`:
   ```
   | [17.1](completed/task17.1.md) | Team lead | PoolBase CUDA interface + Device struct | [x] |
   ```
   Update the link path to `completed/task17.1.md` at the same time.

Both changes belong in the completion commit — not in a separate cleanup commit.

### `New task file:` line (when applicable)

If implementation revealed work not covered by any existing task file, create a new task file named `taskX.Y.Z.md` (where `X.Y` is the parent task and `Z` is a sequential integer starting at 1) and reference it in the commit:

```
New task file: task17.3.1.md — pinned memory teardown on device migration
               not covered by task17.3; requires changes to Network destructor
```

---

## Examples

### Completed task

```
[task 17.1] completed: PoolBase CUDA interface + Device struct

Task: task17.1.md → completed/task17.1.md
Status: Completed
Files: src/cpp/include/hodgkin_huxley/pool/pool_base.hpp,
       src/cpp/include/hodgkin_huxley/device.hpp,
       src/cpp/src/device.cpp,
       src/cpp/CMakeLists.txt,
       task17.md (17.1 checked off),
       completed/task17.1.md (git mv from task17.1.md)

Added five virtual methods to PoolBase with CPU no-op defaults. Device struct
follows PyTorch conventions; cuda_device_count() returns 0 on non-CUDA builds
via HH_USE_CUDA guard in device.cpp. device.cpp added to CMakeLists.txt library
sources.
```

### Partial commit

```
[task 17.3] partial: PoolManager CUDA routing

Task: task17.3.md
Status: Partial — assign_to_device() and synchronize_cuda() added to
        PoolManager; on_cuda() query implemented. Pinned memory allocation
        in Network::simulate_with_descriptors and RegionalNetwork::to()
        not yet done.
Files: src/cpp/include/hodgkin_huxley/network/pool_manager.hpp,
       src/cpp/src/network/pool_manager.cpp

Chose to keep use_cuda_ flag on PoolManager rather than Network to avoid
threading it through the Network public API. CUDA pool headers included
conditionally — non-CUDA builds see no change in compile time.
```

### Commit that creates a new sub-task file

```
[task 17.5] partial: CudaHHPool step kernel + scatter/gather

Task: task17.5.md
Status: Partial — hh_step_kernel and scatter/gather via per-neuron copy
        kernel implemented and compiling. CudaIzPool not yet started.
Files: src/cpp/include/hodgkin_huxley/cuda_hh_pool.hpp,
       src/cpp/src/cuda_hh_pool.cu

Used a small gather kernel for scatter_voltages rather than cudaMemcpy2D —
net_idx_ is not guaranteed contiguous so the gather kernel is simpler and
avoids an extra host-side index sort. Alloc uses cudaMalloc with explicit
cudaSetDevice guard; migrate_to_device requires a host mirror which was not
in the original spec.

New task file: task17.5.1.md — host mirror array for migrate_to_device not
               covered by task17.5; needed to support device-to-device copy
               without a cudaDeviceEnablePeerAccess check.
```

---

## Naming new sub-task files

When implementation uncovers work not in any task file:

1. Name it `taskX.Y.Z.md` where `X.Y` is the task that revealed the gap and `Z` starts at `1`.
2. If a second gap is found in the same task in a later commit, use `Z = 2`, and so on.
3. The new file follows the same format as other task files (Role, Status, Depends on, What to implement, Key files, Contract).
4. Reference the new file in both the commit message and in the parent task file under a **"Sub-tasks discovered during implementation"** section.

---

## Quick reference

```
[task X.Y] completed|partial: <summary ≤72 chars>

Task: taskX.Y.md[→ completed/taskX.Y.md] [, taskX.Y.Z.md (new)]
Status: Completed | Partial — <what done, what not>
Files: <file1>, <file2>, ...
       [task17.md (X.Y checked off)]
       [completed/taskX.Y.md (git mv from taskX.Y.md)]

<design decisions / constraints / deferred items>

[New task file: taskX.Y.Z.md — <reason>]
```

**On task completion, always:**
1. `git mv taskX.Y.md completed/taskX.Y.md`
2. Mark `[x]` in `task17.md` and update the link to `completed/taskX.Y.md`
3. Include both in the same commit as the implementation
