# Task 27.3: SynapseRef Compaction

**Role:** Team lead  
**Status:** Completed  
**Depends on:** task27.2 (forward-injection spike delivery with SynapseRef)

---

## What to implement

Narrow `SynapseRef::syn_idx` from `size_t` (8 bytes) to `uint32_t` (4 bytes).
This eliminates the 4-byte padding hole that follows `delay_steps` (uint32_t),
shrinking the struct from 16 → 8 bytes.

`post_from_` holds one `SynapseRef` per synapse, so this halves its memory cost.

## Key files

- `src/cpp/include/hodgkin_huxley/network.hpp` — `SynapseRef` struct
- `src/cpp/src/network.cpp` — `build_injection_tables()` push cast; all downstream uses of `ref.syn_idx` implicitly promote uint32_t → size_t

## Contract

- `sizeof(SynapseRef)` == 8 (was 16)
- All tests pass unchanged
- Savings: 8 bytes × N_synapses in `post_from_`
