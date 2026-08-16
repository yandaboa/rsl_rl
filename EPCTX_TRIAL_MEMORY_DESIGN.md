# Episode-Context + Trial Memory (K=2) — design spec

Goal: extend the **stable** episode-context line (the code that ran Tillicum 232336) with
cross-episode memory, WITHOUT reintroducing the rejected trial-wait update semantics and WITHOUT
touching the byte-identical core PPO files.

Target config: K=2 episodes/trial, T=80 (episode), L=80 (context), num_steps_per_env=32,
M=8 memory tokens **prepended to the trunk's sequence**. BC→RL on the biased-state curriculum env
with `resample_period="trial"` (bias = per-trial latent z, held fixed across the 2 episodes of a
trial).

**Design revision (2026-08-15, user-mandated).** The first implementation put the memory behind a
gated cross-attention read layer (`MemoryReadLayer`, bit-exact identity at init). That is REJECTED
and removed. The memory tokens are now ordinary rows of the sequence the trunk already runs over,
and the writer is one combined self+cross attention. Two consequences follow, both accepted:

* **Identity at init is gone.** Prepending M rows perturbs a loaded BC trunk from step 0. `z_init`
  and `memory_pos_embed` keep the small `std=0.02` init so the perturbation is small, but it is not
  zero and there is no test asserting it is. How much the BC policy actually moves is measured by
  the closed-loop gate (roll the loaded ckpt, compare success), not by a unit test.
* The trunk now receives gradient **through the memory rows** (they are real tokens), which is
  intended: the write objective can shape the trunk's representation of the memory.

## Where the pieces live today

- **Stable fork (THIS worktree, branch `stable-ppo`)**: `ActorCriticEpisodeContext`
  (modules/actor_critic_episode_context.py), `EpisodeContextRolloutStorage`
  (storage/episode_context_storage.py), `EpisodeContextPPO` (algorithms/episode_context_ppo.py,
  init_storage + process_env_step overrides ONLY). ppo.py / rollout_storage.py /
  on_policy_runner.py byte-identical to feee047 (test-asserted).
- **Experimental fork (worktree `worktree-local-exploration`)**: the OLD trial-memory machinery
  (`ActorCriticTrialMemory`, its `MemoryWriter`, `MemoryReadLayer`). Left untouched; nothing here
  imports from it except the shared `MultiHeadAttention` / `TrunkBlock` primitives.

## The memory as a trunk prefix

One sequence, one causal pass, on BOTH forward paths:

```
[ z_0 + mpe[0], ..., z_{M-1} + mpe[M-1] | x_0, x_1, ..., x_t ]
```

* memory row `i` attends to memory rows `<= i` and to nothing else (no environment token ever
  feeds back into the memory rows — that is what makes the acting-path shortcut below exact),
* an environment token attends to ALL M memory rows plus its usual reach (causal, clipped at its
  own episode start and at `context_span`),
* memory rows carry `memory_pos_embed` only — no `start_embed`, no episode positional embedding.
  They are not frames.

**Acting path.** At `initialize_state` and at every episode/trial boundary in `reset()`, the
affected environments run a `_prefill_memory`: the M memory input rows go through the trunk in one
small batched pass and each layer's K/V for those rows is stored in a dedicated per-env, per-layer
buffer `[N, M, d]` — separate from the episode KV ring, never evicted, refreshed only by a prefill.
Every `forward_step` then attends over `[memory K/V (M) | valid episode ring slots | itself]`. The
prefill also writes the M readouts into slots `0..M-1` of the hidden history, so
`hidden_history_span = M + T` and step `t`'s readout lands in slot `M + t`.

**Update path (`forward_window` / `forward_sequence`).** Each segment's M memory rows are prepended
as REAL rows: the layout is `[mem_seg0 (M) | mem_seg1 (M) | ... | prefix | window]` with a mask that
(a) keeps memory block `k` causal within itself and blind to everything else, (b) lets an episode
token of segment `k` see memory block `k` plus its usual same-episode earlier tokens. Outputs are
sliced back to the window rows. `memory_tokens=0` skips all of this and is byte-identical to the
memory-free policy (test-asserted).

## The writer

`MemoryTokenWriter` (in actor_critic_episode_context.py), `Z_new = G(H)`:

```
q     = LN_q(H[:, :M])          # the memory rows' trunk OUTPUTS of the pass that just ended
k, v  = proj(LN_kv(H))          # ALL of H (memory rows + episode readouts), masked by validity
Z     = H[:, :M] + MHA(q, k, v)
Z     = Z + FF(LN_ff(Z))
```

i.e. the self-attention (memory over memory) and the cross-attention (memory over the finished
episode) are the SAME attention, as mandated. Output projections are residual-scaled like the trunk
blocks. There is no artificial terminal token: the writer sees the readout of the final observation
and, through `H[:, :M]`, the memory the episode ran with — which is what carries the recurrence
`Z_{e+1} = G(trunk(Z_e), episode_e)` across a boundary.

On the acting path the write runs inside `reset()` under `no_grad` on the done environments'
`H`, is detached into `Z`, and is followed by (trial reset →) history clear → prefill. Ordering
contract with `EpisodeContextPPO.process_env_step`: it calls `get_episode_hidden(done_ids)` BEFORE
`policy.reset`, and that returns the full `[M + T]` snapshot of the finished pass — the very same
tensor the acting write consumes, which is why the update can reproduce it exactly.

## Update-time semantics (unchanged from the first design)

Old (rejected earlier line): memory RECOMPUTED per epoch by the latest trunk — no-grad checkpoint
sweep over raw stored trajectories + in-graph re-forward of the full source episode per pair. That
is what required completed trials (the "wait" coupling) and doubled update compute.

Kept: **cached frozen hidden states + in-graph writer.**
- During collection, at each episode end, snapshot that pass's H (detached, `[M + T, d]`) into
  storage, associated per env. Any row of episode e>0 of a trial references the H of episode e-1 —
  which ALWAYS exists by the time the row is collected (no waiting, ever).
- During update, for each window segment: `Z_seg = z_init` (trial's episode 0) or
  `Z_seg = G(H_source)` computed **in-graph** with the CURRENT writer weights, where `H_source` is
  the cached, detached snapshot (the "stop grad on memory" is on H).
- Gradient therefore trains: `MemoryTokenWriter`, `memory_pos_embed`, the trunk (both via its own
  rows and via processing the M memory rows), and `z_init` — the last one through the SOURCE-FREE
  segments only, since the writer queries the cached H rather than `z_init`. The source episode's
  trunk gets no gradient through the memory path (it still trains from its own PPO rows).
- Costs accepted: H is stale by up to ~ceil((T+steps)/steps) ≈ 4 updates. Benefits: ring buffer
  stays `steps+min(L,T)` = 112 (no episode-1 raw obs needed at update), update compute grows only by
  the `n_seg * M` extra rows of the pass plus the tiny `M x (M+T)` writer attention, and the stock
  recurrent-PPO interface still carries everything (prefix AND memory ride in the hidden_states slot).

## Model changes (ActorCriticEpisodeContext)

- New kwargs: `memory_tokens: int = 0` (0 = OFF, exact current behavior — default-off is
  test-asserted) and `episodes_per_trial: int = 2` (K; used only for z_init reset bookkeeping — the
  recurrence supports general K). There is NO `memory_read_layer` kwarg (removed end to end).
- New submodules when `memory_tokens > 0`: `z_init` `[M, d]`, `memory_pos_embed` `[M, d]`,
  `MemoryTokenWriter`. Nothing else; no gates, no buffers.
- New methods: `memory_readout(Z)` (the trunk outputs of the M rows, i.e. rows `0..M-1` of H —
  what a from-scratch reconstruction needs), `write_memory(H, mask)`, `memory_from_prefix(prefix)`,
  `get_episode_hidden(env_ids)`. `memory_gate_values()` is gone.
- Collection path: `forward_step` attends over the cached memory K/V; `reset(dones, trial_dones)`
  → write → trial restore → clear history → prefill.

## Storage changes (EpisodeContextRolloutStorage)

- Capture at collection (driven from the PPO subclass): per-env, per-done H snapshots
  (`hidden_span = M + T`) + per-row `episode_idx_in_trial`. Multiple dones per env per 32-step
  window are possible (early termination on success/off-table), so snapshots are keyed per segment,
  not one per env per rollout.
- `recurrent_mini_batch_generator` additionally yields, through the hidden_states tuple slot
  (alongside the prefix): per-segment `H_source` (padded + mask), per-segment `has_source` and the
  explicit `memory_segments`. The policy builds the memory in-graph from these.
- Ring buffer UNCHANGED (112 slots, normalized frames).

## PPO changes

- `EpisodeContextPPO.process_env_step` (SUBCLASS override — ppo.py the file stays byte-identical):
  reads `infos["trial_done"]` (from uwlab's TrialVecEnvWrapper; falls back to `trial_dones=dones`
  when absent), snapshots `H` into storage BEFORE the reset, calls `policy.reset(dones,
  trial_dones=...)`. No terminal-token call.
- `update()` itself stays the stock one; the memory build happens inside the policy's
  `forward_window` consumption of the extended hidden_states tuple (same trick as the prefix).

## Tests (tests/test_epctx_trial_memory.py, tests/test_epctx_memory_pipeline.py)

1. Default-off equivalence: `memory_tokens=0` → no parameter, no state, no behavior change.
2. Acting-vs-batched parity: prefill + `forward_step` sequence == `forward_sequence` ==
   `forward_window` with the same Z prepended (~1e-15 in float64).
3. Segment isolation: two segments in one pass, neither one's memory reaches the other's rows, and a
   segment's rows equal a pass over that segment alone.
4. Reset/trial bookkeeping: done → `Z = G(H)` (≠ z_init) + history cleared + prefill refreshed (the
   next step's output changes); trial done → `Z = z_init` + prefill.
5. Grad flow: a segment-1 loss trains the writer, `memory_pos_embed` and the trunk block parameters;
   `H_source` stays grad-free; `z_init` is trained by a source-free segment (see the note above).
   A leaf `Z` receives gradient, i.e. the trunk really backpropagates into the memory rows.
6. End-to-end through `EpisodeContextPPO` + storage: epoch-0 canary (ratio == 1) across episode AND
   trial boundaries inside windows, plus a from-scratch replay of the whole history rebuilt by hand
   (`memory_readout(Z)` + episode readouts → `G`) with a memory-chain-cut vacuity guard.
7. Byte-identity of ppo.py / rollout_storage.py / on_policy_runner.py (existing test, still green).

## Uwlab side (stage 2, separate task)

- New env cfg: EpisodeContext env + `advance_trial` event (K=2) + BiasedObjectPoses
  `resample_period="trial"`; registration `...-BiasedState-EpisodeContext-TrialMemory-Curriculum-v0`.
- Runner cfg: same PPO recipe as EpisodeContextSharedCriticPPORunnerCfg (lr 1e-4 adaptive, kl 0.01,
  5x4, steps 32, freeze_obs_norm=True, critic_warmup_iters=50) + policy `memory_tokens=8`.
  critic_design: shared_trunk (validated by 232336).
- BC→RL: the existing gated BC ckpt `init_weights/episode_context/bc_epctx_peg40k_L80_ep021.pt`
  still loads (the memory params are simply missing from it and must be tolerated by train.py's
  strict-load handling, like the critic-only tolerance), but it does NOT reproduce the BC policy
  exactly any more — the M prefix rows perturb it. Gate the loaded policy closed-loop before
  starting RL; if the drop is material, a short BC re-run WITH the memory tokens is the fallback.
