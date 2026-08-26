# Low-confidence Phase-1 draft reuse feasibility screen (2026-08-26)

## Decision

**Reject generating a low-confidence D2 Phase-1 prefix solely for target-side
speculative decoding. Do not add this policy to Pi or the inference control
plane.** The safest candidate ordering reduces simulated target steps, but it
spends 100 same-model probe tokens to save only 19 target steps (5.263 probe
tokens per saved step) and has only five eligible turns.

The reusable analyzer and regression tests are retained. They make a useful
distinction that the action-level D2 analysis could not express: an
unparseable, low-confidence prefix may still be a valid token draft, but its
generation cost must be charged even when it is never eligible to execute as a
tool call.

## Evidence boundary

This is a retrospective feasibility screen on the already observed I27
recording, not a new holdout and not independent confirmation. I27 explicitly
identified its actually generated 20-token D2 prefixes as the only remaining
causal candidate. Before spending another disjoint holdout, this screen asks
whether that candidate can pass even a same-model compute-neutrality bound.
It cannot, so no new model output was generated and no post-hoc product policy
was fitted.

The source recording remains the frozen Claw-SWE I27 artifact:

| Artifact | SHA-256 |
|---|---|
| private `qwen3-d2-bounded-claw-holdout-20260826.json` | `58e9433307fb6c1316732a566a7147bf817c6a0eb5f146e30fc882f698814504` |

Runtime, tokenizer, model, request manifest, and action policy are unchanged
from I27. No DeepSeek action, gold patch, or future target token is used to
select a prefix.

## Causal replay policy

For each local main tool turn with a token boundary:

1. retain the complete, parseable D1 draft only if it finished before the
   actor's tool-call boundary;
2. consider the second recorded probe only when the frozen I27 policy actually
   attempted it after low-confidence D1;
3. retain at most its first 20 raw token IDs, without requiring a parseable
   call or action confidence;
4. estimate its boundary arrival by retaining all measured fixed request cost
   and removing only generation time after token 20;
5. replay the existing serial target-verifier contract at `K=28`.

The primary order is D1 then Phase-1. This preserves the already admitted D1
branch and uses the raw prefix only as fallback. Phase-1 then D1 is reported as
a diagnostic because it can reject a good D1 tail before fallback.

The feasibility gates require at least six eligible turns, zero per-turn target
step regressions, a positive pooled target-step gain, no more than one
same-model probe token per target step saved, and no more than 1.25x proposed
tokens. These are screening gates, not an independent preregistered holdout
claim.

## Result

All five attempted Phase-1 prefixes are conservatively available before their
tool boundary.

| `K=28` serial policy | Target steps | Proposals | Proposed tokens | Accepted tokens |
|---|---:|---:|---:|---:|
| D1 | 116 | 5 | 118 | 28 |
| D1 then Phase-1 | 97 | 6 | 137 | 47 |
| Phase-1 then D1 | 97 | 6 | 116 | 47 |

D1 then Phase-1 saves 19 target steps and raises accepted draft tokens by 19.
Its proposed-token ratio is 1.161x and it has zero per-turn target-step
regressions. The reverse order reaches the same pooled target-step count with
fewer proposed tokens, but regresses the `fmtlib__fmt-2310` turn by one target step;
it is therefore not the safe ordering.

The decisive cost is upstream of verification:

| Metric | Result |
|---|---:|
| Eligible policy turns | 5 |
| Prefixes available before boundary | 5 |
| Additional Phase-1 probe tokens | 100 |
| Target steps saved by safe ordering | 19 |
| Probe tokens per target step saved | **5.263** |
| Per-turn target-step regressions | 0 |

Gate result: validity **fails** (`5 < 6`); target-step safety passes; positive
gain passes; proposed-token cost passes (`1.161x <= 1.25x`); compute
neutrality **fails** (`5.263 > 1.0`). Overall result: **fail**.

If a Phase-1 prefix had already been generated for an independently profitable
action policy, D1-first fallback could recycle that sunk work at zero marginal
draft-generation cost. I27 rejected that action policy, however, so this
counterfactual does not justify generating the prefix or adding a product
path.

## Reproduction and verification

```powershell
python examples/d2_think_tape_ablation.py analyze `
  --recording <private-i27-recording.json> --d3
```

The machine-readable result is
`d3_boundary_reuse.low_confidence_phase1_draft_reuse_k28`. The analyzer commit
is `5251a03` and includes conservative prefix-arrival accounting, both serial
orders, explicit gates, and tests proving that raw unparseable tokens are used
only after the recorded low-confidence retry decision.

Verification passed: 158 unit tests, `compileall`, `pip check`, and
`git diff --check`.

## Product consequence

No runtime, protocol, Pi plugin, action threshold, source ordering, or engine
behavior changes. No fresh holdout is consumed. Future candidates should avoid
serial same-model regeneration and instead exploit work already produced by a
profitable source, or use verifier-native signals whose value can be measured
without an additional autoregressive probe.
