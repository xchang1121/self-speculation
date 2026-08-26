# Delayed-first D2 holdout ablation (2026-08-26)

## Decision

**Reject the token-200 delayed-first D2 policy. Do not add it to the Pi plugin
or inference control plane.** It passes the frozen probe-cost limits only by
removing coverage from short actions, and consequently fails recall,
precision, and usable-recovery gates on the disjoint holdout.

Across ten parseable local Qwen tool turns, the independent token-1 D1 control
hit 9/10 exact calls. Delayed D2 dispatched on only four turns, hit 3/4, lost
six D1 hits, and recovered none. Six eligible main decodes ended after only
22--164 tokens, before the token-200 start. On the four eligible long decodes,
D1 and delayed D2 both hit the same three calls; the remaining `edit` call was
wrong under both policies.

The measured 0.40 attempts per eligible turn and 0.534x D1 probe-token ratio
are therefore not a useful efficiency win: they mostly reflect abstention
before an action candidate exists.

## Frozen method

The policy, holdout, and gates were committed before generating holdout model
output in
[`d2-delayed-first-probe-preregister-2026-08-26.md`](d2-delayed-first-probe-preregister-2026-08-26.md)
(`ec5e0e2`):

- independent D1 control: one full probe at main token 1;
- delayed D2: first probe at token 200, then at most one retry after the
  released 50-token/sentence-snap schedule;
- official SPORK-style Phase 1: selected-token minimum probability over
  positions 2--21, fixed threshold `0.90`;
- a passing parseable call commits; exact normalized tool name and arguments
  against the same local main decode are the only hit;
- Phase-1-accounted tokens determine the frozen efficiency gate.

The recorder executes delayed probes before the independent D1 control so the
control cannot evict the long main prefix from llama.cpp's single-slot cache.
The analyzer uses the independent control while preserving full-probe
accounting for older recordings. That harness was committed and pushed as
`1d8f99e` after 152 unit tests, bytecode compilation, dependency consistency,
and `git diff --check` passed.

Runtime was unchanged from I25:

- Qwen GGUF: `Qwen/Qwen3-1.7B-GGUF:Q8_0`, revision
  `90862c4b9d2787eaed51d12237eafdfe7c5f6077`;
- tokenizer: `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
- official llama.cpp winget build 10615, commit `f280b2698`, Vulkan offload on
  the RTX 4060 Laptop GPU;
- thinking enabled, greedy decode, seed 42, 8192-token context.

## Holdout integrity

| Input tape | SHA-256 |
|---|---|
| `r145-output-informed-rollout-20260823.json` | `31dfc98344482860465407bff990334256f0620a22b099ad9e490ec96239bdb0` |
| `r41-tool-order-20260822.json` | `6980df02763727c2c75a6a38e3e86229b8ecf271b3bff0ed52b2255cea8bce8f` |
| `r53-utility-replacement-20260822.json` | `e3f90d34954f675a944f5b91c2505385e49e1963e970906229ecdf1d9f542e1a` |
| `r56-deadline-dominance-20260822.json` | `b7cdfa957ef1a30f7552febf86edbef54793f29c6fe192240795a156ccaf808e` |
| `r60-runway-live-20260822.json` | `925aeef862e8d39d1be95a86b1aca02822113e07830fdc84c48886d4a840b842` |
| `r65-last-value-20260822.json` | `e2bf8256d179adda77c65ec3d2b21848a2b5f2c4107bb729274dd3a8b73a5ab1` |
| private `qwen3-d2-delayed-holdout-20260826.json` | `925551d39ba9fbe4dfa1250743ea9ee60818c152e0a015dda93231d9774117ec` |

The six tapes yielded 11 canonical-hash-deduplicated Actor requests from seven
user tasks, with zero request-hash overlap with the nine I25 discovery
requests. The one preregistered run completed 11/11 main generations with zero
error turns. It recorded 6,917 main tokens, 11 independent D1 probes / 176
tokens, and ten raw delayed probes / 231 tokens. Raw prompts, reasoning, tool
arguments, and outputs remain outside git.

Ten local main outputs were parseable tool calls, and 8/10 exactly matched the
recorded DeepSeek action. This is substantially better portability evidence
than I25's 0/6, while D1/D2 accuracy remains correctly scored against each
turn's same local main decode.

## Frozen holdout result

| Metric | D1 | Delayed D2 (`theta=0.90`) |
|---|---:|---:|
| Eligible local main tool turns | 10 | 10 |
| Speculative dispatches | 10 | 4 |
| Exact hits | 9 | 3 |
| Precision among dispatches | 90% | 75% |
| Policy-selected probe attempts | 10 | 4 (0.40/turn) |
| Policy probe tokens | 163 | 87 (0.534x) |
| D1 misses recovered / D1 hits lost | -- | 0 / 6 |
| Recovered optimistic runway | -- | none |
| Any-recorded-retry exact oracle | -- | 3 |

All four eligible long turns had a parseable first delayed probe with
confidence above `0.999`; the stop rule therefore selected one attempt for
each. The second recorded probes are useful only for the preregistered oracle
and D3 diagnostics. Threshold sensitivities `0.85`, `0.90`, and `0.95` are
identical, so threshold tuning cannot rescue the candidate.

### Preregistered gates

| Gate | Result |
|---|---|
| At least six eligible main calls | pass (10) |
| No fewer hits, no lost D1 hit, at least one recovery | **fail (9 -> 3; lost 6; recovered 0)** |
| Dispatch precision does not decrease | **fail (90% -> 75%)** |
| Mean attempts <= 2.0 and Phase-1 token ratio <= 1.75 | pass (0.40; 0.534x) |
| Every recovered hit has at least 25 ms runway | **fail (no recovery)** |

Overall product gate: **fail**.

## D3 boundary-draft reuse

The same recording was aligned at Qwen's tool-call boundary. Only parseable
probes that would have completed before the boundary were eligible.

| Policy | Accepted target tokens (of 162) | Probe tokens |
|---|---:|---:|
| Independent D1 | 100 (61.7%) | 163 |
| Delayed D2, stop on first `theta=0.90` commit | 50 (30.9%) | 87 |
| Continue all delayed re-probes; latest available | 50 (30.9%) | 174 |
| Available parseable delayed oracle | 50 (30.9%) | -- |

Delayed D2 loses 50 accepted target tokens relative to D1. Continuing all
re-probes and choosing the available parseable oracle cannot recover any of
them, confirming that the failure is the token-200 coverage boundary rather
than the confidence threshold or retry stop rule.

## Reproduction

Record exactly once on the frozen six-tape holdout (private output path
omitted):

```powershell
python examples/d2_think_tape_ablation.py record `
  --tape <r145.json> --tape <r41.json> --tape <r53.json> `
  --tape <r56.json> --tape <r60.json> --tape <r65.json> `
  --output <private-recording.json> --resume `
  --min-tokens-first-probe 200 --max-retries 2 `
  --phase1-span-tokens 20
```

Analyze action selection, frozen gates, sensitivities, and D3 reuse:

```powershell
python examples/d2_think_tape_ablation.py analyze `
  --recording <private-recording.json> --d3
```

## Product consequence

No Pi plugin or inference control-plane behavior changes. The negative result
rules out a fixed token-200 start as a general replacement for token-1 D1. A
future early-exit policy must preserve a cheap early candidate and condition
additional work on causal evidence available before the action boundary; the
holdout answers must not be used to tune a new cutoff.
