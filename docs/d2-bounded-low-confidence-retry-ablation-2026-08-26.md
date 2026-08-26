# Bounded low-confidence D2 holdout ablation (2026-08-26)

## Decision

**Reject the bounded low-confidence D2 policy. Do not add it to the Pi plugin
or inference control plane.** On the frozen Claw-SWE holdout it fails the
minimum-validity, action-recovery, request-count, probe-token, and usable-runway
gates.

Only five of eleven local Qwen main decodes emitted a parseable tool call. D1
missed all five exact calls. Every D1 was below the fixed `0.90` gate, so the
policy made one D2 request on every eligible turn; no D2 passed the gate and no
action was committed. The result was zero recovery at 2.0 probes per eligible
turn and 1.680x D1's tokens after the preregistered Phase-1 accounting.

One complete diagnostic D2 probe did exactly match its local main action with
13,952.0 ms of optimistic runway, but its `min_top1` confidence was only
`0.6320`. It is an oracle hit, not a policy hit. Lowering the threshold after
seeing this holdout would violate the preregistration; even the frozen `0.85`
sensitivity excludes it.

## Frozen policy and evidence boundary

The full policy, inputs, and gates were committed before any holdout model
output in
[`d2-bounded-low-confidence-retry-preregister-2026-08-26.md`](d2-bounded-low-confidence-retry-preregister-2026-08-26.md)
(`6f1a408`):

- one complete D1 probe after main token 1, always retained as a raw target
  draft;
- if D1 is unparseable or `min_top1 < 0.90`, at most one D2 after the released
  50-token/sentence-snap schedule;
- D2 uses a 20-token Phase 1 and commits only if the fixed score passes;
- committed D2 precedes D1 in a deduplicated `K=28` target-verifier bundle;
- exact normalized name and arguments against the same local main decode are
  the only action hit.

The generic manifest loader, causal recorder stop, configurable frozen cost
gates, and serial D2-to-D1 target fallback simulator were committed and pushed
as `cc9b349`. The raw recorder stops only after a parseable probe reaches
`0.95`, the highest sensitivity threshold. Consequently it records the
counterfactual second attempt needed by `theta=0.95` without charging that work
to a lower-threshold policy that would already have stopped.

This candidate used the released SPORK selected-token metric rather than
pretending selected log probability is full-vocabulary entropy. SpecDec++ uses
a trained acceptance head, while SVIP and AdaEDL use draft-distribution
entropy; none of those unavailable signals was approximated post hoc.

## Holdout and recording integrity

The model-output-free case manifest is drawn from public
`TokenRhythm/Claw-SWE-Bench` `lite/test` at revision
`ca9da7416154a31015f43df71dcf742c6725b312`. It contains the eleven unique
tasks in Pi's checked-in `swe_diverse` and `swe_js` suites, with fixed problem
and canonical request hashes listed in the preregistration. It has zero request
overlap with I25/I26.

| Artifact | SHA-256 |
|---|---|
| `deepseek-mock-deterministic.json` schema template | `e6c150129262b9a4af4d0f3e994498e3404a8077e6e18b5e5e6c761390c204d8` |
| private `claw-swe-i27-cases-20260826.json` | `cc60221714584dfeb9d3ee261dd4fbb10ee9ca1af145b769d954252679e15bf1` |
| private `qwen3-d2-bounded-claw-holdout-20260826.json` | `58e9433307fb6c1316732a566a7147bf817c6a0eb5f146e30fc882f698814504` |

The final recording is 1,358,912 bytes. It completed 11/11 cases with zero
error turns, 24,782 main tokens, and 21 raw probes / 742 raw probe tokens. The
smoke was the first case in this same output; the full command resumed it, so
no case was generated twice. Raw public problem text, model reasoning, tool
arguments, and outputs remain outside git.

Runtime remained fixed at Qwen3-1.7B Q8 GGUF revision
`90862c4b9d2787eaed51d12237eafdfe7c5f6077`, tokenizer revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, official llama.cpp build 10615 /
`f280b2698`, Vulkan, one 8192-token slot, greedy seed 42, and a 3,072-token
main cap.

This holdout intentionally has no recorded reference action. It measures the
self-speculation mechanism against each request's same local main decode and
does not claim DeepSeek-to-Qwen action portability.

## Action result

| Metric | D1 | Bounded D1+D2 (`theta=0.90`) |
|---|---:|---:|
| Recorded requests | 11 | 11 |
| Eligible local main tool turns | 5 | 5 |
| Speculative dispatches | 5 | 0 |
| Exact hits | 0 | 0 |
| Precision among dispatches | 0% | 0% |
| Policy-selected probes | 5 | 10 (2.0/turn) |
| Full probe tokens | 147 | 317 (2.156x) |
| D1-full + D2 Phase-1 tokens | 147 | 247 (1.680x) |
| D1 misses recovered / D1 hits lost | -- | 0 / 0 |
| Any-recorded-retry exact oracle | -- | 1 |

The five valid action turns were:

| Case | Main | Main tokens | D1 confidence / exact | D2 snapshot | D2 confidence / exact | D2 runway |
|---|---|---:|---:|---:|---:|---:|
| `google__gson-1100` | `edit` | 1,417 | 0.7522 / no | 62 | 0.6320 / **yes** | 13,952.0 ms |
| `axios__axios-5316` | `read` | 1,083 | 0.8068 / no | 85 | 0.5359 / no | 9,867.9 ms |
| `fmtlib__fmt-2310` | `read` | 2,546 | 0.4573 / no | 119 | 0.3412 / no | 26,218.4 ms |
| `faker-ruby__faker-2705` | `read` | 398 | 0.6935 / no | 67 | 0.6087 / no | 3,030.7 ms |
| `django__django-11790` | `read` | 1,059 | 0.6341 / no | 53 | 0.4209 / no | 10,332.8 ms |

Five of the six non-action mains reached the 3,072-token cap; the remaining
one ended after 2,919 tokens without a tool call. This is why the preregistered
minimum of six eligible turns also fails.

Thresholds `0.85`, `0.90`, and `0.95` produce identical action and cost
results. There is therefore no preregistered sensitivity that rescues the
candidate.

### Preregistered gates

| Gate | Result |
|---|---|
| At least six eligible main calls | **fail (5)** |
| No fewer hits, no lost D1 hit, at least one recovery | **fail (0 -> 0; recovered 0)** |
| Dispatch precision does not decrease | pass by equality (0% -> 0%), not a benefit |
| Mean probes <= 1.50 and Phase-1 token ratio <= 1.25 | **fail (2.0; 1.680x)** |
| Every recovered hit has at least 25 ms runway | **fail (no policy recovery)** |
| `K=28` D2-to-D1 bundle does not regress D3 | pass, but unchanged |

Overall product gate: **fail**.

## D3 diagnostic

The preregistered product bundle contains only committed D2 plus D1 fallback.
Because no D2 passed the confidence gate, it is exactly D1:

| K=28 serial policy | Accepted target tokens | Target steps | Proposals | Proposed tokens |
|---|---:|---:|---:|---:|
| D1 | 28 | 116 | 5 | 118 |
| Committed D2 then D1 | 28 | 116 | 5 | 118 |

For diagnosis only, substituting each complete, parseable low-confidence D2
probe raises common-prefix acceptance from 28 to 73 of 144 continuation tokens;
the available parseable oracle is 74. Phase-1 accounting would charge 100
additional probe tokens for the 45-token increase (2.22:1).

That is not a realizable result for this policy: a failed 20-token Phase 1
would not continue to generate the complete D2 call used by that diagnostic.
It motivates a separate experiment on recycling the *actually available first
20 D2 tokens* as a target draft, but cannot rescue I27 or justify product code.

## Reproduction

The first-case smoke and full run used the same output and configuration; the
second command omitted only `--limit 1` and resumed:

```powershell
python examples/d2_think_tape_ablation.py record `
  --case-manifest <private-claw-swe-manifest.json> `
  --output <private-recording.json> --resume `
  --max-retries 2 --retry-token-step 50 --min-tokens-first-probe 1 `
  --stop-after-confident-probe `
  --record-stop-confidence-threshold 0.95 `
  --phase1-span-tokens 20 --phase1-first-probe-full `
  --confidence-threshold 0.90 `
  --gate-max-mean-probe-attempts 1.50 `
  --gate-max-probe-token-ratio 1.25 `
  --require-d3-bundle-gate
```

Independent re-analysis:

```powershell
python examples/d2_think_tape_ablation.py analyze `
  --recording <private-recording.json> --d3
```

## Product consequence

No Pi plugin, engine, sidecar, or control-plane behavior changes. The generic
manifest loader and analyzers remain useful reproducibility infrastructure.
The next candidate, if pursued, must treat low-confidence Phase-1 tokens as a
token-level draft only; it must not lower the action-execution threshold on
this observed holdout or assume a full call exists after an early abort.
