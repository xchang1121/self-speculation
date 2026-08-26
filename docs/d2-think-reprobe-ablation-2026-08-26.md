# D2 think-mode re-probe ablation (2026-08-26)

## Decision

**Do not add D2 re-probing to the product path.** The released SPORK policy
demonstrated the intended accuracy mechanism, but failed the preregistered
probe-efficiency gate.

On six parseable local Qwen tool turns, D1+D2 recovered two exact actions that
the token-1 D1 probe missed, lost no D1 hit, and retained 13.70 s / 2.76 s of
optimistic no-contention decode runway. It required 3.17 probes per turn and
2.04 times D1's probe tokens, above the frozen limits of 2.0 and 1.75 times.

This result does not contradict the earlier DeepSeek D2 rejection. The remote
recordings had no sufficiently long safe reasoning prefix. This experiment
supplied D2's actual prerequisites: a local think-mode model, growing token
prefixes, and selected-token log-probabilities.

## Frozen method

The preregistration is
[`d2-think-reprobe-preregister-2026-08-26.md`](d2-think-reprobe-preregister-2026-08-26.md).
The implementation reference was `baihuajun24/spork` commit
`31d5ab6f0740d5b5aa26e6a745dc97bcff5139a3`:

- probe after main token 1;
- retry after 50 additional main tokens, snapped to a sentence boundary or
  forced after 30 more tokens;
- at most five 100-token probes;
- confidence is the minimum selected-token probability over positions 2–21;
- threshold `0.90`, first qualifying parseable probe commits;
- exact normalized name and arguments are the only hit.

Runtime:

- Qwen GGUF: `Qwen/Qwen3-1.7B-GGUF:Q8_0`, revision
  `90862c4b9d2787eaed51d12237eafdfe7c5f6077`;
- prompt tokenizer: `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
- inference server: official `llama.cpp` winget build 10615, commit
  `f280b2698`, Vulkan offload on the RTX 4060 Laptop GPU;
- thinking enabled, greedy decode, seed 42, 8192-token context;
- recorder/analyzer: `examples/d2_think_tape_ablation.py`.

The input adapter structurally converts OpenAI text-content blocks to strings
before applying Qwen's text-only chat template. Assistant tool-call and tool
result fields remain intact. A smoke replay caught this boundary issue before
the full recording.

## Recording integrity

| Artifact | SHA-256 |
|---|---|
| `deepseek-mock-deterministic.json` | `e6c150129262b9a4af4d0f3e994498e3404a8077e6e18b5e5e6c761390c204d8` |
| `deepseek-mock-success.json` | `e79552726706cbcbd9b9690975d428e45aeb6efcd2208fea6eeea70daa57eb72` |
| `pattern-learning.json` | `84870722753363eb150a72f4281c75c675450a649e7e07966e1739de83895cd7` |
| private `qwen3-d2-think-20260826.json` | `6b329f6c078fea761addfb029dddb0fd6d8a595e8f4c83b8c3770476757ee311` |

The three tapes yielded nine canonical-hash-deduplicated Actor action
requests. Recording completed with zero errors: 8,646 main tokens, 44 retained
probe attempts / 1,122 probe tokens, and six local main tool calls. Raw prompts,
reasoning, arguments, and outputs remain outside git.

Only 0/6 local Qwen actions exactly matched a recorded DeepSeek action. D1/D2
accuracy is therefore correctly measured against each turn's same local main
decode, as in SPORK, but this small-model run is mechanism evidence rather than
a portability or production-quality result.

## D1 versus released D1+D2

| Metric | D1 | D1+D2 (`theta=0.90`) |
|---|---:|---:|
| Eligible local main tool turns | 6 | 6 |
| Speculative dispatches | 5 | 4 |
| Exact hits | 0 | 2 |
| Precision among dispatches | 0% | 50% |
| Probe attempts | 6 | 19 (3.17/turn) |
| Probe tokens | 225 | 460 (2.044x) |
| D1 misses recovered / D1 hits lost | — | 2 / 0 |
| Recovered optimistic runway | — | 13,698.6 ms; 2,762.2 ms |
| Any-retry exact oracle | — | 5 |

The five-hit oracle versus two committed hits shows that the confidence gate is
not a correctness oracle. It both rejected later exact low-confidence calls and
committed high-confidence calls that the main decode later changed.

### Preregistered gates

| Gate | Result |
|---|---|
| At least six eligible main calls | pass (6) |
| No fewer hits, no lost D1 hit, at least one recovery | pass (0 → 2; lost 0) |
| Dispatch precision does not decrease | pass (0% → 50%) |
| Mean attempts <= 2.0 and probe-token ratio <= 1.75 | **fail (3.17; 2.044x)** |
| Every recovered hit has at least 25 ms runway | pass |

Overall product gate: **fail**.

Threshold sensitivity does not rescue efficiency:

| Threshold | Exact / dispatch | Mean attempts | Probe tokens / D1 |
|---:|---:|---:|---:|
| 0.85 | 2 / 5 | 3.00 | 444 / 1.973x |
| 0.90 | 2 / 4 | 3.17 | 460 / 2.044x |
| 0.95 | 2 / 3 | 3.67 | 525 / 2.333x |

## D3 boundary-draft reuse

The same recording was token-aligned at Qwen's
`<tool_call>\n{"name": "` boundary. Only parseable probes that would have
finished before that boundary were eligible. Accepted-prefix tokens equal the
target steps saved by a single boundary proposal in this deterministic proxy.

| Policy | Accepted target tokens (of 131) | Probe tokens | Marginal cost versus D1 |
|---|---:|---:|---:|
| D1 | 41 (31.3%) | 225 | — |
| D1+D2, stop on first `theta=0.90` commit | 53 (40.5%) | 460 | +12 accepted for +235 probe tokens (19.58:1) |
| Continue every recorded re-probe; latest available | 76 (58.0%) | 767 | intentionally non-product upper variant |
| Available parseable oracle | 76 (58.0%) | — | diagnostic only |

D2 can improve the draft supplied to Actor speculative decoding even when its
tool execution is rejected. Here the marginal exchange is still strongly
negative: 19.6 extra probe tokens per one additional accepted target token.

The upstream adaptive scheduler's 20-token Phase-1 abort was also replayed
without changing any predictions. It reduces D2 probe tokens from 460 to 349
(1.551x D1), but leaves 3.17 requests per turn and costs 124 extra probe tokens
for the same 12 additional accepted target tokens (10.33:1). It therefore does
not pass the request-count gate or justify a product patch.

## Reproduction

Record (private output path omitted):

```powershell
python examples/d2_think_tape_ablation.py record `
  --tape <deepseek-mock-deterministic.json> `
  --tape <deepseek-mock-success.json> `
  --tape <pattern-learning.json> `
  --output <private-qwen-recording.json> --resume
```

Analyze D1, D2, threshold sensitivity, and D3 boundary reuse:

```powershell
python examples/d2_think_tape_ablation.py analyze `
  --recording <private-qwen-recording.json> --d3
```

## Product consequence

No Pi plugin or inference control-plane behavior changed. The committed
recorder and report preserve the positive mechanism signal and the negative
cost boundary. A future delayed-first-probe policy must be frozen and validated
on disjoint requests; the favorable late snapshots in this recording cannot be
selected post hoc as product evidence.
