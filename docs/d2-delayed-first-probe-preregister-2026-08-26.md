# Delayed-first D2 holdout preregistration (2026-08-26)

## Discovery and candidate

I25 showed that repeated think-mode probes can recover exact local tool calls,
but token-1 start plus up to five retries cost 3.17 attempts per eligible turn
and 2.044 times D1's probe tokens. Favorable predictions appeared only after
the main reasoning had matured.

The next candidate is frozen before generating any holdout model output:

- D1 control: one full probe at main token 1, no pre-dispatch confidence gate.
- Delayed D2: first attempt at main token 200; at most two attempts; the second
  begins after 50 additional main tokens and uses the released sentence snap
  (or the 30-token forced boundary).
- Each D2 attempt uses the official adaptive 20-token Phase-1 span. Minimum
  selected-token probability over positions 2–21 below `0.90` aborts it; a
  passing span continues to the complete call. The first passing parseable call
  commits.
- Exact normalized name and arguments against the same local main decode are
  the only hit. Name-only matches and any-retry oracle matches do not count.
- Model, tokenizer, server build, greedy seed, prompt adapter, context size, and
  tool-call parser remain exactly those frozen in I25.

`200` and the two-attempt cap are discovery-derived and may not be tuned after
holdout results. The policy corresponds to SPORK's existing
`min_tokens_first_probe` and `max_retries` axes; this experiment does not invent
an intent classifier from the answers.

## Disjoint holdout

Use every recorded `deepseek-v4-pro` action request from the following tapes,
canonical-hash deduplicated:

| Tape | SHA-256 |
|---|---|
| `r145-output-informed-rollout-20260823.json` | `31dfc98344482860465407bff990334256f0620a22b099ad9e490ec96239bdb0` |
| `r41-tool-order-20260822.json` | `6980df02763727c2c75a6a38e3e86229b8ecf271b3bff0ed52b2255cea8bce8f` |
| `r53-utility-replacement-20260822.json` | `e3f90d34954f675a944f5b91c2505385e49e1963e970906229ecdf1d9f542e1a` |
| `r56-deadline-dominance-20260822.json` | `b7cdfa957ef1a30f7552febf86edbef54793f29c6fe192240795a156ccaf808e` |
| `r60-runway-live-20260822.json` | `925aeef862e8d39d1be95a86b1aca02822113e07830fdc84c48886d4a840b842` |
| `r65-last-value-20260822.json` | `e2bf8256d179adda77c65ec3d2b21848a2b5f2c4107bb729274dd3a8b73a5ab1` |

Read-only request hashing found 11 requests across seven distinct user-task
texts, with zero request-hash overlap with I25's nine discovery requests. No
Qwen main or probe output from these holdout requests has been generated or
inspected before this preregistration.

## Acceptance gates

Product work is allowed only if every gate passes on the fixed holdout:

1. At least six requests produce a parseable local main tool call.
2. Delayed D2 has no fewer exact hits than D1, loses no D1 exact hit, and
   recovers at least one D1 miss.
3. Exact-hit precision among dispatched delayed-D2 probes is no lower than D1.
4. Mean D2 attempts per eligible turn is at most 2.0 (also a hard policy cap).
5. Phase-1-accounted D2 generated tokens are at most 1.75 times D1 tokens.
6. Every claimed recovered hit completes with at least 25 ms of optimistic
   no-contention main-decode runway.
7. The `0.90` decision stands. `0.85` and `0.95` are sensitivity reports only.

D3 boundary-prefix reuse is reported, including marginal probe tokens per
additional accepted target token, but cannot substitute for a failed action or
probe-efficiency gate.

If any gate fails, retain only the reproducible analyzer/negative report. Do not
add delayed D2 to the Pi plugin or inference control plane.
