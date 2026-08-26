# D3 candidate-agreement cap ablation — 2026-08-26

## Decision

Do not add candidate-agreement clipping to `BoundaryDraftStore`.

Clipping a serial proposal at the first token where compatible candidates
disagree reduces rejected draft work, but it inserts an extra target round when
the first candidate was already exact. On the current width-two tape shapes,
the additional target rounds dominate: a pinned real Transformers verifier was
4.39% slower while producing exactly the same target output.

The offline analyzer and real-model shape harness are retained so this idea is
not repeated without a materially different verifier cost curve.

## Motivation and boundary

Adaptive draft-tree methods optimize expected acceptance length or expected
time gain under a node budget. OPT-Tree searches an adaptive tree for expected
acceptance length; Pruned Candidate Tree retains nodes predicted to contribute
to acceleration; cost-aware draft-tree work makes verifier latency part of the
budget objective. These are parallel tree-verification designs. The current
portable `BoundaryDraftStore` instead offers one linear candidate per target
round, so copying a tree policy without measuring the extra serial round would
be incorrect.

Primary references:

- OPT-Tree: https://aclanthology.org/2025.tacl-1.8/
- Effective Draft Decoder with Pruned Candidate Tree:
  https://aclanthology.org/2025.acl-long.486/
- SpecDec++ adaptive candidate lengths: https://arxiv.org/abs/2405.19715
- Cost-Aware Diffusion Draft Trees: https://arxiv.org/abs/2606.01813

The frozen treatment is deliberately smaller than those systems:

1. Use the current product defaults: Drafter dispatch width two and D3 cap 28.
2. Preserve recorded completion order among the two selected requests.
3. When at least two candidates still match the authoritative generated body,
   propose only their non-empty common next-token prefix.
4. Let the verifier's bonus target token select a branch, then offer the
   surviving suffix on the next round.
5. If the candidates disagree immediately, keep current full-candidate behavior.
6. Never consult the future Actor action to decide whether to clip.

This differs from I13: I13 reordered complete candidates using a prefix medoid;
this experiment leaves ordering unchanged and changes proposal length.

## Strict tape replay

Input is the same three valid private mock/SWE-style tapes, DeepSeek V3
tokenizer revision `e815299b0bcbac849fa540c768ef21845365c9eb`,
`tagged_json`, width two, and cap 28. `deepseek-live.json` remains excluded
because it contains insufficient-balance errors rather than model output.

| Policy | Proposals | Proposed | Accepted | Rejected | Acceptance | Target steps | Steps saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current full candidate | 12 | 198 | 144 | 54 | 72.73% | 278 | 138 |
| Agreement cap | 15 | 183 | 141 | 42 | 77.05% | 281 | 135 |
| Delta | +3 | −15 | −3 | −12 | +4.32 pp | +3 | −3 |

The treatment clips six proposals and avoids 12 rejected tokens, but every
exact-first branch loses one draft-accepted token and needs one more target
round. Under a simple cost model
`round_cost × target_steps + token_cost × proposed_tokens`, clipping needs one
verified draft token to cost more than `3/15 = 0.2` target rounds merely to
break even. The tape has no target-hardware timing, so this mixed result cannot
authorize a runtime change.

## Pinned real Transformers verifier

`transformers_agreement_cap_tape_shapes.py` maps each tape opportunity's exact
candidate lengths, order, same-position token equality, and exact/miss branch
onto one deterministic target continuation. Prompts are not replayed into the
model. One common authoritative token is retained after every mapped action so
max-length termination cannot leave the last proposal token unresolved.

Environment:

- target: `hf-internal-testing/tiny-random-gpt2` revision
  `71034c5d8bde858ff824298bdedc65515b97d2b9`
- target runtime: PyTorch `2.13.0+cpu`, Transformers `5.15.1`
- source tokenizer: the pinned DeepSeek V3 revision above
- 12 opportunities, 11 warm-after-path alternating A/B repetitions
- output token IDs identical for every run

| Policy | Target forwards | Proposed | Accepted | Rejected | Sum of per-opportunity median wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current full candidate | 296 | 198 | 144 | 54 | 1232.135 ms |
| Agreement cap | 299 | 183 | 141 | 42 | 1286.221 ms |
| Delta | +3 | −15 | −3 | −12 | **+54.085 ms / +4.39%** |

The real verifier reproduces the offline proposed/accepted/rejected counts
exactly. Forward counts include the common boundary and post-action sentinel,
so their absolute value differs from the offline target-step proxy while the
policy delta remains +3.

## Reproduction

Run the offline replay:

```powershell
.\.venv\Scripts\python.exe examples\d3_agreement_cap_ablation.py `
  --tape <deepseek-mock-deterministic.json> `
  --tape <deepseek-mock-success.json> `
  --tape <pattern-learning.json> `
  --actor-model deepseek-v4-pro `
  --drafter-model deepseek-v4-flash `
  --tokenizer deepseek-ai/DeepSeek-V3 `
  --revision e815299b0bcbac849fa540c768ef21845365c9eb `
  --drafter-width 2 --max-draft-tokens 28
```

Run the target-verifier shape replay with the same tape arguments:

```powershell
.\.venv\Scripts\python.exe examples\transformers_agreement_cap_tape_shapes.py `
  --tape <...> `
  --actor-model deepseek-v4-pro `
  --drafter-model deepseek-v4-flash `
  --source-revision e815299b0bcbac849fa540c768ef21845365c9eb `
  --revision 71034c5d8bde858ff824298bdedc65515b97d2b9 `
  --drafter-width 2 --max-draft-tokens 28 --repeats 11
```

No product store, endpoint, protocol, or default changed in this ablation.
