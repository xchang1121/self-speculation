# Bounded low-confidence D2 holdout preregistration (2026-08-26)

## Candidate frozen before holdout generation

I25 established that later think-mode probes can recover an exact action, but
unbounded retry work failed the cost gate. I26 established that replacing D1
with a fixed token-200 start removes short-action coverage. The next candidate
preserves the early draft and conditions exactly one additional request on a
causal uncertainty signal:

1. Launch one complete D1 probe after main token 1. Its parseable tool call is
   retained as a target-decoder candidate regardless of confidence.
2. Compute the released SPORK `min_top1` score: minimum selected-token
   probability over positions 2--21. If D1 is parseable and the score is at
   least `0.90`, commit it and issue no D2 request.
3. Otherwise, issue at most one D2 probe after at least 50 additional main
   tokens, after D1 has completed, snapped to the released sentence boundary
   with the existing 30-token forced bound. If main decode ends first, there is
   no D2 request.
4. D2 uses a 20-token Phase 1. A score below `0.90` aborts without Phase 2; a
   passing span continues up to the complete call. Only a passing parseable D2
   call commits for speculative tool execution.
5. The target-decoder bundle is not a new source. It contains the D1 draft and,
   when D2 commits before the action boundary, D2 first followed by D1 as a
   deduplicated fallback. Existing target cap `K=28` applies.

Exact normalized tool name and arguments against the same local main decode
are the only action hit. Name-only matches and any unselected retry oracle do
not count. The fixed `0.90`, 50-token cadence, 20-token Phase 1, sentence snap,
and selected-token metric come from the released SPORK implementation; only
the hard one-retry budget is new.

This is a portable approximation of the rejection-probability threshold policy
in SpecDec++, but it does not claim to implement that paper's trained
acceptance head. SVIP and AdaEDL use full draft-distribution entropy; the
current SPORK/OpenAI-compatible trace exposes selected-token log probability,
not a faithful full-vocabulary entropy. No entropy approximation or learned
head is introduced.

Research references:

- SPORK paper and released scheduler:
  <https://arxiv.org/abs/2607.03333> and
  <https://github.com/baihuajun24/spork/blob/main/spork_core/adaptive_scheduler.py>
- SpecDec++ threshold stopping: <https://arxiv.org/abs/2405.19715>
- SVIP entropy stopping: <https://arxiv.org/abs/2411.18462>
- AdaEDL entropy lower bound: <https://arxiv.org/abs/2410.18351>

## Discovery screen is not holdout evidence

A read-only retrospective replay over the 16 parseable I25+I26 turns selected
this bounded policy. D1 had 9 exact hits from 15 dispatches; the candidate had
10 from 13, lost zero D1 hits, recovered one, and retained 2,762.2 ms of
optimistic runway. It used 21 probes / 1.3125 per turn and 490 full recorded
probe tokens / 1.263x D1. Applying Phase 1 only to the additional D2 attempts
would reduce that ratio further.

Those outputs were already observed and are explicitly discovery data. They
cannot pass the product gate and their favorable numbers may not be used to
tune any I27 parameter.

## Disjoint Claw-SWE holdout

The holdout is constructed without model output from public
`TokenRhythm/Claw-SWE-Bench`, config `lite`, split `test`, repository revision
`ca9da7416154a31015f43df71dcf742c6725b312`. It uses the eleven unique tasks
already checked into the Pi benchmark's `swe_diverse` and `swe_js` suites.

Each case contains:

- the unchanged system message and `read/bash/edit/write` schemas from the
  first request in `deepseek-mock-deterministic.json` (tape SHA-256
  `e6c150129262b9a4af4d0f3e994498e3404a8077e6e18b5e5e6c761390c204d8`);
- one user message containing the public problem statement followed by the
  exact Pi benchmark suffix: `Work in the checked-out repository and finish
  the implementation. Do not use network access to look up the answer.`;
- no prior action, tool result, model answer, gold patch, or reference call.

Template integrity:

- source request hash:
  `1a1736472720349e55b087dc82a8f3d2839758b528b8506464479ebc975c77c0`;
- canonical system-message SHA-256:
  `6b48440dbf8258009979ee5bf1d839b4cfe2752e9c4570008a634c5b7020836c`;
- canonical tool-schema SHA-256:
  `9fcdbe8dd3e57f8515ecb57dce3432e2e73d7d28a9d06bb196ea864378692097`.

The private, model-output-free case manifest is
`claw-swe-i27-cases-20260826.json`, 114,000 bytes, SHA-256
`cc60221714584dfeb9d3ee261dd4fbb10ee9ca1af145b769d954252679e15bf1`.
Its eleven request hashes have zero overlap with the twenty I25+I26 requests.

| Instance | Problem SHA-256 | Canonical request SHA-256 |
|---|---|---|
| `google__gson-1100` | `9c9098493f160a5ba4a6ae7c1afc0bcf12866ae5999b441506b643bb03019136` | `25c4af8b8a9abcc6e04ee5cdd820162ce468ce9f9a99000197dcd50af427bc29` |
| `caddyserver__caddy-5995` | `16dceb9c000eb8ac6a6b17f96e8ff903eac8388b8ba863b39aa15965090ee0b0` | `f617773507583cc96ee9b9e6c7a2fdf56336242dbb5f4e71539a13af6aeb277a` |
| `burntsushi__ripgrep-2576` | `d3605b6eaf20bc233f84b1f827537fb37e49f8567b99c4a2ebdbca816d724e59` | `27ca1661274c37a5f2fc814e9136774683d5469361bc696d9caa38d72d8031d3` |
| `axios__axios-5316` | `38b0a0cecc18528c658fd2a013a02b7ab1cd9e1d40d7a97a7a0c5ba3ac20b0bc` | `85b02ea3011cc4e4a222ac1080d8d72801a45a46473e6059dfacad62022627da` |
| `fmtlib__fmt-2310` | `bfad59f14f650f36f4b7fdaed8e2f7da08189603acf2adb97b09bd93c35032d1` | `8654c7abf0e5fed26e0cfeeae9bc5d515e8e26fe65ba0a864d5de44d01a90547` |
| `faker-ruby__faker-2705` | `ddaf9e19e0128ad1f1361d1f0c6dc08fe1c08d33f3f4cc56430f3e0b6c52909c` | `adf6d88f920bda8f27465e75d4083a32f80600418c2c51ec37b1215d5e2e01f9` |
| `briannesbitt__carbon-2752` | `92856bb41308774422954a1a5bfcf3d946f0e314004076304d0be407aea019fd` | `bc83eef959400af7ebe566811055446df1bd117329a495339f4828f82975445b` |
| `django__django-11790` | `4482bb11e178b43b0d8d59d373ea8ed2631996c27f7ec51d89d65dad47ab37cb` | `06186180c876b70b146cb93714e35a1b83de9cd31fb17c8b81f0ec768d6a1ced` |
| `preactjs__preact-2757` | `c4a64c16f0197524c52600a28f78e570a172ca47421b25874890028a74e5b394` | `dbdfb03f7ab78aaa3bf32d1b7703587cf2cc394eb68390cd80deaeefc2070b8e` |
| `preactjs__preact-4182` | `d050cd97a3150c5807d1f54772fa5a6422491365045fb8d7c78bbde2270a9246` | `e756ebc22cf43297c230fdf93f880dcb45518a9eaa27803764c104dbe5594bd7` |
| `vuejs__core-11589` | `ea9945de86fb689dd595a44b2ca544815062a644513667522ecb498bf052ede7` | `d7c6cc946e4be2bbc7269d663aa99888c356e2e376e2210623dba6925d41a9b4` |

No local Qwen main, D1, or D2 output for these requests has been generated or
inspected before this preregistration.

## Runtime held fixed

- Qwen GGUF `Qwen/Qwen3-1.7B-GGUF:Q8_0`, revision
  `90862c4b9d2787eaed51d12237eafdfe7c5f6077`;
- tokenizer `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
- official llama.cpp build 10615 / commit `f280b2698`, one 8192-token slot,
  greedy seed 42, thinking enabled;
- main cap 3,072 tokens; each complete probe cap 100 tokens; target draft cap
  `K=28`.

## Acceptance gates

Product work is allowed only if every gate passes at `theta=0.90` on the fixed
eleven-case holdout:

1. At least six requests produce a parseable local main tool call.
2. The bounded policy has no fewer exact action hits than D1, loses no D1 exact
   hit, and recovers at least one D1 miss.
3. Exact-hit precision among committed calls is no lower than D1.
4. Mean probes per eligible turn are at most `1.50`; D1-full plus
   Phase-1-accounted D2 tokens are at most `1.25x` D1 full-probe tokens.
5. Every recovered action has at least 25 ms of optimistic no-contention main
   decode runway after its selected probe completes.
6. With committed D2 first and D1 fallback, target cap `K=28`, D3 accepted
   target tokens do not decrease and simulated target steps do not increase
   versus D1 alone.
7. `0.90` is the decision threshold. `0.85` and `0.95` are sensitivity reports
   only and cannot rescue a failure.

If any gate fails, retain only the generic manifest loader, reproducible
analyzer, tests, and negative report. Do not change Pi runtime or inference
control-plane defaults. If all gates pass, any product patch must still be
minimal, keep D1 and D2 in the unified candidate bundle, preserve explicit user
limits, and pass both repositories' full verification suites.
