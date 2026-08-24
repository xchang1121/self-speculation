# self-speculation

`self-speculation` is a small, engine-agnostic library for the reusable core of
[SPORK](https://github.com/baihuajun24/spork): start one speculative streaming
fork from an already-running inference stream, decode that fork into structured
tool calls, and optionally feed the speculative action back to a capable engine
as draft tokens for verified speculative decoding.

The project intentionally excludes benchmark harnesses, datasets, tool
execution, agent loops, and model-specific serving launchers. Its public surface
is organized around four independent extension points:

- a one-method streaming inference-engine protocol;
- a fork controller that starts after the main stream has produced its first
  useful chunk, preserving SPORK's prefix-cache-friendly D1 timing;
- a registry of streaming tool-call parser branches;
- an optional draft-feedback protocol for D3-style engine-side verification.

The refactor is currently being implemented feature by feature. Each completed
feature is tested, committed, and pushed independently.

## Attribution

This repository is derived from SPORK and preserves its Git history and MIT
license. The source repository is tracked as the `upstream` Git remote:

- SPORK source: <https://github.com/baihuajun24/spork>
- working fork used as the starting point: <https://github.com/xchang1121/spork>
- paper: [SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference](https://arxiv.org/abs/2607.03333)

If this project contributes to research results, cite the SPORK paper:

```bibtex
@misc{bai2026spork,
  title         = {SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference},
  author        = {Bai, Huajun and Lv, Weiwei and Zheng, Huichuan and Lu, Youyou and Shu, Jiwu},
  year          = {2026},
  eprint        = {2607.03333},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}
```

See [NOTICE.md](NOTICE.md) for the retained upstream attribution.

## License

MIT. See [LICENSE](LICENSE).
