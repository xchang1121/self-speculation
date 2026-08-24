from __future__ import annotations

import asyncio
import importlib.util
import unittest

from self_speculation import (
    BoundaryDraftStore,
    DraftBoundary,
    DraftRequest,
    InferenceRequest,
    TransformersEngine,
)


HAS_TRANSFORMERS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers")
)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the transformers extra")
class TransformersEngineTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(7)
        config = GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
        cls.model = GPT2LMHeadModel(config).eval()

    async def test_streams_prompt_tokens_and_usage(self) -> None:
        import torch

        class TinyTokenizer:
            def encode(self, text, *, add_special_tokens=True):
                del text
                return [1, 4, 5] if add_special_tokens else [4, 5]

            def __call__(
                self,
                text,
                *,
                return_tensors=None,
                add_special_tokens=True,
            ):
                del text, return_tensors
                ids = self.encode("", add_special_tokens=add_special_tokens)
                tensor = torch.tensor([ids], dtype=torch.long)
                return {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                }

            def decode(self, token_ids, **kwargs):
                del kwargs
                return "".join(f"t{int(token_id)} " for token_id in token_ids)

            def apply_chat_template(self, messages, **kwargs):
                del kwargs
                return "|".join(str(message["content"]) for message in messages)

        engine = TransformersEngine(self.model, TinyTokenizer())
        chunks = [
            chunk
            async for chunk in engine.stream(
                InferenceRequest(prompt="hello", max_tokens=3)
            )
        ]

        self.assertEqual(sum(len(chunk.token_ids) for chunk in chunks), 3)
        self.assertTrue("".join(chunk.text for chunk in chunks))
        self.assertEqual(chunks[-1].finish_reason, "length")
        self.assertEqual(chunks[-1].usage["prompt_tokens"], 3)
        self.assertEqual(chunks[-1].usage["completion_tokens"], 3)

    async def test_renders_chat_and_accepts_async_renderer(self) -> None:
        class Tokenizer:
            def encode(self, text, *, add_special_tokens=True):
                del add_special_tokens
                return list(range(len(text)))

            def apply_chat_template(self, messages, **kwargs):
                self.seen = (messages, kwargs)
                return "rendered"

        tokenizer = Tokenizer()
        engine = TransformersEngine(self.model, tokenizer)
        request = InferenceRequest(
            messages=({"role": "user", "content": "hi"},),
            tools=({"type": "function", "function": {"name": "f"}},),
        )
        self.assertEqual(await engine.render_prompt(request), "rendered")
        self.assertEqual(tokenizer.seen[1]["tools"][0]["type"], "function")

        async def render(_: InferenceRequest) -> str:
            await asyncio.sleep(0)
            return "custom"

        custom = TransformersEngine(self.model, tokenizer, prompt_renderer=render)
        self.assertEqual(await custom.render_prompt(request), "custom")

    async def test_verifies_boundary_draft_with_fewer_target_forwards(self) -> None:
        import torch

        class NumericTokenizer:
            def encode(self, text, *, add_special_tokens=True):
                del text
                return [1, 4, 5] if add_special_tokens else [4, 5]

            def __call__(self, text, *, return_tensors=None, add_special_tokens=True):
                del text, return_tensors
                ids = self.encode("", add_special_tokens=add_special_tokens)
                tensor = torch.tensor([ids], dtype=torch.long)
                return {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                }

            def decode(self, token_ids, **kwargs):
                del kwargs
                return "".join(f"t{int(token_id)} " for token_id in token_ids)

        tokenizer = NumericTokenizer()
        inputs = tokenizer("prompt", return_tensors="pt")

        baseline_calls = 0

        def baseline_hook(*args):
            nonlocal baseline_calls
            del args
            baseline_calls += 1

        handle = self.model.register_forward_hook(baseline_hook)
        try:
            baseline = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
            )
        finally:
            handle.remove()

        generated = tuple(int(token) for token in baseline[0, 3:].tolist())
        self.assertEqual(len(generated), 4)
        store = BoundaryDraftStore(max_draft_tokens=2)
        store.register(
            DraftRequest(
                request_id="transformers-d3",
                token_ids=generated[1:3],
                boundary=DraftBoundary(token_ids=(generated[0],)),
                prompt_token_count=3,
            )
        )
        engine = TransformersEngine(
            self.model,
            tokenizer,
            draft_store=store,
            max_draft_tokens=2,
        )

        assisted_calls = 0

        def assisted_hook(*args):
            nonlocal assisted_calls
            del args
            assisted_calls += 1

        handle = self.model.register_forward_hook(assisted_hook)
        try:
            chunks = [
                chunk
                async for chunk in engine.stream(
                    InferenceRequest(
                        prompt="prompt",
                        request_id="transformers-d3",
                        max_tokens=4,
                    )
                )
            ]
        finally:
            handle.remove()

        streamed_ids = tuple(
            token_id for chunk in chunks for token_id in chunk.token_ids
        )
        self.assertEqual(streamed_ids, generated)
        self.assertEqual(store.snapshot().injections, 1)
        self.assertLess(assisted_calls, baseline_calls)
        await engine.clear("transformers-d3")
        self.assertEqual(store.snapshot().active_requests, 0)


if __name__ == "__main__":
    unittest.main()
