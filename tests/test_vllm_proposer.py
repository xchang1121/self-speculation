from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from self_speculation import (
    VLLMBoundaryProposer,
    install_vllm_request_id_hook,
)


class FakeRunner:
    def __init__(self, drafter, request_ids: list[str]) -> None:
        self.drafter = drafter
        self.input_batch = SimpleNamespace(req_ids=request_ids)

    def propose_draft_token_ids(self, value: str) -> str:
        return value


class VLLMRequestIdHookTest(unittest.TestCase):
    def test_routes_stable_batch_ids_and_installs_once(self) -> None:
        class Drafter:
            request_ids: tuple[str, ...] = ()

            def set_request_ids(self, request_ids: tuple[str, ...]) -> None:
                self.request_ids = request_ids

        drafter = Drafter()
        runner = FakeRunner(drafter, ["main", "main:fork"])
        self.assertTrue(install_vllm_request_id_hook(FakeRunner))
        self.assertFalse(install_vllm_request_id_hook(FakeRunner))
        self.assertEqual(runner.propose_draft_token_ids("result"), "result")
        self.assertEqual(drafter.request_ids, ("main", "main:fork"))

    def test_rejects_an_incompatible_runner(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported vLLM"):
            install_vllm_request_id_hook(type("Unsupported", (), {}))


class VLLMBoundaryProposerTest(unittest.TestCase):
    def proposer(self) -> VLLMBoundaryProposer:
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=4)
        )
        with patch(
            "self_speculation.integrations.vllm.install_vllm_request_id_hook"
        ):
            return VLLMBoundaryProposer(config)

    def test_proposes_for_matching_stable_id_after_batch_reordering(self) -> None:
        proposer = self.proposer()
        proposer.register_draft("main", [20, 21, 22], [10, 11], 2)
        proposer.set_request_ids(("main:fork", "main"))

        proposals = proposer.propose(
            sampled_token_ids=[[7], [8]],
            num_tokens_no_spec=[3, 6],
            token_ids_cpu=[
                [90, 91, 7, 0, 0, 0],
                [90, 91, 5, 10, 11, 20],
            ],
        )

        self.assertEqual(proposals, [[], [21, 22]])
        self.assertEqual(proposer.status()["injections"], 1)

    def test_returns_empty_without_an_exact_request_mapping(self) -> None:
        proposer = self.proposer()
        proposer.register_draft("main", [1], [9])
        self.assertEqual(
            proposer.propose([[1]], [1], [[9]]),
            [[]],
        )

    def test_exposes_request_scoped_cleanup(self) -> None:
        proposer = self.proposer()
        result = proposer.register_draft("main", [1, 2], [9])
        self.assertTrue(result["registered"])
        self.assertTrue(proposer.clear_request("main")["removed"])
        self.assertEqual(proposer.clear_all()["removed"], 0)


if __name__ == "__main__":
    unittest.main()
