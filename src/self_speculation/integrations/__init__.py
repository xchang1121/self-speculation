"""Optional inference-engine integrations."""

from .vllm import VLLMBoundaryProposer, install_vllm_request_id_hook

__all__ = ["VLLMBoundaryProposer", "install_vllm_request_id_hook"]
