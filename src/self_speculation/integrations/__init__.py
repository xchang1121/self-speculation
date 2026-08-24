"""Optional inference-engine integrations."""

from .vllm import (
    CLEAR_DRAFT_RPC,
    DRAFT_STATUS_RPC,
    REGISTER_DRAFT_RPC,
    VLLMBoundaryProposer,
    VLLMCollectiveRPCDraftFeedback,
    VLLMIntegrationError,
    install_vllm_request_id_hook,
    install_vllm_worker_rpc,
)

__all__ = [
    "CLEAR_DRAFT_RPC",
    "DRAFT_STATUS_RPC",
    "REGISTER_DRAFT_RPC",
    "VLLMBoundaryProposer",
    "VLLMCollectiveRPCDraftFeedback",
    "VLLMIntegrationError",
    "install_vllm_request_id_hook",
    "install_vllm_worker_rpc",
]
