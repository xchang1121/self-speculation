"""Optional inference-engine integrations."""

from .sglang import (
    SGLANG_CONTROL_PREFIX,
    SGLangHTTPDraftFeedback,
    SGLangIntegrationError,
    install_sglang_plugin,
)

from .vllm import (
    CLEAR_DRAFT_RPC,
    DRAFT_STATUS_RPC,
    REGISTER_DRAFT_BUNDLE_RPC,
    SelfSpeculationEndpointPlugin,
    VLLMBoundaryProposer,
    VLLMCollectiveRPCDraftFeedback,
    VLLMHTTPDraftFeedback,
    VLLMIntegrationError,
    VLLMRequestNotActiveError,
    install_vllm_request_id_hook,
    install_vllm_http_routes,
    install_vllm_worker_rpc,
)

__all__ = [
    "CLEAR_DRAFT_RPC",
    "DRAFT_STATUS_RPC",
    "REGISTER_DRAFT_BUNDLE_RPC",
    "SGLANG_CONTROL_PREFIX",
    "SGLangHTTPDraftFeedback",
    "SGLangIntegrationError",
    "SelfSpeculationEndpointPlugin",
    "VLLMBoundaryProposer",
    "VLLMCollectiveRPCDraftFeedback",
    "VLLMHTTPDraftFeedback",
    "VLLMIntegrationError",
    "VLLMRequestNotActiveError",
    "install_vllm_request_id_hook",
    "install_vllm_http_routes",
    "install_vllm_worker_rpc",
    "install_sglang_plugin",
]
