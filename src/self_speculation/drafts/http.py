"""HTTP side-channel adapters for D3-capable inference services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING
from urllib.parse import quote

from .base import DraftBundle, DraftReceipt, DraftRequest, DraftVerificationOutcome
from .callable import normalize_draft_receipt

if TYPE_CHECKING:
    import httpx


class DraftFeedbackHTTPError(RuntimeError):
    pass


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "HTTPDraftFeedback requires: pip install 'self-speculation[http]'"
        ) from error
    return httpx


def _boundary_payload(draft: DraftRequest) -> dict[str, Any] | None:
    if draft.boundary is None:
        return None
    return {
        "text": draft.boundary.text,
        "token_ids": list(draft.boundary.token_ids),
    }


def _draft_payload(draft: DraftRequest) -> dict[str, Any]:
    return {
        "text": draft.text,
        "token_ids": list(draft.token_ids),
        "boundary": _boundary_payload(draft),
        "prompt_token_count": draft.prompt_token_count,
        "tool_calls": [
            {
                "name": call.name,
                "arguments": call.arguments,
                "id": call.call_id,
                "index": call.index,
                "format": call.format,
                "raw": call.raw,
            }
            for call in draft.tool_calls
        ],
        "metadata": dict(draft.metadata),
    }


class HTTPDraftFeedback:
    """Replace request-scoped draft candidates through a sidecar.

    The default contract is ``POST /drafts`` followed by
    ``DELETE /drafts/{request_id}``. Subclasses can override payload methods for
    an engine-specific wire format without changing the controller.
    """

    def __init__(
        self,
        base_url: str,
        *,
        submit_path: str = "/drafts",
        clear_path: str = "/drafts/{request_id}",
        clear_method: str = "DELETE",
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = 30.0,
        client: "httpx.AsyncClient | None" = None,
        name: str = "http-draft-feedback",
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not submit_path.startswith("/") or not clear_path.startswith("/"):
            raise ValueError("HTTP draft paths must start with '/'")
        if not clear_method.strip():
            raise ValueError("clear_method must not be empty")
        module = _httpx()
        self.base_url = base_url.rstrip("/")
        self.submit_path = submit_path
        self.clear_path = clear_path
        self.clear_method = clear_method.upper()
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.name = name
        self._client = client or module.AsyncClient()
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def submit_payload(self, bundle: DraftBundle) -> Mapping[str, Any]:
        return {
            "request_id": bundle.request_id,
            "drafts": [_draft_payload(draft) for draft in bundle.drafts],
            "metadata": dict(bundle.metadata),
        }

    def clear_payload(self, request_id: str) -> Mapping[str, Any] | None:
        return None

    def _url(self, path: str, request_id: str | None = None) -> str:
        if request_id is not None:
            path = path.replace("{request_id}", quote(request_id, safe=""))
        return self.base_url + path

    @staticmethod
    def _response_payload(response: "httpx.Response") -> Any:
        if not response.content:
            return None
        try:
            payload = response.json()
        except ValueError as error:
            raise DraftFeedbackHTTPError("draft sidecar returned invalid JSON") from error
        if isinstance(payload, Mapping):
            status = str(payload.get("status", "")).lower()
            if payload.get("error") or status in {"error", "failed", "failure"}:
                raise DraftFeedbackHTTPError(str(payload.get("error") or payload))
        return payload

    async def submit(self, bundle: DraftBundle) -> DraftReceipt:
        response = await self._client.post(
            self._url(self.submit_path),
            json=dict(self.submit_payload(bundle)),
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = self._response_payload(response)
        return normalize_draft_receipt(payload, bundle)

    async def clear(
        self, request_id: str
    ) -> DraftVerificationOutcome | None:
        payload = self.clear_payload(request_id)
        request_kwargs: dict[str, Any] = {
            "headers": self._headers(),
            "timeout": self.timeout,
        }
        if payload is not None:
            request_kwargs["json"] = dict(payload)
        response = await self._client.request(
            self.clear_method,
            self._url(self.clear_path, request_id),
            **request_kwargs,
        )
        response.raise_for_status()
        response_payload = self._response_payload(response)
        if not isinstance(response_payload, Mapping):
            return None
        verification = response_payload.get("verification")
        if not isinstance(verification, Mapping):
            return None
        return DraftVerificationOutcome.from_mapping(
            request_id,
            verification,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "HTTPDraftFeedback":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()


class SporkHTTPDraftFeedback(HTTPDraftFeedback):
    """Compatibility adapter for SPORK's original global vLLM sidecar."""

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        kwargs.setdefault("submit_path", "/spork/set_tokens")
        kwargs.setdefault("clear_path", "/spork/clear")
        kwargs.setdefault("clear_method", "POST")
        kwargs.setdefault("name", "spork-http-draft-feedback")
        super().__init__(base_url, **kwargs)

    def submit_payload(self, bundle: DraftBundle) -> Mapping[str, Any]:
        if len(bundle.drafts) != 1:
            raise ValueError("original SPORK feedback accepts one draft only")
        draft = bundle.drafts[0]
        if not draft.token_ids:
            raise ValueError("SPORK HTTP feedback requires tokenized draft content")
        return {
            "request_id": draft.request_id,
            "draft_tokens": list(draft.token_ids),
            "prompt_len": draft.prompt_token_count or 0,
        }
