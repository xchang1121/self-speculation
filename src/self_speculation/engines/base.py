"""The minimal protocol required to plug in an inference engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from ..models import EngineCapabilities, InferenceRequest, StreamChunk


class EngineCapabilityError(ValueError):
    """Raised before dispatch when an engine cannot serve a request form."""


@runtime_checkable
class InferenceEngine(Protocol):
    """Any engine with this one async-stream method can be forked."""

    name: str
    capabilities: EngineCapabilities

    def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        ...


def validate_request(engine: InferenceEngine, request: InferenceRequest) -> None:
    if not engine.capabilities.supports(request):
        raise EngineCapabilityError(
            f"engine {engine.name!r} does not support {request.input_mode} requests"
        )
