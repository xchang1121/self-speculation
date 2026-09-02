"""Format-safe continuations for forcing an Actor stream toward a tool call."""

from __future__ import annotations

from dataclasses import dataclass

from .drafts.formatters import tool_call_syntax
from .models import StreamSnapshot


class ContinuationFormatError(ValueError):
    """Raised when normalized output cannot reproduce the Actor token framing."""


@dataclass(frozen=True, slots=True)
class ReasoningEnvelope:
    """One balanced textual CoT envelope and its canonical exit spacing."""

    name: str
    open: str
    close: str
    line_break_before_close: bool = True
    trailing: str = "\n\n"

    def exit(self, content: str) -> str:
        leading = "\n" if self.line_break_before_close and not content.endswith("\n") else ""
        return leading + self.close + self.trailing


DEFAULT_REASONING_ENVELOPES = (
    ReasoningEnvelope("think_xml", "<think>", "</think>"),
    ReasoningEnvelope("analysis_xml", "<analysis>", "</analysis>"),
    ReasoningEnvelope("reasoning_xml", "<reasoning>", "</reasoning>"),
    ReasoningEnvelope("think_bracket", "[THINK]", "[/THINK]"),
    ReasoningEnvelope("analysis_bracket", "[ANALYSIS]", "[/ANALYSIS]"),
)


@dataclass(frozen=True, slots=True)
class ForkContinuation:
    """Exact observed text plus the speculative suffix appended after it."""

    observed_text: str
    forced_suffix: str
    decoder_prefix: str
    boundary: str
    tool_format: str
    reasoning_format: str | None = None
    reconstructed_transition: bool = False


def _unclosed_envelope(
    text: str,
    envelopes: tuple[ReasoningEnvelope, ...],
) -> ReasoningEnvelope | None:
    matches = [
        (text.rfind(envelope.open), envelope)
        for envelope in envelopes
        if text.rfind(envelope.open) > text.rfind(envelope.close)
    ]
    return max(matches, default=(-1, None), key=lambda item: item[0])[1]


def _active_envelope(
    base_prompt: str,
    generated_text: str,
    envelopes: tuple[ReasoningEnvelope, ...],
) -> ReasoningEnvelope | None:
    """Find an open generation envelope without inspecting earlier user text."""

    generated = _unclosed_envelope(generated_text, envelopes)
    if generated is not None:
        return generated
    trailing = [
        (position, envelope)
        for envelope in envelopes
        if (position := base_prompt.rfind(envelope.open))
        > base_prompt.rfind(envelope.close)
        and not base_prompt[position + len(envelope.open) :].strip()
    ]
    for position, envelope in sorted(
        trailing, key=lambda item: item[0], reverse=True
    ):
        if _unclosed_envelope(
            base_prompt[position:] + generated_text,
            (envelope,),
        ) is not None:
            return envelope
    return None


@dataclass(frozen=True, slots=True)
class ContinuationPlanner:
    """Compose a probe without crossing textual and structured CoT protocols.

    Providers often expose reasoning and visible content as separate fields even
    though a local chat template places them inside a textual envelope.  When
    the opening marker is present in the rendered prompt, the planner restores
    that transition in its original position.  Opaque or separately signed
    reasoning has no safe textual reconstruction and is rejected.
    """

    envelopes: tuple[ReasoningEnvelope, ...] = DEFAULT_REASONING_ENVELOPES

    def plan(
        self,
        base_prompt: str,
        snapshot: StreamSnapshot,
        *,
        tool_format: str,
        forced_prefix: str | None = None,
    ) -> ForkContinuation:
        syntax = tool_call_syntax(tool_format)
        tool_prefix = (
            forced_prefix if forced_prefix is not None else syntax.probe_prefix
        )
        if not tool_prefix:
            raise ContinuationFormatError("forced tool-call prefix must not be empty")
        if not tool_prefix.startswith(syntax.boundary):
            raise ContinuationFormatError(
                f"tool-call prefix for {tool_format!r} must start with "
                f"{syntax.boundary!r}"
            )

        normalized_split = snapshot.generated_text in (
            "",
            snapshot.reasoning + snapshot.content,
        )
        if snapshot.reasoning and normalized_split:
            envelope = _active_envelope(
                base_prompt,
                snapshot.reasoning,
                self.envelopes,
            )
            if envelope is None:
                raise ContinuationFormatError(
                    "reasoning was exposed as a separate field without a matching "
                    "text envelope; use an engine-native fork that preserves its "
                    "structured or signed reasoning state"
                )
            transition = envelope.exit(snapshot.reasoning)
            if snapshot.content:
                return ForkContinuation(
                    observed_text=(
                        snapshot.reasoning + transition + snapshot.content
                    ),
                    forced_suffix=tool_prefix,
                    decoder_prefix=tool_prefix,
                    boundary=syntax.boundary,
                    tool_format=tool_format,
                    reasoning_format=envelope.name,
                    reconstructed_transition=True,
                )
            return ForkContinuation(
                observed_text=snapshot.reasoning,
                forced_suffix=transition + tool_prefix,
                decoder_prefix=tool_prefix,
                boundary=syntax.boundary,
                tool_format=tool_format,
                reasoning_format=envelope.name,
            )

        observed = snapshot.generated_text
        envelope = _active_envelope(base_prompt, observed, self.envelopes)
        exit_text = envelope.exit(observed) if envelope is not None else ""
        return ForkContinuation(
            observed_text=observed,
            forced_suffix=exit_text + tool_prefix,
            decoder_prefix=tool_prefix,
            boundary=syntax.boundary,
            tool_format=tool_format,
            reasoning_format=envelope.name if envelope is not None else None,
        )


__all__ = [
    "ContinuationFormatError",
    "ContinuationPlanner",
    "DEFAULT_REASONING_ENVELOPES",
    "ForkContinuation",
    "ReasoningEnvelope",
]
