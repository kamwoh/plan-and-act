"""Model-agnostic chat interface: messages with images in, a stream of events out."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    text: str
    images: tuple[bytes, ...] = ()  # PNG bytes


@dataclass(frozen=True)
class Delta:
    text: str


@dataclass(frozen=True)
class Final:
    text: str
    parsed: dict | None


@dataclass(frozen=True)
class Usage:
    model: str
    latency_s: float
    in_tokens: int | None = None
    out_tokens: int | None = None
    cost_usd: float | None = None
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None


@dataclass(frozen=True)
class LLMError:
    # resume: the saved chat is gone. too_long: the chat has too many images or tokens
    kind: Literal["timeout", "cancelled", "backend", "auth", "limit", "schema", "resume", "too_long"]
    message: str


@dataclass(frozen=True)
class Progress:
    """Liveness of a running turn: the CLI answered ("started"), then how much has streamed so far."""
    stage: Literal["started", "receiving"]
    chars: int = 0


LLMEvent = Delta | Final | Usage | LLMError | Progress


@runtime_checkable
class Conversation(Protocol):
    session_id: str | None  # the saved chat of a chain, once it is known. None for a chat that is not saved

    def send(self, msg: Message, schema: dict | None = None) -> AsyncIterator[LLMEvent]: ...
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    async def open(self, system: str, chain: str | None = None, resume: str | None = None) -> Conversation:
        """A new chat, or with `chain` a saved chat of that chain. With `resume`, it continues that saved chat.
        When the saved chat is gone, the first send gives an LLMError of kind "resume"."""
        ...
