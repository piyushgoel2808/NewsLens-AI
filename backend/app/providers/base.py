"""Provider interfaces for all model types in NewsLens-AI.

Every LLM, embedding, VLM, and OCR integration must implement one of these
Protocols. All other application code depends only on these interfaces —
never on a concrete provider class — so providers can be swapped via config.

Design: Python Protocol (PEP 544) provides structural subtyping (duck-typing)
without forced inheritance. This makes it trivial to wrap third-party clients
that don't share a common base class.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Raised when a model provider call fails or is misconfigured."""

    pass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProviderType(StrEnum):
    OLLAMA = "ollama"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL_SENTENCE_TRANSFORMERS = "local_sentence_transformers"
    TESSERACT = "tesseract"
    PADDLE_OCR = "paddleocr"
    HOSTED_OCR = "hosted_ocr"
    MINERU = "mineru"


# ---------------------------------------------------------------------------
# Common dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProviderCapability:
    """Capability tags describing what a provider can do."""

    supports_vision: bool = False
    supports_tool_use: bool = False
    supports_streaming: bool = False
    supports_structured_output: bool = False
    supports_layout: bool = False
    context_window: int = 8192
    embedding_dim: int | None = None


@dataclass
class Message:
    """A single chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str | list[dict[str, Any]]  # str for text; list for multimodal parts


@dataclass
class ToolDefinition:
    """A tool/function the model can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    """A tool invocation returned by the model in its response."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str = ""


@dataclass
class ModelResponse:
    """Unified response from any ChatModelProvider or VisionModelProvider."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    parsed: Any | None = None   # Structured output (when response_schema was given)
    raw: Any | None = None      # Raw provider response (for debugging)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Approximate cost in USD. Overridden in provider subclasses."""
        return 0.0


@dataclass
class OCRBlock:
    """A single block of OCR output with position and confidence."""

    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    confidence: float                          # 0.0 to 1.0
    language: str | None = None


@dataclass
class OCRResult:
    """Full OCR output for one image/page."""

    blocks: list[OCRBlock]
    full_text: str
    mean_confidence: float
    language: str | None = None


# ---------------------------------------------------------------------------
# Provider Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ChatModelProvider(Protocol):
    """Interface for text-based chat LLMs."""

    @property
    def capability(self) -> ProviderCapability:
        """Describes what this provider supports."""
        ...

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'ollama', 'anthropic')."""
        ...

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        response_schema: dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Run a chat completion.

        Args:
            messages: Conversation history.
            tools: Optional tool/function definitions for tool-calling.
            response_schema: Optional JSON Schema — instructs model to return
                             valid JSON matching this schema. Stored in
                             ModelResponse.parsed when present.
            stream: If True, returns an empty ModelResponse.text; use
                    complete_stream() for streaming.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            ModelResponse with text and/or tool_calls populated.

        Raises:
            ProviderError: On API errors, auth failures, or timeouts.
        """
        ...

    def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming text completion. Yields text chunks as they arrive."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for text embedding models."""

    @property
    def capability(self) -> ProviderCapability:
        ...

    @property
    def provider_name(self) -> str:
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimension of the output embedding vectors."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ProviderError: On API errors.
        """
        ...

    async def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper to embed a single string."""
        ...


@runtime_checkable
class VisionModelProvider(Protocol):
    """Interface for vision-language models (layout analysis, image understanding)."""

    @property
    def capability(self) -> ProviderCapability:
        ...

    @property
    def provider_name(self) -> str:
        ...

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Analyze an image with a text prompt.

        Args:
            image_bytes: Raw image data (PNG or JPEG).
            prompt: Instruction for the model.
            response_schema: Optional JSON Schema for structured output.
            max_tokens: Maximum output tokens.

        Returns:
            ModelResponse with the model's analysis.

        Raises:
            ProviderError: On API errors or if provider lacks vision support.
        """
        ...


@runtime_checkable
class OCREngine(Protocol):
    """Interface for OCR engines used on scanned newspaper pages."""

    @property
    def provider_name(self) -> str:
        ...

    async def ocr(
        self,
        image_bytes: bytes,
        lang_hint: str | None = None,
    ) -> OCRResult:
        """Run OCR on an image.

        Args:
            image_bytes: Raw image data (PNG or JPEG).
            lang_hint: ISO 639-1 language code hint (e.g. 'en', 'hi').

        Returns:
            OCRResult with text blocks and confidence scores.

        Raises:
            ProviderError: On OCR engine errors.
        """
        ...


@dataclass
class ExtractedTableData:
    """A structured table extracted from a page."""

    bbox: tuple[float, float, float, float]
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    raw_markdown: str = ""
    raw_html: str | None = None


@dataclass
class ExtractedPhotoData:
    """A photo or visual figure extracted from a page."""

    bbox: tuple[float, float, float, float]
    caption: str | None = None
    image_bytes: bytes | None = None


@dataclass
class ExtractedDocumentNode:
    """A structured layout node extracted by MinerU / document parser."""

    node_type: str  # 'title', 'text', 'table', 'image', 'caption', 'header', 'footer'
    text: str
    bbox: tuple[float, float, float, float]
    reading_order: int = 0
    level: int | None = None  # Heading level (1 = banner, 2 = major, 3 = subhead)
    table_data: ExtractedTableData | None = None
    photo_data: ExtractedPhotoData | None = None


@dataclass
class MinerUParseResult:
    """Consolidated document layout and reading order output from MinerU."""

    page_number: int
    nodes: list[ExtractedDocumentNode] = field(default_factory=list)
    markdown_content: str = ""
    is_ocr_fallback: bool = False
    ocr_confidence: float = 1.0


@runtime_checkable
class DocumentLayoutProvider(Protocol):
    """Interface for neural document layout analysis & reading order engines."""

    @property
    def capability(self) -> ProviderCapability:
        ...

    @property
    def provider_name(self) -> str:
        ...

    async def parse_pdf_document(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
    ) -> list[MinerUParseResult]:
        """Parse complete PDF document using neural layout and reading order."""
        ...

    async def parse_page_image(
        self,
        image_bytes: bytes,
        page_number: int = 1,
        lang: str = "en",
    ) -> MinerUParseResult:
        """Parse a single page raster image using neural layout analysis."""
        ...

