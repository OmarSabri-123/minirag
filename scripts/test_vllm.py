"""Minimal smoke test for the vLLM generation provider.

Start the vLLM service first, then run: ``python scripts/test_vllm.py``.
The configuration is loaded from ``src/.env`` via the application's
``helpers.Settings`` class.
"""

import asyncio
import sys
from pathlib import Path


# The application modules use absolute imports (for example, ``from logger``),
# so expose src when this script is executed directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpers import get_settings
from stores.llm.LLMEnums import LLMEnums
from stores.llm.LLMProviderFactory import LLMProviderFactory


async def generate() -> None:
    """Generate a short response through the local vLLM OpenAI-compatible API."""
    settings = get_settings()
    provider_factory = LLMProviderFactory(settings)
    provider = provider_factory.create(LLMEnums.VLLM.value)
    provider.set_generation_model(settings.GENERATION_MODEL_ID)

    response = await provider.generate_text(
        "In one sentence, explain what retrieval-augmented generation is.",
        max_output_tokens=100,
        temperature=0.1,
    )
    print(response or "No response received from vLLM.")


if __name__ == "__main__":
    asyncio.run(generate())
