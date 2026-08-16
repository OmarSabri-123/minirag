"""Minimal smoke test for the vLLM embedding provider.

Start the embedding vLLM service first, then run:
``python scripts/test_embeddings.py``. Configuration is loaded from ``src/.env``
via the application's ``helpers.Settings`` class.
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


async def embed() -> None:
    """Create an embedding through the local vLLM OpenAI-compatible API."""
    settings = get_settings()
    provider_factory = LLMProviderFactory(settings)
    provider = provider_factory.create(LLMEnums.VLLM.value, for_embedding=True)
    provider.set_embedding_model(
        settings.EMBEDDING_MODEL_ID,
        settings.EMBEDDING_MODEL_DIMENSION,
    )

    vectors = await provider.embed_text("Retrieval-augmented generation uses relevant context.")
    if not vectors:
        print("No embedding received from vLLM.")
        return

    print(f"Received {len(vectors)} embedding(s) with {len(vectors[0])} dimensions.")
    print(vectors[0][:10])


if __name__ == "__main__":
    asyncio.run(embed())
