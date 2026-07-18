"""Startup validation for the configured Azure OpenAI client."""

from __future__ import annotations

import logging

import config


def check_openai_client(logger: logging.Logger) -> bool:
    """Initialize and close a client without making a network request."""
    try:
        if not config.AZURE_OPENAI_KEY:
            raise RuntimeError("AZURE_OPENAI_KEY is not configured")
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            timeout=config.AI_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
        client.close()
    except Exception as exc:
        logger.critical(
            "OpenAI client failed to initialize: %s — "
            "LLM-dependent features will silently degrade to fallback paths.",
            exc,
        )
        return False
    logger.info("OpenAI client initialized successfully")
    return True
