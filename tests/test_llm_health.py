import logging
from unittest.mock import Mock

from modules.llm_health import check_openai_client


def test_startup_check_initializes_without_network(monkeypatch, caplog):
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr("config.AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setattr("openai.AzureOpenAI", constructor)

    with caplog.at_level(logging.INFO):
        assert check_openai_client(logging.getLogger("startup-test"))

    constructor.assert_called_once()
    client.close.assert_called_once()
    assert "OpenAI client initialized successfully" in caplog.text


def test_startup_check_logs_critical_failure(monkeypatch, caplog):
    monkeypatch.setattr("config.AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setattr(
        "openai.AzureOpenAI", Mock(side_effect=TypeError("unexpected keyword argument 'proxies'"))
    )

    with caplog.at_level(logging.CRITICAL):
        assert not check_openai_client(logging.getLogger("startup-test"))

    assert "OpenAI client failed to initialize" in caplog.text
    assert "silently degrade" in caplog.text
