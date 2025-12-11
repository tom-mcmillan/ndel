import asyncio

from ndel.mcp_server import _health_impl


def test_health_returns_status_and_version():
    result = asyncio.run(_health_impl())
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    assert result.get("version")
    assert "privacy_safe_mode" in result
