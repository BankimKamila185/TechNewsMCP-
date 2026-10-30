"""
Unit tests for Render web application endpoints.
"""

from unittest.mock import AsyncMock, patch
import pytest
from starlette.testclient import TestClient

import app


@pytest.fixture(autouse=True)
def disable_scheduler(monkeypatch):
    """Disable background scheduler task during tests."""
    monkeypatch.setattr(app, "ENABLE_SCHEDULER", False)


def test_health_check_endpoint():
    """Verify health endpoint returns 200 and healthy status."""
    with TestClient(app.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "tech-news-mcp"


def test_root_endpoint():
    """Verify root endpoint returns 200."""
    with TestClient(app.app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


def test_manual_trigger_endpoint_mocked():
    """Verify on-demand trigger endpoint invokes agent."""
    with patch("app.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {"status": "Digest sent successfully"}
        with TestClient(app.app) as client:
            resp = client.get("/trigger")
            assert resp.status_code == 200
            assert resp.json()["status"] == "success"
            assert mock_run.called

