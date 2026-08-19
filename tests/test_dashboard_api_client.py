"""tests/test_dashboard_api_client.py"""

import pytest
import requests

from dashboard.api_client import APIClientError, get_segments


class _FakeResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data


def test_get_segments_returns_data_on_success(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeResponse(200, [{"segment_id": 0}])

    monkeypatch.setattr("dashboard.api_client.requests.get", fake_get)
    assert get_segments() == [{"segment_id": 0}]


def test_get_segments_raises_when_api_unreachable(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr("dashboard.api_client.requests.get", fake_get)
    with pytest.raises(APIClientError):
        get_segments()


def test_get_segments_raises_clear_message_on_401(monkeypatch):
    def fake_get(*args, **kwargs):
        return _FakeResponse(401)

    monkeypatch.setattr("dashboard.api_client.requests.get", fake_get)
    with pytest.raises(APIClientError, match="API key"):
        get_segments()


def test_get_segments_raises_on_503(monkeypatch):
    def fake_get(*args, **kwargs):
        return _FakeResponse(503)

    monkeypatch.setattr("dashboard.api_client.requests.get", fake_get)
    with pytest.raises(APIClientError):
        get_segments()


def test_get_segments_raises_on_404(monkeypatch):
    def fake_get(*args, **kwargs):
        return _FakeResponse(404)

    monkeypatch.setattr("dashboard.api_client.requests.get", fake_get)
    with pytest.raises(APIClientError):
        get_segments()


def test_get_segments_raises_on_unexpected_status(monkeypatch):
    def fake_get(*args, **kwargs):
        return _FakeResponse(418)

    monkeypatch.setattr("dashboard.api_client.requests.get", fake_get)
    with pytest.raises(APIClientError):
        get_segments()
