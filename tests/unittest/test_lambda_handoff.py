"""Unit tests for the Lambda ack-then-work hand-off (``pr_agent.servers.lambda_handoff``).

These tests intentionally avoid network, external credentials, and real sleeps.
The dispatcher and the boto3 client are mocked; the webhook verification
(HMAC signature, GitLab token) runs for real against test secrets.
"""

import asyncio
import copy
import hashlib
import hmac
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("GITLAB__URL", "https://gitlab.example.com")

from pr_agent.servers import github_app, gitlab_webhook, lambda_handoff

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def webhook_settings():
    """Set test webhook secrets on the global settings, restoring afterwards.

    Snapshots the whole GITHUB/GITLAB sections: ``settings.unset`` does not
    remove nested Dynaconf keys, so per-key restore would leak the test
    secrets into later tests.
    """
    from pr_agent.config_loader import get_settings

    settings = get_settings(use_context=False)
    original_github = copy.deepcopy(settings.get("GITHUB", None))
    original_gitlab = copy.deepcopy(settings.get("GITLAB", None))
    settings.set("GITHUB.WEBHOOK_SECRET", "test-secret")
    settings.set("GITLAB.SHARED_SECRET", "topsecret")
    settings.set("GITLAB.PERSONAL_ACCESS_TOKEN", "glpat-dummy")
    yield settings
    if original_github is not None:
        settings.set("GITHUB", original_github)
    else:
        settings.unset("GITHUB", force=True)
    if original_gitlab is not None:
        settings.set("GITLAB", original_gitlab)
    else:
        settings.unset("GITLAB", force=True)


class MangumSpy:
    """Stands in for the Mangum ASGI handler; records whether it ran."""

    def __init__(self):
        self.calls = []

    def __call__(self, event, context):
        self.calls.append((event, context))
        return {"statusCode": 200, "body": "mangum"}


def _capture_invoke(monkeypatch, result=True):
    """Replace invoke_self_async with a recorder; returns the captured payloads."""
    calls = []

    def fake(payload):
        calls.append(payload)
        return result

    monkeypatch.setattr(lambda_handoff, "invoke_self_async", fake)
    return calls


def _github_event(body, secret="test-secret", event="pull_request", delivery="d-123",
                  signature=None, path="/api/v1/github_webhooks"):
    raw = json.dumps(body).encode("utf-8")
    if signature is None:
        signature = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {
            "content-type": "application/json",
            "x-github-event": event,
            "x-github-delivery": delivery,
            "x-hub-signature-256": signature,
        },
        "requestContext": {
            "http": {"method": "POST", "path": path, "protocol": "HTTP/1.1"},
        },
        "body": raw.decode("utf-8"),
        "isBase64Encoded": False,
    }


def _gitlab_event(body, token="topsecret", delivery="gl-456"):
    raw = json.dumps(body).encode("utf-8")
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": "/webhook",
        "rawQueryString": "",
        "headers": {
            "content-type": "application/json",
            "x-gitlab-token": token,
            "x-gitlab-event": "Merge Request Hook",
            "x-gitlab-event-uuid": delivery,
        },
        "requestContext": {
            "http": {"method": "POST", "path": "/webhook", "protocol": "HTTP/1.1"},
        },
        "body": raw.decode("utf-8"),
        "isBase64Encoded": False,
    }


def _github_body():
    return {"action": "opened", "installation": {"id": 99},
            "pull_request": {"url": "https://api.github.com/repos/o/r/pulls/1"}}


# ---------------------------------------------------------------------------
# is_handoff_event / build_handoff_payload
# ---------------------------------------------------------------------------


def test_is_handoff_event_accepts_valid_markers():
    assert lambda_handoff.is_handoff_event(
        lambda_handoff.build_handoff_payload("github", {"action": "opened"}))
    assert lambda_handoff.is_handoff_event(
        lambda_handoff.build_handoff_payload("gitlab", {"object_kind": "note"}))


def test_is_handoff_event_rejects_anything_else():
    assert not lambda_handoff.is_handoff_event(None)
    assert not lambda_handoff.is_handoff_event("event")
    assert not lambda_handoff.is_handoff_event({})
    assert not lambda_handoff.is_handoff_event({"body": {}})
    assert not lambda_handoff.is_handoff_event(
        {"pr_agent_lambda_handoff": {"version": 999, "provider": "github"}})
    assert not lambda_handoff.is_handoff_event(
        {"pr_agent_lambda_handoff": {"version": 1, "provider": "bitbucket"}})


def test_build_handoff_payload_carries_delivery_id_but_no_token_for_github():
    payload = lambda_handoff.build_handoff_payload(
        "github", {"action": "opened"}, event="pull_request", delivery_id="d-1")
    assert payload["delivery_id"] == "d-1"
    assert payload["event"] == "pull_request"
    assert "token" not in payload


def test_build_handoff_payload_carries_token_for_gitlab():
    payload = lambda_handoff.build_handoff_payload(
        "gitlab", {"object_kind": "note"}, delivery_id="gl-1", token="tok")
    assert payload["token"] == "tok"
    assert payload["delivery_id"] == "gl-1"


# ---------------------------------------------------------------------------
# Request construction from Lambda events
# ---------------------------------------------------------------------------


def test_build_request_round_trips_body_and_headers():
    request = lambda_handoff.build_request(
        "POST", "/webhook", {"X-Gitlab-Token": "topsecret"}, b'{"a": 1}')
    assert asyncio.run(request.body()) == b'{"a": 1}'
    assert asyncio.run(request.json()) == {"a": 1}
    assert request.headers["x-gitlab-token"] == "topsecret"


def test_request_from_lambda_event_v2_decodes_base64_body():
    raw = json.dumps({"action": "opened"}).encode("utf-8")
    import base64

    event = {
        "version": "2.0",
        "rawPath": "/api/v1/github_webhooks",
        "headers": {"X-GitHub-Event": "pull_request"},
        "requestContext": {"http": {"method": "POST", "path": "/api/v1/github_webhooks"}},
        "body": base64.b64encode(raw).decode("ascii"),
        "isBase64Encoded": True,
    }
    request = lambda_handoff.request_from_lambda_event(event)
    assert request.method == "POST"
    assert asyncio.run(request.json()) == {"action": "opened"}
    assert request.headers["x-github-event"] == "pull_request"


def test_request_from_lambda_event_v1_uses_http_method_and_path():
    event = {
        "httpMethod": "POST",
        "path": "/webhook",
        "headers": {"X-Gitlab-Token": "topsecret"},
        "body": '{"object_kind": "note"}',
        "isBase64Encoded": False,
    }
    request = lambda_handoff.request_from_lambda_event(event)
    assert request.method == "POST"
    assert request.url.path == "/webhook"
    assert asyncio.run(request.json()) == {"object_kind": "note"}


# ---------------------------------------------------------------------------
# First invocation: verify, hand off, ack immediately
# ---------------------------------------------------------------------------


def test_github_webhook_returns_200_without_running_the_dispatcher(monkeypatch, webhook_settings):
    """The ack must not wait for the dispatcher: the review never starts in
    the first invocation, so it cannot delay the 200."""
    dispatched = []

    async def record_dispatch(*args, **kwargs):
        dispatched.append((args, kwargs))

    monkeypatch.setattr(github_app, "handle_request", record_dispatch)
    invoked = _capture_invoke(monkeypatch)
    mangum = MangumSpy()

    response = lambda_handoff.lambda_handler_for(
        "github", _github_event(_github_body()), None, mangum)

    assert response["statusCode"] == 200
    assert dispatched == []
    assert mangum.calls == []
    assert len(invoked) == 1
    payload = invoked[0]
    assert payload["body"] == _github_body()
    assert payload["event"] == "pull_request"
    assert payload["delivery_id"] == "d-123"
    assert lambda_handoff.is_handoff_event(payload)


def test_github_webhook_with_bad_signature_is_rejected_without_handoff(monkeypatch, webhook_settings):
    dispatched = []

    async def record_dispatch(*args, **kwargs):
        dispatched.append(True)

    monkeypatch.setattr(github_app, "handle_request", record_dispatch)
    invoked = _capture_invoke(monkeypatch)
    mangum = MangumSpy()

    event = _github_event(_github_body(), signature="sha256=wrong")
    response = lambda_handoff.lambda_handler_for("github", event, None, mangum)

    assert response["statusCode"] == 403
    assert dispatched == []
    assert invoked == []
    assert mangum.calls == []


def test_github_webhook_without_a_configured_secret_is_rejected(monkeypatch, webhook_settings):
    from pr_agent.config_loader import get_settings

    # ``settings.unset`` does not remove nested Dynaconf keys, so clear the
    # secret by setting it to None (``get_body`` treats that as unconfigured).
    get_settings(use_context=False).set("GITHUB.WEBHOOK_SECRET", None)
    invoked = _capture_invoke(monkeypatch)
    mangum = MangumSpy()

    response = lambda_handoff.lambda_handler_for(
        "github", _github_event(_github_body()), None, mangum)

    assert response["statusCode"] == 403
    assert invoked == []
    assert mangum.calls == []


def test_gitlab_webhook_returns_200_and_carries_the_token(monkeypatch, webhook_settings):
    dispatched = []

    async def record_dispatch(*args, **kwargs):
        dispatched.append(True)

    monkeypatch.setattr(gitlab_webhook, "handle_gitlab_event", record_dispatch)
    invoked = _capture_invoke(monkeypatch)
    mangum = MangumSpy()

    body = {"object_kind": "merge_request", "object_attributes": {"action": "open"}}
    response = lambda_handoff.lambda_handler_for("gitlab", _gitlab_event(body), None, mangum)

    assert response["statusCode"] == 200
    assert dispatched == []
    assert mangum.calls == []
    assert len(invoked) == 1
    assert invoked[0]["token"] == "topsecret"
    assert invoked[0]["delivery_id"] == "gl-456"
    assert invoked[0]["body"] == body


def test_gitlab_webhook_with_wrong_token_is_rejected_without_handoff(monkeypatch, webhook_settings):
    invoked = _capture_invoke(monkeypatch)
    mangum = MangumSpy()

    response = lambda_handoff.lambda_handler_for(
        "gitlab", _gitlab_event({"object_kind": "note"}, token="wrong"), None, mangum)

    assert response["statusCode"] == 401
    assert invoked == []
    assert mangum.calls == []


def test_non_webhook_requests_fall_through_to_mangum(monkeypatch, webhook_settings):
    mangum = MangumSpy()
    event = {
        "version": "2.0",
        "rawPath": "/",
        "headers": {},
        "requestContext": {"http": {"method": "GET", "path": "/"}},
        "body": "",
        "isBase64Encoded": False,
    }
    response = lambda_handoff.lambda_handler_for("github", event, None, mangum)
    assert response == {"statusCode": 200, "body": "mangum"}
    assert len(mangum.calls) == 1


def test_self_invoke_failure_falls_back_to_mangum(monkeypatch, webhook_settings):
    """A misconfigured IAM policy must not drop the review: fall back to the
    old behaviour (slow acknowledgement, review still lands)."""
    dispatched = []

    async def record_dispatch(*args, **kwargs):
        dispatched.append(True)

    monkeypatch.setattr(github_app, "handle_request", record_dispatch)
    monkeypatch.setattr(
        lambda_handoff, "invoke_self_async",
        lambda payload: (_ for _ in ()).throw(RuntimeError("not authorized to invoke")))
    mangum = MangumSpy()

    response = lambda_handoff.lambda_handler_for(
        "github", _github_event(_github_body()), None, mangum)

    assert len(mangum.calls) == 1
    assert response == {"statusCode": 200, "body": "mangum"}


def test_missing_function_name_falls_back_to_mangum(monkeypatch, webhook_settings):
    """Without AWS_LAMBDA_FUNCTION_NAME there is nothing to invoke, so the
    webhook runs inline exactly as before."""
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    mangum = MangumSpy()

    response = lambda_handoff.lambda_handler_for(
        "github", _github_event(_github_body()), None, mangum)

    assert len(mangum.calls) == 1
    assert response == {"statusCode": 200, "body": "mangum"}


# ---------------------------------------------------------------------------
# Second invocation: run the dispatcher from the hand-off payload
# ---------------------------------------------------------------------------


def test_github_handoff_runs_the_dispatcher(monkeypatch, webhook_settings):
    dispatched = []

    async def record_dispatch(body, event, delivery_id=None):
        dispatched.append((body, event, delivery_id))

    monkeypatch.setattr(github_app, "handle_request", record_dispatch)
    payload = lambda_handoff.build_handoff_payload(
        "github", _github_body(), event="pull_request", delivery_id="d-123")

    asyncio.run(lambda_handoff.run_handoff(payload))

    assert dispatched == [(_github_body(), "pull_request", "d-123")]


def test_gitlab_handoff_reauthenticates_and_dispatches(monkeypatch, webhook_settings):
    dispatched = []

    async def record_dispatch(data, log_context):
        dispatched.append(data)

    monkeypatch.setattr(gitlab_webhook, "handle_gitlab_event", record_dispatch)
    body = {"object_kind": "merge_request", "object_attributes": {"action": "open"}}
    payload = lambda_handoff.build_handoff_payload(
        "gitlab", body, delivery_id="gl-456", token="topsecret")

    asyncio.run(lambda_handoff.run_handoff(payload))

    assert dispatched == [body]


def test_gitlab_handoff_with_a_stale_token_dispatches_nothing(monkeypatch, webhook_settings):
    dispatched = []

    async def record_dispatch(data, log_context):
        dispatched.append(data)

    monkeypatch.setattr(gitlab_webhook, "handle_gitlab_event", record_dispatch)
    payload = lambda_handoff.build_handoff_payload(
        "gitlab", {"object_kind": "note"}, delivery_id="gl-456", token="wrong")

    asyncio.run(lambda_handoff.run_handoff(payload))

    assert dispatched == []


def test_handoff_event_returns_200_from_the_lambda_entry(monkeypatch, webhook_settings):
    dispatched = []

    async def record_dispatch(body, event, delivery_id=None):
        dispatched.append(True)

    monkeypatch.setattr(github_app, "handle_request", record_dispatch)
    mangum = MangumSpy()
    payload = lambda_handoff.build_handoff_payload(
        "github", _github_body(), event="pull_request", delivery_id="d-123")

    response = lambda_handoff.lambda_handler_for("github", payload, None, mangum)

    assert response["statusCode"] == 200
    assert dispatched == [True]
    assert mangum.calls == []


# ---------------------------------------------------------------------------
# invoke_self_async
# ---------------------------------------------------------------------------


def test_invoke_self_async_calls_lambda_with_event_invocation(monkeypatch):
    invocations = []

    class FakeLambdaClient:
        def invoke(self, **kwargs):
            invocations.append(kwargs)
            return {"StatusCode": 202}

    fake_boto3 = SimpleNamespace(client=lambda service: FakeLambdaClient())
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "pr-agent-github")
    payload = lambda_handoff.build_handoff_payload("github", {"action": "opened"})

    assert lambda_handoff.invoke_self_async(payload) is True
    assert len(invocations) == 1
    assert invocations[0]["FunctionName"] == "pr-agent-github"
    assert invocations[0]["InvocationType"] == "Event"
    assert json.loads(invocations[0]["Payload"]) == payload


def test_invoke_self_async_returns_false_without_a_function_name(monkeypatch):
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    assert lambda_handoff.invoke_self_async({"body": {}}) is False


# ---------------------------------------------------------------------------
# Lambda entry points (the real ``lambda_handler`` in each image)
# ---------------------------------------------------------------------------


def test_github_lambda_entry_point_hands_off(monkeypatch, webhook_settings):
    """Drive the real github_lambda_webhook.lambda_handler: verify, hand off,
    ack — the dispatcher never runs in this invocation."""
    pytest.importorskip("mangum")
    import pr_agent.servers.github_lambda_webhook as gh_lambda

    dispatched = []

    async def record_dispatch(*args, **kwargs):
        dispatched.append(True)

    monkeypatch.setattr(github_app, "handle_request", record_dispatch)
    invoked = _capture_invoke(monkeypatch)

    response = gh_lambda.lambda_handler(_github_event(_github_body()), None)

    assert response["statusCode"] == 200
    assert dispatched == []
    assert len(invoked) == 1


def test_gitlab_lambda_entry_point_hands_off(monkeypatch, webhook_settings):
    """Drive the real gitlab_lambda_webhook.lambda_handler the same way."""
    pytest.importorskip("mangum")
    import pr_agent.servers.gitlab_lambda_webhook as gl_lambda

    dispatched = []

    async def record_dispatch(*args, **kwargs):
        dispatched.append(True)

    monkeypatch.setattr(gitlab_webhook, "handle_gitlab_event", record_dispatch)
    invoked = _capture_invoke(monkeypatch)

    body = {"object_kind": "merge_request", "object_attributes": {"action": "open"}}
    response = gl_lambda.lambda_handler(_gitlab_event(body), None)

    assert response["statusCode"] == 200
    assert dispatched == []
    assert len(invoked) == 1
