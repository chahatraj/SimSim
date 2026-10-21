"""OpenAIChat against a local fake OpenAI-compatible server (no network, no key)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from simsim import Agent, OpenAIEmbedder, make_llm, tool


class FakeOpenAI(BaseHTTPRequestHandler):
    script: list = []          # queued (status, body) replies for /chat/completions
    requests: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.requests.append((self.path, body))
        if self.path.endswith("/embeddings"):
            data = [{"object": "embedding", "index": i, "embedding": [float(len(t)), 1.0]}
                    for i, t in enumerate(body["input"])]
            return self._send(200, {"object": "list", "data": data, "model": body["model"],
                                    "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        status, reply = FakeOpenAI.script.pop(0)
        self._send(status, reply)

    def _send(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def completion(content=None, tool_calls=None, reasoning=None, finish="stop", logprobs=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [{"id": f"call{i}", "type": "function",
                              "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(tool_calls)]
    if reasoning:
        msg["reasoning_content"] = reasoning
    choice = {"index": 0, "message": msg, "finish_reason": finish}
    if logprobs:
        choice["logprobs"] = {"content": [{"token": "x", "logprob": 0.0, "bytes": None, "top_logprobs": [
            {"token": tok, "logprob": lp, "bytes": None} for tok, lp in logprobs]}]}
    return {"id": "c", "object": "chat.completion", "created": 0, "model": "fake", "choices": [choice],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


@pytest.fixture
def server(monkeypatch):
    FakeOpenAI.script, FakeOpenAI.requests = [], []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr("simsim.llm.time.sleep", lambda s: None)   # no real backoff waits
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_chat_tools_reasoning_and_usage(server):
    @tool
    def get_weather(city: str) -> str:
        """Weather."""
        return "sunny"

    FakeOpenAI.script = [
        (200, completion(tool_calls=[("get_weather", '{"city": "Cairo"}')], reasoning="need weather")),
        (200, completion("  It is sunny.  ")),
    ]
    llm = make_llm(f"vllm:fake@{server}", seed=3, price=(1.0, 1.0),
                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    turn = Agent("a", llm, tools=[get_weather]).act("Weather in Cairo?")
    assert turn.text == "It is sunny." and turn.tool_calls[0]["result"] == "sunny"
    assert "need weather" in turn.reasoning
    first = FakeOpenAI.requests[0][1]
    assert first["tools"][0]["function"]["name"] == "get_weather"
    assert first["chat_template_kwargs"] == {"enable_thinking": False}   # extra_body is merged into the request
    assert isinstance(first["seed"], int) and first["seed"] != FakeOpenAI.requests[1][1]["seed"]
    second = FakeOpenAI.requests[1][1]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "call0", "content": "sunny"}
    assert llm.usage["calls"] == 2 and llm.usage["prompt_tokens"] == 20 and llm.usage["cost_usd"] == pytest.approx(3e-5)


def test_malformed_tool_arguments_and_truncation(server):
    FakeOpenAI.script = [(200, completion(tool_calls=[("f", "{bad json")])), (200, completion("", finish="length"))]
    llm = make_llm(f"vllm:fake@{server}")
    resp = llm.chat([{"role": "user", "content": "x"}])
    assert resp.tool_calls[0].arguments == {"_raw": "{bad json"}
    assert llm.chat([{"role": "user", "content": "x"}]).finish_reason == "length"


def test_retries_transient_errors_but_not_bad_requests(server):
    err = {"error": {"message": "slow down", "type": "rate_limit"}}
    FakeOpenAI.script = [(429, err), (500, err), (200, completion("ok"))]
    llm = make_llm(f"vllm:fake@{server}", retries=3)
    assert llm.chat([{"role": "user", "content": "x"}]).text == "ok" and llm.usage["retries"] == 2

    import openai

    FakeOpenAI.script = [(400, {"error": {"message": "bad request", "type": "invalid"}})]
    with pytest.raises(openai.BadRequestError):
        llm.chat([{"role": "user", "content": "x"}])
    assert FakeOpenAI.script == []   # exactly one attempt


def test_choice_probs_from_logprobs(server):
    import math

    FakeOpenAI.script = [(200, completion("1", logprobs=[("1", math.log(0.6)), (" 2", math.log(0.2)), ("zzz", -1.0)]))]
    llm = make_llm(f"vllm:fake@{server}")
    probs = Agent("a", llm).survey("Q?", {"1": "yes", "2": "no", "3": "maybe"})
    assert probs == pytest.approx({"1": 0.75, "2": 0.25, "3": 0.0})
    req = FakeOpenAI.requests[0][1]
    assert req["logprobs"] is True and req["max_tokens"] == 1 and req["temperature"] == 0


def test_openai_embedder(server):
    emb = OpenAIEmbedder("fake-embed", base_url=server, batch_size=2)
    vecs = emb.embed(["a", "bbb", "cc"])
    assert vecs == [[1.0, 1.0], [3.0, 1.0], [2.0, 1.0]]
    assert len([r for r in FakeOpenAI.requests if r[0].endswith("/embeddings")]) == 2   # batched
    emb.embed(["a"])
    assert len(FakeOpenAI.requests) == 2   # cached
