import json

import pytest

from simsim import (
    Agent,
    Conversation,
    Experiment,
    Feed,
    FullHistory,
    LLMResponse,
    MarkdownMemory,
    NoMemory,
    Persona,
    ScoredMemory,
    ScriptedLLM,
    Simulation,
    Task,
    TaskSequence,
    ToolCall,
    WindowMemory,
    everyone,
    grid,
    make_llm,
    read_jsonl,
    tool,
)
from simsim.memory import MemoryEntry


def call(name, **args):
    return LLMResponse(tool_calls=[ToolCall(id=f"c-{name}", name=name, arguments=args)])


def test_tool_schema_and_errors():
    @tool
    def add(a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    s = add.schema()["function"]
    assert s["description"] == "Add two numbers."
    assert s["parameters"]["properties"]["a"] == {"type": "integer"}
    assert s["parameters"]["required"] == ["a"]
    assert add(a=2) == "3"
    assert add(a="x").startswith("Error: TypeError")


def test_agent_tool_loop_and_memory():
    @tool
    def lookup(key: str) -> str:
        """Look up a value."""
        return {"color": "blue"}[key]

    llm = ScriptedLLM([call("lookup", key="color"), "It is blue."])
    agent = Agent("a", llm, persona="A helper.", memory=FullHistory(), tools=[lookup])
    turn = agent.act("What color?", t=3)
    assert turn.text == "It is blue."
    assert turn.tool_calls == [{"name": "lookup", "arguments": {"key": "color"}, "result": "blue"}]
    # second model call saw the tool result
    assert llm.calls[1][-1] == {"role": "tool", "tool_call_id": "c-lookup", "content": "blue"}
    mem = agent.memory.read()
    assert "What color?" in mem and "lookup" in mem and "It is blue." in mem


def test_tool_loop_cap_forces_text():
    llm = ScriptedLLM(lambda m, tools: call("noop") if tools else "done")

    @tool
    def noop() -> str:
        """Do nothing."""
        return "ok"

    turn = Agent("a", llm, tools=[noop], max_tool_steps=2).act("go")
    assert turn.text == "done" and len(turn.tool_calls) == 2


def test_no_memory_prompt_has_no_memory_section():
    agent = Agent("a", ScriptedLLM(), persona=Persona("a", "Kind.", {"age": 40}))
    p = agent.system_prompt()
    assert "Kind." in p and "- age: 40" in p and "memory" not in p.lower()


def test_markdown_memory_tools(tmp_path):
    path = tmp_path / "mem.md"
    mem = MarkdownMemory(path, initial="")
    llm = ScriptedLLM([call("memory_append", note="User likes tea."), "Noted."])
    Agent("a", llm, memory=mem).act("I like tea.")
    assert path.read_text() == "User likes tea.\n"
    # the next turn's system prompt contains the note
    assert "User likes tea." in Agent("b", ScriptedLLM(), memory=MarkdownMemory(path)).system_prompt()


def test_window_and_scored_memory():
    w = WindowMemory(k=2)
    for i in range(5):
        w.write(MemoryEntry(i, f"event {i}"))
    assert w.read() == "[t=3] event 3\n[t=4] event 4"

    s = ScoredMemory(k=1, weights=(0, 0, 1))
    s.write(MemoryEntry(0, "the password is swordfish"))
    s.write(MemoryEntry(1, "lunch was pasta"))
    assert "swordfish" in s.read("what is the password")


def test_conversation_round_robin():
    a = Agent("buyer", ScriptedLLM(["Offer 100.", "Deal. [END]"]), memory=FullHistory())
    b = Agent("seller", ScriptedLLM(["Counter 150.", "unused"]))
    env = Conversation("Negotiate a bike price.", private={"buyer": "Budget 140."})
    res = Simulation(env, [a, b], max_steps=10).run()
    assert [m["text"] for m in res.metrics["transcript"]] == ["Offer 100.", "Counter 150.", "Deal."]
    assert res.metrics["ended_by"] == "buyer" and res.termination == "ended by buyer"
    first_obs, second_obs = a.llm.calls[0][1]["content"], a.llm.calls[1][1]["content"]
    assert "Budget 140." in first_obs and "Budget 140." not in second_obs
    assert "seller: Counter 150." in second_obs and "Offer 100." not in second_obs  # show="new"
    assert "Budget 140." not in b.llm.calls[0][1]["content"]


def test_feed_actions():
    a = Agent("alice", ScriptedLLM([call("post", text="hello"), "posted", "idle"]))
    b = Agent("bob", ScriptedLLM([call("like", post_id=0), "liked", "idle"]))
    env = Feed(news=["Rain today."])
    res = Simulation(env, [a, b], scheduler=everyone, max_steps=2).run()
    assert res.metrics["posts"][0]["likes"] == ["bob"]
    assert "#0 @alice: hello" in b.llm.calls[0][1]["content"]
    assert "News: Rain today." in a.llm.calls[0][1]["content"]


def test_simultaneous_agents_see_same_state():
    a = Agent("a", ScriptedLLM([call("post", text="from a"), "ok"]))
    b = Agent("b", ScriptedLLM([call("post", text="from b"), "ok"]))
    Simulation(Feed(), [a, b], scheduler=everyone, max_steps=1, simultaneous=True).run()
    assert "(empty)" in a.llm.calls[0][1]["content"] and "(empty)" in b.llm.calls[0][1]["content"]


def build_task_sim(cond, seed):
    def tools(world):
        @tool
        def set_pref(key: str, value: str) -> str:
            """Save a user preference."""
            world[key] = value
            return "ok"

        return [set_pref]

    tasks = [
        Task("My seat preference is aisle.", id="plant"),
        Task("Book me a seat.", id="probe", score=lambda turn, w: {"success": float("aisle" in turn.text)}),
    ]

    def script(messages, tools):
        sys, user = messages[0]["content"], messages[1]["content"]
        if messages[-1]["role"] == "tool":
            return "Saved."
        if "preference is" in user:
            return call("set_pref", key="seat", value="aisle")
        return "Booked an aisle seat." if "aisle" in sys else "Booked a seat."

    agent = Agent("assistant", ScriptedLLM(script, seed=seed), memory=cond["memory"]())
    return Simulation(TaskSequence(tasks, tools=tools), [agent], seed=seed)


def test_experiment_grid_logs_and_resume(tmp_path):
    conds = grid(memory={"none": NoMemory, "full": FullHistory})
    assert list(conds) == ["memory=none", "memory=full"]
    exp = Experiment(build_task_sim, conds, seeds=[0, 1], out=tmp_path / "exp")
    rows = exp.run(workers=2, progress=False)
    assert len(rows) == 4
    by = {(r["condition"], r["seed"]): r["metrics"]["mean_success"] for r in rows}
    assert by[("memory=full", 0)] == 1.0 and by[("memory=none", 0)] == 0.0
    log = read_jsonl(tmp_path / "exp" / "memory=full" / "seed0.jsonl")
    assert {e["event"] for e in log} >= {"start", "llm_call", "tool_call", "turn", "end"}
    assert exp.run(progress=False) == []  # resume: nothing left
    assert len(exp.results()) == 4
    assert json.loads((tmp_path / "exp" / "conditions.json").read_text())["memory=none"]["memory"] == "NoMemory"


def test_experiment_records_errors(tmp_path):
    def bad(cond, seed):
        raise RuntimeError("boom")

    rows = Experiment(bad, {"c": {}}, out=tmp_path / "e").run(progress=False)
    assert rows[0]["error"] == "RuntimeError: boom"


def test_make_llm_specs():
    llm = make_llm("vllm:Qwen/Qwen2.5-7B-Instruct@http://gpu01:8000/v1", temperature=0)
    assert llm.model == "Qwen/Qwen2.5-7B-Instruct" and llm.base_url == "http://gpu01:8000/v1"
    assert make_llm("ollama:llama3.1").base_url == "http://localhost:11434/v1"
    with pytest.raises(ValueError):
        make_llm("nope:x")
