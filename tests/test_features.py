import json

import pytest

from simsim import (
    END,
    LLM,
    Agent,
    Conversation,
    FullHistory,
    HumanAgent,
    Judge,
    LLMResponse,
    ScoredMemory,
    ScriptedAgent,
    ScriptedLLM,
    Simulation,
    Task,
    TaskSequence,
    ToolCall,
    add_agent,
    at,
    every,
    extract_json,
    parse_react,
    personas_from_rows,
    probe,
    rescore,
    sample_personas,
    structured_chat,
    survey,
    tool,
    user_simulator,
    whisper,
)
from simsim.structured import StructuredOutputError, parse

# ---- llm ---------------------------------------------------------------------

class Flaky(LLM):
    model = "flaky"

    def __init__(self, fails, **kw):
        super().__init__(**kw)
        self.fails = fails

    def _chat(self, messages, tools, **kw):
        if self.fails:
            self.fails -= 1
            raise ConnectionError("temporary")
        return LLMResponse("fine", usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})


def test_retry_backoff_and_cost(monkeypatch):
    monkeypatch.setattr("simsim.llm.time.sleep", lambda s: None)
    llm = Flaky(fails=2, retries=3, price=(0.15, 0.60))
    assert llm.chat([{"role": "user", "content": "hi"}]).text == "fine"
    assert llm.usage["retries"] == 2 and llm.usage["cost_usd"] == pytest.approx(0.75)
    with pytest.raises(ConnectionError):
        Flaky(fails=5, retries=1).chat([])


def test_programming_errors_are_not_retried(monkeypatch):
    class Bad(LLM):
        def _chat(self, m, t, **kw):
            raise TypeError("bug")

    sleeps = []
    monkeypatch.setattr("simsim.llm.time.sleep", sleeps.append)
    with pytest.raises(TypeError):
        Bad(retries=5).chat([])
    assert sleeps == []


# ---- structured --------------------------------------------------------------

SCHEMA = {"analysis": (str, "private"), "reply": str, "stance": ["agree", "disagree"], "conf": (int, 1, 7)}


def test_extract_and_validate():
    text = 'Sure!\n```json\n{"analysis": "x {y}", "reply": "ok", "stance": "Agree", "conf": "5"}\n```'
    assert parse(text, SCHEMA) == {"analysis": "x {y}", "reply": "ok", "stance": "agree", "conf": 5}
    assert extract_json('<think>{"no": 1}</think> answer {"a": {"b": 2}} tail') == {"a": {"b": 2}}
    with pytest.raises(StructuredOutputError, match="between 1 and 7"):
        parse('{"analysis": "", "reply": "", "stance": "agree", "conf": 9}', SCHEMA)
    with pytest.raises(StructuredOutputError, match="missing field 'reply'"):
        parse('{"analysis": ""}', SCHEMA)


def test_structured_chat_repairs():
    llm = ScriptedLLM(["not json", '{"analysis": "a", "reply": "r", "stance": "agree", "conf": 3}'])
    data, messages, errors = structured_chat(llm, [{"role": "user", "content": "go"}], SCHEMA)
    assert data["reply"] == "r" and len(errors) == 1
    assert "could not be used" in messages[2]["content"]


# ---- agent -------------------------------------------------------------------

def test_private_reasoning_split_and_memory():
    llm = ScriptedLLM(['{"analysis": "they seem weak", "reply": "I offer 90.", "stance": "agree", "conf": 6}'])
    a = Agent("buyer", llm, output_schema=SCHEMA, memory=FullHistory())
    turn = a.act("Seller: 120?")
    assert turn.text == "I offer 90." and turn.private == {"analysis": "they seem weak", "stance": "agree", "conf": 6}
    assert "they seem weak" in a.memory.read()  # the agent remembers its own private notes
    b = Agent("buyer", ScriptedLLM([llm.script[0]]), output_schema=SCHEMA, memory=FullHistory(), remember_private=False)
    b.act("x")
    assert "they seem weak" not in b.memory.read()


def test_schema_failure_is_recorded_not_raised():
    a = Agent("a", ScriptedLLM(["nope", "still nope", "never"]), output_schema={"reply": str})
    turn = a.act("hi")
    assert turn.error and turn.text == "nope"


def test_react_mode_with_parse_repair():
    @tool
    def lookup(key: str) -> str:
        """Look up a key."""
        return {"capital": "Paris"}[key]

    llm = ScriptedLLM([
        "I will just answer",  # unparseable -> repair message
        'Thought: need data\nAction: lookup\nAction Input: {"key": "capital"}',
        "Thought: got it\nFinal Answer: Paris",
    ])
    a = Agent("a", llm, tools=[lookup], action_format="react")
    turn = a.act("Capital of France?")
    assert turn.text == "Paris" and turn.tool_calls[0]["result"] == "Paris"
    assert "Action Input:" in llm.calls[0][0]["content"]  # tool format in system prompt
    assert "Could not parse" in llm.calls[1][-1]["content"]
    assert llm.calls[2][-1]["content"] == "Observation: Paris"
    assert "need data" in turn.reasoning


def test_parse_react():
    assert parse_react("Final Answer: 42") == ("final", "42", "")
    kind, (name, args), _ = parse_react('Thought: t\nAction: `search`\nAction Input: {"q": "x"}\nObservation: fake')
    assert (kind, name, args) == ("action", "search", {"q": "x"})
    assert parse_react("Action: s\nAction Input: not json")[0] == "error"


def test_bad_tool_arguments_reported():
    @tool
    def f(x: int) -> int:
        """f"""
        return x

    llm = ScriptedLLM([LLMResponse(tool_calls=[ToolCall("1", "f", {"_raw": "{x:"})]), "done"])
    turn = Agent("a", llm, tools=[f]).act("go")
    assert "not valid JSON" in turn.tool_calls[0]["result"]


def test_thinking_reasoning_captured_and_usage():
    llm = ScriptedLLM([LLMResponse("hi", reasoning="let me think", usage={"prompt_tokens": 5, "completion_tokens": 2})])
    turn = Agent("a", llm).act("hello")
    assert turn.reasoning.strip() == "let me think" and turn.usage == {"prompt_tokens": 5, "completion_tokens": 2, "calls": 1}


def test_one_memory_entry_per_turn_and_truncated_observations():
    calls = []
    mem = ScoredMemory(importance=lambda text: calls.append(text) or 0.5)
    a = Agent("a", ScriptedLLM(["ok"]), memory=mem, remember_observations=5)
    a.act("0123456789")
    assert len(calls) == 1 and "Observed: 01234\n" in calls[0]


def test_memory_max_chars():
    a = Agent("a", ScriptedLLM(), memory=FullHistory(), memory_max_chars=20)
    for i in range(10):
        a.act(f"message number {i}")
    assert len(a.memory_text()) <= 24 and a.memory_text().startswith("...")


def test_ask_does_not_change_memory():
    a = Agent("a", ScriptedLLM(['{"rating": 4}', "free text"]), memory=FullHistory())
    assert a.ask("Rate it", schema={"rating": (int, 1, 5)}) == {"rating": 4}
    assert a.ask("Why?") == "free text"
    assert a.memory.read() == ""


def test_survey_uses_choice_probs():
    a = Agent("a", ScriptedLLM(["B"]))
    assert a.survey("Agree?", ["Yes", "No"]) == {"A": 0.0, "B": 1.0}


def test_scripted_and_human_agents_in_conversation():
    bot = ScriptedAgent("bot", ["Hello.", "Bye. [END]"])
    human = HumanAgent("human", input_fn=lambda prompt: "Hi bot.")
    res = Simulation(Conversation("Chat."), [bot, human], max_steps=10).run()
    assert [m["text"] for m in res.metrics["transcript"]] == ["Hello.", "Hi bot.", "Bye."]


# ---- environments ------------------------------------------------------------

def test_user_simulator_with_tools_and_enders():
    @tool
    def refund(order_id: str) -> str:
        """Refund an order."""
        return "refunded"

    user = user_simulator("user", ScriptedLLM(["I want a refund for order 7.", "Thanks ###STOP###"]), goal="refund order 7")
    assistant = Agent("assistant", ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("1", "refund", {"order_id": "7"})]), "Done, refunded. ###STOP###",
    ]))
    env = Conversation(
        "Customer support chat.", end_token="###STOP###", enders=["user"], tools={"assistant": [refund]},
        score=lambda transcript, world: {"n": len(transcript)},
    )
    res = Simulation(env, [user, assistant], max_steps=10).run()
    assert res.termination == "ended by user"  # the assistant's ###STOP### did not end it
    assert res.metrics["n"] == 3
    assert "refund order 7" in user.llm.calls[0][0]["content"]
    assert assistant.llm.calls[0][-1]["content"].count("###STOP###") == 0  # only enders are told about it


def test_end_token_only_on_last_line():
    a = ScriptedAgent("a", ['She wrote "[END]" in the book.\nAnyway, continue.'])
    res = Simulation(Conversation("x"), [a, ScriptedAgent("b", ["ok"])], max_steps=3).run()
    assert res.termination == "max_steps"


def test_conversation_judge_stops():
    judge = Judge(ScriptedLLM(lambda m, t: '{"answer": "yes"}'), rubric="")
    env = Conversation("x", end_token=None, judge=judge.stopper("Agreed?"), judge_every=2)
    res = Simulation(env, [ScriptedAgent("a", ["p"]), ScriptedAgent("b", ["q"])], max_steps=10).run()
    assert res.termination == "judge" and res.steps == 2


def test_task_sequence_multi_agent():
    tasks = [Task("t1", score=lambda turn, w: {"ok": 1.0}), Task("t2")]
    agents = [ScriptedAgent("a", ["x"]), ScriptedAgent("b", ["y"])]
    from simsim import everyone

    res = Simulation(TaskSequence(tasks), agents, scheduler=everyone, max_steps=5).run()
    assert res.termination == "all tasks done" and len(res.metrics["tasks"]) == 4
    assert {r["agent"] for r in res.metrics["tasks"]} == {"a", "b"}


# ---- interventions, probes, evaluators ---------------------------------------

def test_interventions_probes_and_evaluators():
    talker = Agent("a", ScriptedLLM(lambda m, t: "rating" in m[-1]["content"] and '{"rating": 2}' or "hm"),
                   memory=FullHistory())
    other = ScriptedAgent("b", ["ok"])
    newcomer = ScriptedAgent("c", ["hello all"])
    judge = Judge(ScriptedLLM(lambda m, t: '{"reasoning": "fine", "score": 7}'), rubric="Rate {text}")
    sim = Simulation(
        Conversation("x", end_token=None), [talker, other], max_steps=4,
        interventions=[
            at(1, whisper("a", "You were insulted.")),
            at(2, add_agent(newcomer)),
            every(2, probe("Give a rating", schema={"rating": (int, 1, 5)}, agents=["a"])),
            at(END, probe("Final rating", schema={"rating": (int, 1, 5)}, agents=["a"], name="final")),
        ],
        evaluators={"judge": judge.evaluator()},
    )
    res = sim.run()
    assert "You were insulted." in talker.memory.read()
    assert any(t.agent == "c" for t in res.turns)
    assert [(p["name"], p["t"]) for p in res.probes] == [("probe", 0), ("probe", 2), ("final", 4)]
    assert res.probes[0]["answer"] == {"rating": 2}
    assert res.metrics["judge"] == {"reasoning": "fine", "score": 7}


def test_survey_intervention():
    a = Agent("a", ScriptedLLM(lambda m, t: "7"))
    res = Simulation(Conversation("x", end_token=None), [a], max_steps=1,
                     interventions=[at(0, survey("Agree?", {str(i): str(i) for i in range(1, 8)}))]).run()
    assert res.probes[0]["probs"]["7"] == 1.0


def test_usage_per_agent_with_shared_backend():
    shared = ScriptedLLM([LLMResponse("x", usage={"prompt_tokens": 3, "completion_tokens": 1})] * 4)
    res = Simulation(Conversation("x", end_token=None), [Agent("a", shared), Agent("b", shared)], max_steps=4).run()
    assert res.usage["agents"]["a"]["calls"] == 2 and res.usage["agents"]["b"]["calls"] == 2
    assert list(res.usage["backends"].values())[0]["calls"] == 4  # counted once, not per agent


def test_rescore(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"condition": "c", "seed": 0, "metrics": {"transcript": [{"agent": "a", "text": "hi"}]}}) + "\n")
    judge = Judge(ScriptedLLM(lambda m, t: '{"reasoning": "r", "score": 3}'), rubric="Score it.")
    rows = rescore(path, judge, render=lambda r: r["metrics"]["transcript"][0]["text"], out=tmp_path / "re.jsonl")
    assert rows[0]["judge"]["score"] == 3 and (tmp_path / "re.jsonl").exists()


# ---- personas ----------------------------------------------------------------

def test_sample_personas_deterministic_and_described():
    attrs = {"age": range(18, 80), "party": {"D": 0.5, "R": 0.5}, "x": lambda rng: rng.random() < 2}
    a = sample_personas(5, attrs, seed=1)
    b = sample_personas(5, attrs, seed=1)
    assert [p.attributes for p in a] == [p.attributes for p in b] and a[0].attributes["x"] is True
    described = sample_personas(1, attrs, describe=ScriptedLLM(["A retired welder from Ohio."]))
    assert described[0].description == "A retired welder from Ohio."


def test_personas_from_rows():
    rows = [{"id": "r1", "age": 30, "party": "D"}, {"id": "r2", "age": 60, "party": "R"}]
    ps = personas_from_rows(rows, fields=["age", "party"], name_key="id")
    assert ps[1].name == "r2" and ps[1].attributes == {"age": 60, "party": "R"}
    assert len(personas_from_rows(rows, sample=1)) == 1


def test_min_turns_ignores_early_end():
    a = ScriptedAgent("a", ["Hi there. [END]", "Bye. [END]"])
    b = ScriptedAgent("b", ["Hello."])
    res = Simulation(Conversation("x", min_turns=2), [a, b], max_steps=10).run()
    assert [m["text"] for m in res.metrics["transcript"]] == ["Hi there.", "Hello.", "Bye."]
    assert res.termination == "ended by a"


def test_truncation_reported():
    llm = ScriptedLLM([LLMResponse("", reasoning="long thoughts", finish_reason="length")] * 3)
    turn = Agent("a", llm, output_schema={"reply": str}).act("hi")
    assert "cut off at max_tokens" in turn.error and "thinking" in turn.error


def test_structured_cut_off_is_explained_and_not_retried():
    llm = ScriptedLLM([LLMResponse("", reasoning="long thoughts", finish_reason="length")] * 5)
    data, _, errors = structured_chat(llm, [{"role": "user", "content": "go"}], {"x": str})
    assert data is None and len(llm.calls) == 1 and "cut off at max_tokens" in errors[0] and "thinking" in errors[0]
