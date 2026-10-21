import json
from datetime import timedelta

import pytest

from simsim import (
    Agent,
    CachedLLM,
    Clock,
    Conversation,
    FullHistory,
    Goal,
    HashEmbedder,
    Impressions,
    InnerState,
    LLMResponse,
    MarkdownMemory,
    MemoryEntry,
    NextSpeaker,
    Plan,
    ReflectiveMemory,
    ScoredMemory,
    ScriptedAgent,
    ScriptedLLM,
    Simulation,
    SummaryMemory,
    bernoulli,
    by_hour,
    emulated_tool,
    everyone,
    probe,
    at,
)
from simsim.clock import parse_step


# ---- embeddings and scored memory --------------------------------------------

def test_hash_embedder_similarity_and_cache():
    emb = HashEmbedder()
    assert emb.similarity("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert emb.similarity("the cat sat", "cat sat the") == pytest.approx(1.0)
    assert emb.similarity("bank account password", "lunch was pasta") < 0.75
    emb.embed(["x"])
    assert "x" in emb._cache


def test_scored_memory_with_embedder_and_access():
    mem = ScoredMemory(k=1, embedder=HashEmbedder(), weights=(0, 0, 1))
    mem.write(MemoryEntry(0, "my bank password is swordfish"))
    mem.write(MemoryEntry(1, "we had pasta for lunch"))
    assert "swordfish" in mem.read("what is my bank password")
    assert mem.entries[0].meta["last_access"] == 1  # retrieval counts as access
    mem.write(MemoryEntry(5, "later event"))
    mem.peek("bank password")
    assert mem.entries[0].meta["last_access"] == 1  # peek does not


def test_scored_memory_normalization_lets_weights_compare():
    mem = ScoredMemory(k=1, importance=lambda text: 0.9 if "important" in text else 0.1, weights=(0, 1, 0))
    mem.write(MemoryEntry(0, "important thing"))
    mem.write(MemoryEntry(1, "trivial thing"))
    assert "important" in mem.read()


def test_reflective_memory_writes_insights():
    llm = ScriptedLLM(lambda m, t: "What does Bob value?" if "questions" in m[-1]["content"]
                      else "Bob values honesty.\nBob dislikes delays.")
    mem = ReflectiveMemory(llm, threshold=1.0, n_questions=1, n_insights=2, importance=lambda text: 0.6)
    mem.write(MemoryEntry(0, "Bob said he hates being lied to."))
    assert not any(e.kind == "reflection" for e in mem.entries)
    mem.write(MemoryEntry(1, "Bob was angry the delivery was late."))  # budget 1.2 >= 1.0
    refl = [e for e in mem.entries if e.kind == "reflection"]
    assert [e.text for e in refl] == ["Bob values honesty.", "Bob dislikes delays."]
    assert refl[0].meta["question"] == "What does Bob value?"


def test_memory_dump_load_roundtrip(tmp_path):
    for mem, fresh in [
        (FullHistory(), FullHistory()),
        (ScoredMemory(), ScoredMemory()),
        (SummaryMemory(ScriptedLLM(), every=100), SummaryMemory(ScriptedLLM(), every=100)),
        (MarkdownMemory(initial="note"), MarkdownMemory()),
    ]:
        mem.write(MemoryEntry(3, "something happened"))
        fresh.load(json.loads(json.dumps(mem.dump())))
        assert fresh.read() == mem.read(), type(mem)


# ---- components --------------------------------------------------------------

def test_components_in_prompt_and_logged():
    def script(messages, tools):
        content = messages[-1]["content"]
        if "Write your plan" in content:
            return "- open low\n- concede slowly"
        if "Update your state" in content:
            return '{"mood": "annoyed", "trust": 3}'
        if "impression of Seller" in content:
            return "The seller seems pushy."
        return "I offer $100."

    llm = ScriptedLLM(script)
    buyer = Agent("Buyer", llm, components=[
        Goal("Pay at most $150."), Plan(every=2), Impressions(),
        InnerState({"mood": ["calm", "annoyed"], "trust": (int, 1, 7)}),
    ])
    seller = ScriptedAgent("Seller", ["$200, firm.", "Still $200."])
    res = Simulation(Conversation("Bike sale.", end_token=None), [seller, buyer], max_steps=4).run()
    system = [m for m in llm.calls if m[0]["role"] == "system" and "# Your goal" in m[0]["content"]][-1][0]["content"]
    for part in ["# Your goal\nPay at most $150.", "# Your current plan\n- open low", "# Your current state\n- mood: annoyed",
                 "# Your impressions of others\n- Seller: The seller seems pushy."]:
        assert part in system, part
    buyer_turn = [t for t in res.turns if t.agent == "Buyer"][-1]
    assert buyer_turn.components["InnerState"] == {"mood": "annoyed", "trust": 3}
    assert buyer_turn.components["Impressions"] == {"Seller": "The seller seems pushy."}


def test_plan_regenerates_every_k_turns():
    plans = iter(["plan A", "plan B"])
    llm = ScriptedLLM(lambda m, t: next(plans) if "Write your plan" in m[-1]["content"] else "ok")
    a = Agent("a", llm, components=[Plan(every=2)])
    seen = []
    for i in range(4):
        a.act(f"obs {i}", t=i)
        seen.append(a.components[0].plan)
    assert seen == ["plan A", "plan A", "plan B", "plan B"]


def test_probes_do_not_change_components():
    calls = []
    llm = ScriptedLLM(lambda m, t: calls.append(1) or "plan" if "Write your plan" in m[-1]["content"] else "fine")
    a = Agent("a", llm, components=[Plan(every=1)])
    a.ask("How are you?")
    assert calls == [] and a.components[0].plan == ""


def test_agent_state_save_restore(tmp_path):
    llm = ScriptedLLM(lambda m, t: '{"mood": "happy"}' if "Update your state" in m[-1]["content"] else "hi")
    a = Agent("a", llm, memory=FullHistory(), components=[InnerState({"mood": ["happy", "sad"]})])
    a.act("hello")
    a.save(tmp_path / "a.json")
    b = Agent("a", ScriptedLLM(), memory=FullHistory(), components=[InnerState({"mood": ["happy", "sad"]})])
    b.restore(tmp_path / "a.json")
    assert b.memory.read() == a.memory.read() and b.components[0].values == {"mood": "happy"}


# ---- cache -------------------------------------------------------------------

def test_cached_llm_replays_in_order(tmp_path):
    counter = iter(range(100))
    inner = ScriptedLLM(lambda m, t: LLMResponse(f"sample {next(counter)}", usage={"prompt_tokens": 1}))
    msgs = [{"role": "user", "content": "same"}]
    first = CachedLLM(inner, tmp_path / "c.sqlite")
    a = [first.chat(msgs).text for _ in range(3)]
    assert a == ["sample 0", "sample 1", "sample 2"]  # repeated sampling still differs
    second = CachedLLM(inner, tmp_path / "c.sqlite")  # a rerun of the same code
    assert [second.chat(msgs).text for _ in range(3)] == a and second.hits == 3
    assert second.chat(msgs).text == "sample 3"  # beyond the cache: goes to the model
    assert second.chat([{"role": "user", "content": "different"}]).text == "sample 4"


def test_cached_tool_calls_and_probs(tmp_path):
    from simsim import ToolCall

    inner = ScriptedLLM([LLMResponse(tool_calls=[ToolCall("1", "f", {"x": 1})]), "B"])
    c = CachedLLM(inner, tmp_path / "c.sqlite")
    c.chat([{"role": "user", "content": "q"}])
    assert c.choice_probs([{"role": "user", "content": "p"}], ["A", "B"]) == {"A": 0.0, "B": 1.0}
    c2 = CachedLLM(ScriptedLLM([]), tmp_path / "c.sqlite")
    assert c2.chat([{"role": "user", "content": "q"}]).tool_calls[0].arguments == {"x": 1}
    assert c2.choice_probs([{"role": "user", "content": "p"}], ["A", "B"])["B"] == 1.0


# ---- rate limits -------------------------------------------------------------

def test_rpm_spaces_calls(monkeypatch):
    sleeps = []
    t = [0.0]
    monkeypatch.setattr("simsim.llm.time.monotonic", lambda: t[0])
    monkeypatch.setattr("simsim.llm.time.sleep", lambda s: sleeps.append(s))
    llm = ScriptedLLM(rpm=60)
    for _ in range(3):
        llm.chat([])
    assert sleeps == [1.0, 2.0]


def test_max_concurrency_limits_parallel_calls():
    import threading
    import time

    active, peak, lock = [0], [0], threading.Lock()

    def slow(m, t):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return "ok"

    shared = ScriptedLLM(slow, max_concurrency=2)
    agents = [Agent(f"a{i}", shared) for i in range(6)]
    Simulation(Conversation("x", end_token=None), agents, scheduler=everyone, simultaneous=True, max_steps=1).run()
    assert peak[0] == 2


# ---- clock and schedulers ----------------------------------------------------

def test_clock_and_timestamps():
    assert parse_step("15min") == timedelta(minutes=15) and parse_step("2 days") == timedelta(days=2)
    with pytest.raises(ValueError):
        parse_step("soon")
    clock = Clock("2024-03-04 08:00", "30min", fmt="%H:%M")
    llm = ScriptedLLM(lambda m, t: "ok")
    a = Agent("a", llm, memory=FullHistory())
    res = Simulation(Conversation("x", end_token=None), [a], max_steps=2, clock=clock).run()
    assert "# Current time\n08:30" in llm.calls[1][0]["content"]
    assert res.turns[1].time == "08:30" and "[08:00]" in a.memory.read()


def test_by_hour_activity():
    clock = Clock("2024-01-01 00:00", "1h")
    agents = [ScriptedAgent(f"a{i}", ["x"]) for i in range(5)]
    activity = {h: (1.0 if 9 <= h < 17 else 0.0) for h in range(24)}
    res = Simulation(Conversation("x", end_token=None), agents, scheduler=by_hour(activity),
                     max_steps=24, clock=clock).run()
    assert {t.t for t in res.turns} == set(range(9, 17)) and len(res.turns) == 40


def test_bernoulli_scheduler():
    agents = [ScriptedAgent(f"a{i}", ["x"]) for i in range(10)]
    res = Simulation(Conversation("x", end_token=None), agents, scheduler=bernoulli(0.0), max_steps=3).run()
    assert res.turns == []


def test_next_speaker_moderator():
    mod = NextSpeaker(ScriptedLLM(lambda m, t: "Carol should speak."))
    agents = [ScriptedAgent(n, [f"{n} talks"]) for n in ["Alice", "Bob", "Carol"]]
    res = Simulation(Conversation("x", end_token=None), agents, scheduler=mod, max_steps=3).run()
    assert [t.agent for t in res.turns] == ["Alice", "Carol", "Alice"]  # no immediate repeats: falls back


# ---- emulated tools ----------------------------------------------------------

def test_emulated_tool_consistency_context():
    llm = ScriptedLLM(lambda m, t: '{"balance": 40}')
    check = emulated_tool(llm, "get_balance", "Get the account balance.", {"account": str}, world="Balance is $40.")
    assert check.schema()["function"]["parameters"]["required"] == ["account"]
    assert check(account="main") == '{"balance": 40}'
    check(account="main")
    second_prompt = llm.calls[1][-1]["content"]
    assert "Balance is $40." in second_prompt and 'get_balance({"account": "main"})' in second_prompt


# ---- checkpoints -------------------------------------------------------------

def test_checkpoint_and_resume(tmp_path):
    path = tmp_path / "sim.ckpt"
    a = Agent("a", ScriptedLLM(lambda m, t: "turn"), memory=FullHistory())

    class Crash(Exception):
        pass

    flag = tmp_path / "crashed"

    def crash(sim, t):  # crashes the first run only
        if not flag.exists():
            flag.write_text("1")
            raise Crash()

    sim = Simulation(Conversation("x", end_token=None), [a], max_steps=6, checkpoint=path, checkpoint_every=2,
                     interventions=[at(5, crash)])
    with pytest.raises(Crash):
        sim.run()
    assert path.exists()
    import cloudpickle  # noqa: F401  (lambdas in this setup need it)

    res = Simulation.resume(path)
    assert len(res.turns) == 6 and res.steps == 6 and not path.exists()
    assert res.turns[-1].t == 5


def test_request_seed_varies_but_reproduces():
    from simsim import LLM

    a, b = LLM(seed=7), LLM(seed=7)
    m1, m2 = [{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]
    seq_a = [a.request_seed(m1), a.request_seed(m1), a.request_seed(m2)]
    seq_b = [b.request_seed(m1), b.request_seed(m1), b.request_seed(m2)]
    assert seq_a == seq_b and len(set(seq_a)) == 3  # reproducible, yet each call differs
    assert LLM(seed=8).request_seed(m1) != seq_a[0] and LLM().request_seed(m1) is None
