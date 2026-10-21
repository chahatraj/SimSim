"""Check that a model endpoint supports everything SimSim uses.

    python scripts/smoke_test.py vllm:meta-llama/Llama-3.1-8B-Instruct@http://gpu027:8000/v1
    python scripts/smoke_test.py openai:gpt-4o-mini
    python scripts/smoke_test.py vllm:Qwen/Qwen3.5-9B@http://gpu027:8001/v1 --no-thinking

Prints PASS/FAIL per feature and exits non-zero if any check fails.
"""

import argparse
import sys
import time
import traceback

import tempfile

from simsim import (
    Agent, CachedLLM, Conversation, FullHistory, Impressions, InnerState, MemoryEntry, NextSpeaker, OpenAIEmbedder,
    Plan, ReflectiveMemory, ScoredMemory, Simulation, make_llm, tool,
)

WEATHER = {"paris": "rainy, 14C", "cairo": "sunny, 31C"}


@tool
def get_weather(city: str) -> str:
    """Get today's weather for a city."""
    return WEATHER.get(city.lower().strip(), "unknown city")


def check_chat(llm):
    out = llm.complete("Reply with exactly the word: pong")
    assert "pong" in out.lower(), out
    return out.strip()[:60]


def check_tools(llm):
    turn = Agent("assistant", llm, tools=[get_weather]).act("What's the weather in Cairo today? Use the tool.")
    assert turn.tool_calls and turn.tool_calls[0]["name"] == "get_weather", turn.tool_calls
    assert "31" in turn.text or "sunny" in turn.text.lower(), turn.text
    return f"{turn.tool_calls[0]['arguments']} -> {turn.text[:60]!r}"


def check_react(llm):
    turn = Agent("assistant", llm, tools=[get_weather], action_format="react").act("What's the weather in Paris today?")
    assert turn.tool_calls and "rain" in turn.text.lower() + turn.tool_calls[0]["result"], (turn.tool_calls, turn.text)
    return f"{len(turn.tool_calls)} tool call(s) -> {turn.text[:60]!r}"


def check_structured(llm):
    schema = {"analysis": (str, "private reasoning"), "reply": str, "offer": (int, 0, 1000)}
    turn = Agent("buyer", llm, persona="You are buying a used bike listed at $200.", output_schema=schema).act(
        "Seller: The price is $200. What do you offer?")
    assert not turn.error and isinstance(turn.private["offer"], int), (turn.error, turn.text)
    return f"offer={turn.private['offer']} reply={turn.text[:50]!r}"


def check_survey(llm):
    a = Agent("respondent", llm, persona="You love cats and dislike dogs.")
    probs = a.survey("Do you like dogs?", {"1": "Not at all", "2": "Somewhat", "3": "Very much"})
    assert abs(sum(probs.values()) - 1) < 1e-6 and max(probs, key=probs.get) == "1", probs
    return {k: round(v, 3) for k, v in probs.items()}


def check_conversation(llm):
    buyer = Agent("Buyer", llm, persona="You want to buy a used bike.", memory=FullHistory())
    seller = Agent("Seller", llm, persona="You are selling a used bike.", memory=FullHistory())
    env = Conversation("Negotiate the price of a bike listed at $200. Keep messages short.",
                       private={"Buyer": "Your budget is $150."}, max_turns=6)
    res = Simulation(env, [buyer, seller], max_steps=6).run()
    assert res.metrics["n_messages"] >= 2
    return f"{res.metrics['n_messages']} messages, termination={res.termination}"


def check_components(llm):
    buyer = Agent("Buyer", llm, persona="You want to buy a used bike cheaply.", components=[
        Plan(every=3), Impressions(), InnerState({"mood": ["calm", "annoyed", "pleased"], "trust_in_seller": (int, 1, 7)}),
    ])
    seller = Agent("Seller", llm, persona="You are selling a used bike and will not go below $180.")
    env = Conversation("Negotiate the price of a bike listed at $200. Keep messages short.", end_token=None)
    res = Simulation(env, [seller, buyer], max_steps=4).run()
    state = [t for t in res.turns if t.agent == "Buyer"][-1].components
    assert buyer.components[0].plan and "Seller" in state["Impressions"] and "trust_in_seller" in state["InnerState"], state
    return f"plan={buyer.components[0].plan[:40]!r} state={state['InnerState']}"


def check_reflection(llm):
    mem = ReflectiveMemory(llm, threshold=1.5, n_questions=1, n_insights=2, importance=lambda text: 0.8)
    for i, text in enumerate(["Maria lent me her car when mine broke down.",
                              "Maria stayed late to help me finish the report."]):
        mem.write(MemoryEntry(i, text))
    refl = [e.text for e in mem.entries if e.kind == "reflection"]
    assert refl, "no reflections written"
    return refl[0][:80]


def check_next_speaker(llm):
    names = ["Alice", "Bob", "Carol"]
    agents = [Agent(n, llm, persona=f"You are {n}. Keep messages to one sentence.") for n in names]
    mod = NextSpeaker(llm, guidance="Pick whoever was just asked a question.")
    res = Simulation(Conversation("Plan a team lunch. Ask each other questions.", end_token=None), agents,
                     scheduler=mod, max_steps=4).run()
    assert len(res.turns) == 4
    return " -> ".join(t.agent for t in res.turns)


def check_cache(llm):
    with tempfile.TemporaryDirectory() as d:
        msgs = [{"role": "user", "content": "Name a random fruit, one word."}]
        with CachedLLM(llm, f"{d}/c.sqlite") as c:
            first = c.chat(msgs).text
        with CachedLLM(llm, f"{d}/c.sqlite") as again:
            assert again.chat(msgs).text == first and again.hits == 1
    return f"replayed {first!r}"


CHECKS = [("chat", check_chat), ("native tool calls", check_tools), ("react tool use", check_react),
          ("structured output", check_structured), ("survey (log-probs)", check_survey),
          ("conversation", check_conversation), ("components", check_components), ("reflection", check_reflection),
          ("next speaker", check_next_speaker), ("cache", check_cache)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--no-thinking", action="store_true", help="disable thinking on Qwen3-style models")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--embed", help="embedding endpoint spec, e.g. BAAI/bge-small-en-v1.5@http://node:8002/v1")
    args = ap.parse_args()

    kw = {"temperature": 0, "max_tokens": 512, "retries": 2}
    if args.no_thinking:
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    llm = make_llm(args.model, **kw)
    failed = 0
    for name, fn in CHECKS:
        if name in args.skip:
            continue
        start = time.time()
        try:
            detail = fn(llm)
            print(f"PASS  {name:20s} ({time.time() - start:.1f}s)  {detail}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name:20s} ({time.time() - start:.1f}s)  {type(e).__name__}: {str(e)[:300]}")
            if not isinstance(e, AssertionError):
                traceback.print_exc(limit=3)
    if args.embed:
        model, _, url = args.embed.partition("@")
        start = time.time()
        try:
            mem = ScoredMemory(k=1, embedder=OpenAIEmbedder(model, base_url=url or None), weights=(0, 0, 1))
            mem.write(MemoryEntry(0, "My sister's wedding is in June in Lisbon."))
            mem.write(MemoryEntry(1, "The quarterly budget report is due Friday."))
            got = mem.read("When is the family celebration?")
            assert "wedding" in got, got
            print(f"PASS  {'embedding retrieval':20s} ({time.time() - start:.1f}s)  {got[:60]!r}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {'embedding retrieval':20s} {type(e).__name__}: {str(e)[:300]}")
    print(f"\nusage: {llm.usage}")
    sys.exit(1 if failed else 0)
