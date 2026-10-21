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

from simsim import Agent, Conversation, FullHistory, Simulation, make_llm, tool

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


CHECKS = [("chat", check_chat), ("native tool calls", check_tools), ("react tool use", check_react),
          ("structured output", check_structured), ("survey (log-probs)", check_survey),
          ("conversation", check_conversation)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--no-thinking", action="store_true", help="disable thinking on Qwen3-style models")
    ap.add_argument("--skip", nargs="*", default=[])
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
    print(f"\nusage: {llm.usage}")
    sys.exit(1 if failed else 0)
