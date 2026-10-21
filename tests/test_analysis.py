import json
import math

import pytest

pd = pytest.importorskip("pandas")

from simsim import (  # noqa: E402
    END,
    Agent,
    Conversation,
    Experiment,
    FullHistory,
    InnerState,
    Judge,
    ScriptedAgent,
    ScriptedLLM,
    Simulation,
    at,
    grid,
    probe,
    survey,
)
from simsim.analysis import (  # noqa: E402
    cost_report,
    expected_value,
    load_probes,
    load_runs,
    load_turns,
    parse_condition,
    pass_hat_k,
    summarize,
    transcript,
)
from simsim.viewer import write_viewer  # noqa: E402

LIKERT = {str(i): str(i) for i in range(1, 6)}


def build(cond, seed):
    def script(m, t):
        content = m[-1]["content"]
        if "label only" in content:
            return "4" if cond["style"] == "warm" else "2"
        if "Update your state" in content:
            return json.dumps({"trust": 5 if cond["style"] == "warm" else 2})
        if "Rate the" in content:
            return json.dumps({"score": 8 if cond["style"] == "warm" else 3})
        return json.dumps({"thoughts": "hmm", "reply": f"Hello from {cond['style']}"})

    llm = ScriptedLLM(script, price=(1.0, 2.0))
    a = Agent("A", llm, memory=FullHistory(), output_schema={"thoughts": str, "reply": str},
              components=[InnerState({"trust": (int, 1, 7)})])
    b = ScriptedAgent("B", ["Hi.", "Bye."])
    judge = Judge(ScriptedLLM(script), rubric="Rate the chat.", schema={"score": (int, 1, 10)})
    return Simulation(Conversation("Chat.", end_token=None), [a, b], max_steps=4, seed=seed,
                      interventions=[at(0, survey("I trust B.", LIKERT, agents=["A"])),
                                     at(END, probe("Rate the other person", schema={"score": (int, 1, 10)}, agents=["A"]))],
                      evaluators={"judge": judge.evaluator()})


@pytest.fixture(scope="module")
def exp_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("exp")
    Experiment(build, grid(style=["warm", "cold"]), seeds=range(3), out=out).run(progress=False)
    return out


def test_parse_condition():
    assert parse_condition("model=a,memory=none") == {"model": "a", "memory": "none"}
    assert parse_condition("plain") == {}


def test_load_runs_and_summarize(exp_dir):
    runs = load_runs(exp_dir)
    assert len(runs) == 6 and set(runs["style"]) == {"warm", "cold"}
    assert "judge.score" in runs and runs.loc[runs["style"] == "warm", "judge.score"].eq(8).all()
    assert "transcript" not in runs.columns and runs["usage.cost_usd"].ge(0).all()
    s = summarize(runs, by="style", metrics="judge.score")
    warm = s[s["style"] == "warm"].iloc[0]
    assert warm["mean"] == 8 and warm["n"] == 3 and warm["sem"] == 0


def test_load_turns_has_private_and_components(exp_dir):
    turns = load_turns(exp_dir)
    a = turns[turns["agent"] == "A"]
    assert len(turns) == 6 * 4 and (a["private.thoughts"] == "hmm").all()
    assert set(a["component.InnerState.trust"]) == {5, 2}


def test_load_probes_expected_and_answers(exp_dir):
    probes = load_probes(exp_dir)
    surveys = probes[probes["name"] == "survey"]
    assert set(surveys["top"]) == {"4", "2"} and set(surveys["expected"]) == {4.0, 2.0}
    rated = probes[probes["name"] == "probe"]
    assert set(rated["answer.score"]) == {8, 3}


def test_expected_value_non_numeric():
    assert expected_value({"A": 0.5, "B": 0.5}) is None


def test_pass_hat_k():
    df = pd.DataFrame({"condition": ["t1"] * 4 + ["t2"] * 4, "success": [1, 1, 1, 1, 1, 0, 1, 0]})
    assert pass_hat_k(df, "success", 1) == pytest.approx((1 + 0.5) / 2)
    assert pass_hat_k(df, "success", 2) == pytest.approx((1 + 1 / 6) / 2)
    assert math.isnan(pass_hat_k(df, "success", 5))


def test_cost_report(exp_dir):
    runs = load_runs(exp_dir)
    report = cost_report(runs, by="style")
    assert set(report.columns) >= {"style", "usage.calls", "usage.cost_usd"}


def test_transcript_and_viewer(exp_dir):
    log = next(exp_dir.rglob("seed0.jsonl"))
    text = transcript(log, private=True)
    assert "A: Hello from" in text and "(private)" in text
    page = write_viewer(exp_dir)
    content = page.read_text()
    assert page.name == "view.html" and "Hello from warm" in content and "<\\/script" not in content.split("const RUNS")[0]
    assert content.count("<script>") == 1
