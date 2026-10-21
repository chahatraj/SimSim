import json
import textwrap

import pytest

from simsim.cli import main
from simsim.config import ConfigError, ConfigExperiment, load_config, resolve, substitute, to_schema
from simsim.record import read_jsonl


def test_substitute():
    vals = {"style": "tough", "persona": {"description": "x"}}
    assert substitute("You are ${style}.", vals) == "You are tough."
    assert substitute("${persona}", vals) == {"description": "x"}
    assert substitute(["${style}", {"a": "${missing}"}], vals) == ["tough", {"a": "${missing}"}]


def test_to_schema():
    s = to_schema({"a": "str", "b": ["x", "y"], "c": ["int", 1, 7], "d": {"type": "float", "min": 0, "max": 1},
                   "e": {"type": "str", "description": "private"}, "f": {"choices": ["p", "q"]}})
    assert s == {"a": str, "b": ["x", "y"], "c": (int, 1, 7), "d": (float, 0, 1), "e": (str, "private"), "f": ["p", "q"]}
    with pytest.raises(ConfigError):
        to_schema({"a": "banana"})


def test_resolve(tmp_path):
    (tmp_path / "mytools.py").write_text("def hello():\n    return 'hi'\n")
    assert resolve("Conversation").__name__ == "Conversation"
    assert resolve("mytools:hello", str(tmp_path))() == "hi"
    with pytest.raises(ConfigError):
        resolve("NoSuchThing")


CONFIG = """
name: t
seeds: 2
models:
  m: {spec: "openai:gpt-4o-mini"}
grid:
  style: {calm: "Stay calm.", rude: "Be rude."}
agents:
  - name: A
    model: m
    persona: "You are A. ${style}"
    memory: {type: WindowMemory, k: 3}
    components: [{type: Goal, text: "Win."}, {type: InnerState, schema: {mood: [calm, angry]}}]
    output_schema: {thoughts: str, reply: str}
  - name: B
    model: m
    persona: {description: "You are B.", age: 30}
    memory: FullHistory
environment:
  type: Conversation
  scenario: "Talk."
  end_token: null
  max_turns: 4
simulation: {max_steps: 4, scheduler: round_robin}
probes:
  - {type: survey, question: "I like B.", options: likert5, agents: [A], at: [0, end]}
  - {type: probe, question: "Score B", schema: {score: [int, 1, 10]}, at: end}
  - {type: whisper, agent: A, text: "B insulted you.", at: 2}
judge:
  model: m
  rubric: "Rate it. {text}"
  schema: {reasoning: str, score: [int, 1, 10]}
"""


def test_config_builds_everything(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(CONFIG)
    ce = ConfigExperiment(load_config(path), dry_run=True)
    assert list(ce.conditions()) == ["style=calm", "style=rude"]
    sim = ce.build({"style": "Be rude."}, seed=0)
    a, b = sim.agents
    assert "Be rude." in a.persona.description and a.memory.k == 3 and b.persona.attributes == {"age": 30}
    assert [type(c).__name__ for c in a.components] == ["Goal", "InnerState"]
    assert a.output_schema == {"thoughts": str, "reply": str}
    res = sim.run()
    assert len(res.turns) == 4 and res.metrics["judge"]["score"] == 1
    assert [p["name"] for p in res.probes] == ["survey", "survey", "probe", "probe"]
    assert "B insulted you." in a.memory.read()


def test_population_config(tmp_path):
    path = tmp_path / "pop.yaml"
    path.write_text(textwrap.dedent("""
        models: {m: "openai:gpt-4o-mini"}
        population:
          count: 5
          attributes: {age: [20, 30, 40], party: {D: 0.5, R: 0.5}}
          agent: {model: m, memory: WindowMemory, persona: "You use social media."}
        environment: {type: Feed, news: ["Rain today."]}
        simulation: {max_steps: 2, scheduler: everyone, simultaneous: true}
    """))
    sim = ConfigExperiment(load_config(path), dry_run=True).build({}, seed=1)
    assert len(sim.agents) == 5 and "You use social media." in sim.agents[0].persona.description
    assert set(sim.agents[0].persona.attributes) == {"age", "party"}
    assert len(sim.run().turns) == 10


def test_cli_new_run_view_summarize(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["new", "study"]) == 0
    monkeypatch.chdir(tmp_path / "study")
    assert main(["run", "experiment.yaml", "--dry-run", "--seeds", "1", "--view"]) == 0
    assert (tmp_path / "study/runs/study/view.html").exists()
    assert (tmp_path / "study/runs/study/config.yaml").exists()
    rows = read_jsonl(tmp_path / "study/runs/study/results.jsonl")
    assert len(rows) == 2 and not any(r.get("error") for r in rows)
    assert main(["run", "experiment.py", "--dry-run", "--seeds", "1"]) == 0
    capsys.readouterr()
    pytest.importorskip("pandas")
    assert main(["summarize", "runs/study", "--by", "style", "--metric", "n_messages"]) == 0
    assert "n_messages" in capsys.readouterr().out
    log = next((tmp_path / "study/runs/study").rglob("seed0.jsonl"))
    assert main(["transcript", str(log)]) == 0 and "Buyer:" in capsys.readouterr().out


def test_cli_new_refuses_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["new", "x"])
    with pytest.raises(SystemExit):
        main(["new", "x"])
