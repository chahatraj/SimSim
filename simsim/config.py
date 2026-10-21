"""Experiments described in YAML (or JSON) instead of Python.

    simsim run experiment.yaml

A config names its models, the factors to cross, the agents (or a generated
population), the environment, probes, and an optional judge. Anywhere in the
config, `${factor}` is replaced by the current condition's value for that
factor; a string that is exactly `${factor}` takes the value as is (so a factor
can hold a whole dict). Classes and functions are looked up in `simsim` by
name, or imported from your own code with "module:attribute" (e.g. a tool
"mytools:get_weather" or a scoring function "scoring:refund_done").

See examples/configs/ for complete configs.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import schedule as schedule_mod
from .agent import Agent, Persona
from .components import Component
from .experiment import Experiment, grid
from .judge import Judge
from .llm import LLM, ScriptedLLM, make_llm
from .memory import Memory, NoMemory
from .personas import sample_personas
from .sim import END, Simulation, at, every, probe, survey, whisper

LIKERT = {
    "likert5": {"1": "Strongly disagree", "2": "Disagree", "3": "Neutral", "4": "Agree", "5": "Strongly agree"},
    "likert7": {"1": "Strongly disagree", "2": "Disagree", "3": "Slightly disagree", "4": "Neutral",
                "5": "Slightly agree", "6": "Agree", "7": "Strongly agree"},
}
_TYPES = {"str": str, "string": str, "int": int, "integer": int, "float": float, "number": float, "bool": bool}


class ConfigError(ValueError):
    pass


# ---- helpers -------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        cfg = json.loads(text)
    else:
        import yaml

        cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    cfg.setdefault("_dir", str(path.parent.resolve()))
    return cfg


def resolve(name: str, base_dir: str | None = None) -> Any:
    """'Conversation' -> simsim.Conversation; 'pkg.mod:attr' -> imported attribute."""
    if ":" in name:
        module, _, attr = name.partition(":")
        if base_dir and base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        obj = importlib.import_module(module)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj
    import simsim

    if hasattr(simsim, name):
        return getattr(simsim, name)
    raise ConfigError(f"Unknown name {name!r}: not in simsim; use 'module:attribute' for your own code")


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute(obj: Any, values: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        whole = _VAR.fullmatch(obj)
        if whole and whole.group(1) in values:
            return copy.deepcopy(values[whole.group(1)])
        return _VAR.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), obj)
    if isinstance(obj, list):
        return [substitute(x, values) for x in obj]
    if isinstance(obj, dict):
        return {k: substitute(v, values) for k, v in obj.items()}
    return obj


def to_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """YAML schema -> Python schema. Types are names ("str", "int"); a list of strings is a
    choice; [int, 1, 7] or {type: int, min: 1, max: 7} is a range; {..., description: ...} adds one."""
    out = {}
    for name, s in spec.items():
        desc = ""
        if isinstance(s, dict):
            desc = s.get("description", "")
            if "choices" in s:
                s = list(s["choices"])
            elif "min" in s or "max" in s:
                s = [s.get("type", "int"), s.get("min"), s.get("max")]
            else:
                s = s.get("type", "str")
        if isinstance(s, str):
            if s not in _TYPES:
                raise ConfigError(f"schema field {name!r}: unknown type {s!r}")
            v: Any = _TYPES[s]
        elif isinstance(s, list) and len(s) == 3 and isinstance(s[0], str) and s[0] in _TYPES and \
                all(isinstance(x, (int, float)) for x in s[1:]):
            v = (_TYPES[s[0]], s[1], s[2])
        elif isinstance(s, list):
            v = [str(x) for x in s]
        else:
            raise ConfigError(f"schema field {name!r}: cannot read {s!r}")
        out[name] = (v, desc) if desc else v
    return out


def _construct(spec: Any, base: type, base_dir: str | None, llm: LLM | None = None, schema_keys=("schema",)):
    """Build an object from "Name" or {type: Name, ...kwargs}; passes the agent's llm if the class wants one."""
    if isinstance(spec, (base,)):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = dict(spec)
    cls = resolve(spec.pop("type"), base_dir)
    for k in schema_keys:
        if k in spec and isinstance(spec[k], dict):
            spec[k] = to_schema(spec[k])
    params = inspect.signature(cls).parameters
    if "llm" in params and "llm" not in spec and llm is not None:
        spec["llm"] = llm
    for key in ("embedder",):
        if isinstance(spec.get(key), (str, dict)):
            espec = spec[key] if isinstance(spec[key], dict) else {"type": spec[key]}
            espec = dict(espec)
            spec[key] = resolve(espec.pop("type"), base_dir)(**espec)
    return cls(**spec)


# ---- the builder ---------------------------------------------------------------

class ConfigExperiment:
    """Turns a config into `Experiment(build, conditions, seeds, out)`."""

    def __init__(self, cfg: dict[str, Any], dry_run: bool = False):
        self.cfg, self.dry_run = cfg, dry_run
        self.base_dir = cfg.get("_dir")
        for key in ("agents", "environment"):
            if key not in cfg and not (key == "agents" and "population" in cfg):
                raise ConfigError(f"config needs '{key}'")

    def conditions(self) -> dict[str, dict[str, Any]]:
        factors = self.cfg.get("grid") or {}
        return grid(**factors) if factors else {"default": {}}

    def model(self, ref: Any, seed: int) -> LLM:
        if self.dry_run:
            return ScriptedLLM(_dry_script)
        models = self.cfg.get("models") or {}
        spec = models.get(ref, ref) if isinstance(ref, str) else ref
        if isinstance(spec, str):
            spec = {"spec": spec}
        spec = dict(spec)
        if "spec" not in spec:
            raise ConfigError(f"model {ref!r} needs a 'spec' like 'openai:gpt-4o-mini'")
        cache = spec.pop("cache", None)
        llm = make_llm(spec.pop("spec"), seed=spec.pop("seed", seed), **spec)
        if cache:
            from .cache import CachedLLM

            llm = CachedLLM(llm, Path(self.base_dir or ".") / cache)
        return llm

    def _agent(self, spec: dict[str, Any], seed: int, persona: Persona | None = None) -> Agent:
        spec = dict(spec)
        llm = self.model(spec.pop("model", self.cfg.get("default_model", "default")), seed)
        memory = _construct(spec.pop("memory", "NoMemory"), Memory, self.base_dir, llm)
        components = [_construct(c, Component, self.base_dir, None) for c in spec.pop("components", [])]
        tools = [resolve(t, self.base_dir) for t in spec.pop("tools", [])]
        if "output_schema" in spec:
            spec["output_schema"] = to_schema(spec["output_schema"])
        p = spec.pop("persona", "")
        if persona is None:
            persona = Persona(spec["name"], p.get("description", ""), {k: v for k, v in p.items()
                              if k not in ("description", "name")}) if isinstance(p, dict) else p
        elif isinstance(p, str) and p:  # population: a shared persona text prefix
            persona.description = f"{p}\n{persona.description}".strip()
        return Agent(spec.pop("name"), llm, persona=persona, memory=memory, components=components, tools=tools, **spec)

    def agents(self, seed: int) -> list[Agent]:
        agents = [self._agent(a, seed) for a in self.cfg.get("agents") or []]
        pop = self.cfg.get("population")
        if pop:
            pop = dict(pop)
            template = pop.pop("agent", {})
            if "file" in pop:
                personas = Persona.load_jsonl(Path(self.base_dir or ".") / pop.pop("file"))
                if "count" in pop:
                    import random

                    personas = random.Random(seed).sample(personas, pop.pop("count"))
            else:
                describe = pop.pop("describe", None)
                personas = sample_personas(pop.pop("count"), pop.pop("attributes"), seed=seed,
                                           describe=self.model(describe, seed) if describe else None)
            for p in personas:
                agents.append(self._agent({**template, "name": p.name}, seed, persona=p))
        return agents

    def environment(self, seed: int):
        spec = dict(self.cfg["environment"])
        cls = resolve(spec.pop("type"), self.base_dir)
        for key in ("score", "judge"):
            if isinstance(spec.get(key), str):
                spec[key] = resolve(spec[key], self.base_dir)
        if "tools" in spec and isinstance(spec["tools"], dict):  # {agent: [tool, ...] or "module:factory"}
            spec["tools"] = {name: resolve(ts, self.base_dir) if isinstance(ts, str) else
                             [resolve(t, self.base_dir) for t in ts] for name, ts in spec["tools"].items()}
        elif "tools" in spec and isinstance(spec["tools"], list):
            spec["tools"] = [resolve(t, self.base_dir) for t in spec["tools"]]
        if cls.__name__ == "TaskSequence":
            from .envs import Task

            spec["tasks"] = [Task(**{**t, "score": resolve(t["score"], self.base_dir) if isinstance(t.get("score"), str)
                              else t.get("score")}) for t in spec["tasks"]]
            if isinstance(spec.get("tools"), str):
                spec["tools"] = resolve(spec["tools"], self.base_dir)
        return cls(**spec)

    def interventions(self) -> list:
        out = []
        for p in self.cfg.get("probes") or []:
            p = dict(p)
            kind = p.pop("type", "probe")
            when = p.pop("at", None)
            k = p.pop("every", None)
            if kind == "survey":
                opts = p.pop("options")
                fn = survey(p.pop("question"), LIKERT.get(opts, opts) if isinstance(opts, str) else opts,
                            agents=p.pop("agents", None), name=p.pop("name", "survey"))
            elif kind == "probe":
                schema = to_schema(p.pop("schema")) if "schema" in p else None
                fn = probe(p.pop("question"), schema=schema, agents=p.pop("agents", None), name=p.pop("name", "probe"))
            elif kind == "whisper":
                fn = whisper(p.pop("agent"), p.pop("text"))
            else:
                raise ConfigError(f"unknown probe type {kind!r} (probe, survey, whisper)")
            if k is not None:
                out.append(every(int(k), fn, start=int(p.pop("start", 0)), at_end=bool(p.pop("at_end", False))))
            else:
                steps = when if isinstance(when, list) else [0 if when is None else when]
                out.append(at([END if s == "end" else int(s) for s in steps], fn))
        return out

    def evaluators(self, seed: int) -> dict:
        jcfg = self.cfg.get("judge")
        if not jcfg:
            return {}
        jcfg = dict(jcfg)
        judge = Judge(self.model(jcfg.pop("model", "judge"), seed), rubric=jcfg.pop("rubric"),
                      schema=to_schema(jcfg.pop("schema")) if "schema" in jcfg else None)
        return {jcfg.pop("name", "judge"): judge.evaluator()}

    def build(self, cond: dict[str, Any], seed: int) -> Simulation:
        saved = self.cfg
        self.cfg = substitute({k: v for k, v in saved.items() if k != "grid"}, cond)
        try:
            sim_cfg = dict(self.cfg.get("simulation") or {})
            sched = sim_cfg.pop("scheduler", "round_robin")
            if isinstance(sched, str):
                sched = getattr(schedule_mod, sched, None) or resolve(sched, self.base_dir)
            elif isinstance(sched, dict):
                sched = dict(sched)
                sched = getattr(schedule_mod, sched.pop("type"))(**sched)
            if "clock" in sim_cfg:
                from .clock import Clock

                sim_cfg["clock"] = Clock(**sim_cfg["clock"])
            return Simulation(self.environment(seed), self.agents(seed), scheduler=sched, seed=seed,
                              interventions=self.interventions(), evaluators=self.evaluators(seed), **sim_cfg)
        finally:
            self.cfg = saved

    def experiment(self, out: str | None = None, seeds: int | None = None) -> Experiment:
        n = seeds if seeds is not None else self.cfg.get("seeds", 1)
        seed_list = list(range(n)) if isinstance(n, int) else list(n)
        out_dir = out or self.cfg.get("out") or f"runs/{self.cfg.get('name', 'experiment')}"
        return Experiment(self.build, self.conditions(), seeds=seed_list, out=out_dir)


def _dry_script(messages, tools):
    """A fake model for --dry-run: valid-looking answers for any schema."""
    content = messages[-1]["content"] or ""
    if "label only" in content:
        m = re.search(r"\(([^)]*)\)\.\s*$", content)
        return (m.group(1).split(",")[0].strip() if m else "A")
    if "Respond with only a JSON object" in content:
        fields = re.findall(r'^\s+"([^"]+)":\s*(.+?)(?:\s+//.*)?$', content.split("Respond with only a JSON object")[-1],
                            flags=re.M)
        out = {}
        for name, spec in fields:
            if spec.startswith("one of "):
                out[name] = json.loads(spec[len("one of "):].split(", ")[0])
            elif spec.startswith(("int", "float")) and " from " in spec:
                out[name] = int(float(spec.split(" from ")[1].split(" to ")[0]))
            elif spec.startswith(("integer", "number")):
                out[name] = 1
            elif spec.startswith("true"):
                out[name] = True
            else:
                out[name] = "(dry run)"
        return json.dumps(out)
    return "(dry run reply)"


def run_config(path: str | Path, workers: int = 4, dry_run: bool = False, out: str | None = None,
               seeds: int | None = None, progress: bool = True):
    cfg = load_config(path)
    exp = ConfigExperiment(cfg, dry_run=dry_run).experiment(out=out, seeds=seeds)
    Path(exp.out).mkdir(parents=True, exist_ok=True)
    (Path(exp.out) / "config.yaml").write_text(Path(path).read_text())
    rows = exp.run(workers=workers, progress=progress)
    return exp, rows
