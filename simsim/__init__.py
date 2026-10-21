"""SimSim: build, run, and log LLM agent simulations for research."""

from .agent import Agent, HumanAgent, Persona, ScriptedAgent, Turn, parse_react
from .cache import CachedLLM
from .clock import Clock
from .components import Component, Goal, Impressions, InnerState, Plan
from .embed import Embedder, HashEmbedder, OpenAIEmbedder, cosine
from .envs import Conversation, Environment, Feed, Task, TaskSequence, user_simulator
from .experiment import Experiment, grid
from .judge import Judge, rescore
from .llm import LLM, LiteLLMChat, LLMResponse, OpenAIChat, ScriptedLLM, ToolCall, make_llm
from .memory import (
    FullHistory,
    MarkdownMemory,
    Memory,
    MemoryEntry,
    NoMemory,
    ReflectiveMemory,
    ScoredMemory,
    SummaryMemory,
    WindowMemory,
    llm_importance,
)
from .personas import describe_persona, personas_from_rows, sample_personas
from .record import Recorder, read_jsonl
from .schedule import NextSpeaker, bernoulli, by_hour, everyone, everyone_shuffled, random_one, round_robin, zipf
from .sim import END, Intervention, Result, Simulation, add_agent, at, every, probe, survey, whisper
from .structured import StructuredOutputError, extract_json, format_instructions, structured_chat
from .tools import Tool, emulated_tool, tool

__version__ = "0.7.0"

# The public, stable API (see docs: "Stability"). Anything not listed here may change.
__all__ = [
    "add_agent",
    "Agent",
    "at",
    "bernoulli",
    "by_hour",
    "CachedLLM",
    "Clock",
    "Component",
    "Conversation",
    "cosine",
    "describe_persona",
    "Embedder",
    "emulated_tool",
    "END",
    "Environment",
    "every",
    "everyone",
    "everyone_shuffled",
    "Experiment",
    "extract_json",
    "Feed",
    "format_instructions",
    "FullHistory",
    "Goal",
    "grid",
    "HashEmbedder",
    "HumanAgent",
    "Impressions",
    "InnerState",
    "Intervention",
    "Judge",
    "LiteLLMChat",
    "LLM",
    "llm_importance",
    "LLMResponse",
    "make_llm",
    "MarkdownMemory",
    "Memory",
    "MemoryEntry",
    "NextSpeaker",
    "NoMemory",
    "OpenAIChat",
    "OpenAIEmbedder",
    "parse_react",
    "Persona",
    "personas_from_rows",
    "Plan",
    "probe",
    "random_one",
    "read_jsonl",
    "Recorder",
    "ReflectiveMemory",
    "rescore",
    "Result",
    "round_robin",
    "sample_personas",
    "ScoredMemory",
    "ScriptedAgent",
    "ScriptedLLM",
    "Simulation",
    "structured_chat",
    "StructuredOutputError",
    "SummaryMemory",
    "survey",
    "Task",
    "TaskSequence",
    "tool",
    "Tool",
    "ToolCall",
    "Turn",
    "user_simulator",
    "whisper",
    "WindowMemory",
    "zipf",
]
