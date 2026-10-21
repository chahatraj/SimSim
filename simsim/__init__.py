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

__version__ = "0.6.0"
