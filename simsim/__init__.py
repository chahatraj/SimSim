"""SimSim: build, run, and log LLM agent simulations for research."""

from .agent import Agent, Persona, Turn
from .envs import Conversation, Environment, Feed, Task, TaskSequence
from .experiment import Experiment, grid
from .llm import LLM, LiteLLMChat, LLMResponse, OpenAIChat, ScriptedLLM, ToolCall, make_llm
from .memory import (
    FullHistory,
    MarkdownMemory,
    Memory,
    MemoryEntry,
    NoMemory,
    ScoredMemory,
    SummaryMemory,
    WindowMemory,
)
from .record import Recorder, read_jsonl
from .schedule import everyone, everyone_shuffled, random_one, round_robin, zipf
from .sim import Result, Simulation
from .tools import Tool, tool

__version__ = "0.1.0"
