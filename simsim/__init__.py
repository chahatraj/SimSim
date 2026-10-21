"""SimSim: build, run, and log LLM agent simulations for research."""

from .agent import Agent, HumanAgent, Persona, ScriptedAgent, Turn, parse_react
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
    ScoredMemory,
    SummaryMemory,
    WindowMemory,
)
from .personas import describe_persona, personas_from_rows, sample_personas
from .record import Recorder, read_jsonl
from .schedule import everyone, everyone_shuffled, random_one, round_robin, zipf
from .sim import END, Intervention, Result, Simulation, add_agent, at, every, probe, survey, whisper
from .structured import StructuredOutputError, extract_json, format_instructions, structured_chat
from .tools import Tool, tool

__version__ = "0.2.0"
