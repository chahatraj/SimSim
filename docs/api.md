# API reference

Generated from the docstrings. Everything listed here is importable from `simsim`
unless the heading names a module.

## Models

::: simsim.llm
    options:
      members: [make_llm, LLM, OpenAIChat, LiteLLMChat, ScriptedLLM, LLMResponse, ToolCall]

::: simsim.cache.CachedLLM

## Agents

::: simsim.agent
    options:
      members: [Agent, Persona, Turn, ScriptedAgent, HumanAgent]

::: simsim.tools
    options:
      members: [tool, Tool, emulated_tool]

::: simsim.structured
    options:
      members: [structured_chat, format_instructions, extract_json, StructuredOutputError]

## Memory and components

::: simsim.memory
    options:
      members: [Memory, MemoryEntry, NoMemory, FullHistory, WindowMemory, SummaryMemory, MarkdownMemory, ScoredMemory, ReflectiveMemory, llm_importance]

::: simsim.embed
    options:
      members: [Embedder, OpenAIEmbedder, HashEmbedder]

::: simsim.components
    options:
      members: [Component, Goal, Plan, Impressions, InnerState]

## Environments

::: simsim.envs.base.Environment

::: simsim.envs.conversation
    options:
      members: [Conversation, user_simulator]

::: simsim.envs.tasks
    options:
      members: [Task, TaskSequence]

::: simsim.envs.feed
    options:
      members: [Feed, Post]

## Running

::: simsim.sim
    options:
      members: [Simulation, Result, at, every, whisper, add_agent, probe, survey, Intervention]

::: simsim.schedule
    options:
      members: [round_robin, everyone, everyone_shuffled, random_one, bernoulli, zipf, by_hour, NextSpeaker]

::: simsim.clock.Clock

::: simsim.experiment
    options:
      members: [Experiment, grid]

::: simsim.judge
    options:
      members: [Judge, rescore]

::: simsim.personas
    options:
      members: [sample_personas, personas_from_rows, describe_persona]

## Analysis

::: simsim.analysis
    options:
      members: [load_runs, load_turns, load_probes, summarize, pass_hat_k, cost_report, transcript]

::: simsim.viewer
    options:
      members: [write_viewer]

::: simsim.record
    options:
      members: [Recorder, read_jsonl]
