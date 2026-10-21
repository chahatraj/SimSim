# Environments

An environment is the world: it decides what each agent observes, which tools it
gets, how actions change the world, and when the run ends.

```python
class Environment:
    def reset(self, agents, rng): ...        # start of a run
    def observe(self, agent, t) -> str | None: ...   # None skips the agent's turn
    def tools_for(self, agent, t) -> list[Tool]: ...
    def apply(self, turn, t): ...            # update the world
    def metrics(self) -> dict: ...           # saved with the run
    def add_agent(self, agent): ...          # someone joins mid-run
    # call self.stop("reason") to end the run; the reason is recorded
```

## Conversation

Agents talking in a shared room: negotiation, debate, persuasion, interviews,
co-writing, or a user simulator testing an assistant.

```python
from simsim import Conversation
env = Conversation(
    "A buyer and a seller negotiate the price of a bike listed at $200.",
    private={"Buyer": "Your budget is $170."},   # shown only to that agent
    show="new",            # each agent sees messages since its last turn ("all": full transcript)
    end_token="[END]",     # an agent ends it by putting this on the last line
    min_turns=2,           # earlier end tokens are ignored (models tend to "end" at once)
    enders=None,           # who may end it (default anyone)
    max_turns=12,
    tools={"Seller": [check_inventory]},     # per agent; or a factory world -> tools
    judge=None,            # (transcript) -> bool, e.g. Judge.stopper("Did they agree?")
    score=None,            # (transcript, world) -> dict of metrics
)
```

Metrics: `n_messages`, `ended_by`, `transcript`, plus whatever `score` returns.

### User simulators

```python
from simsim import user_simulator
user = user_simulator("user", llm, goal="Get a refund for order A100; you have the receipt.")
env = Conversation("Customer support chat.", end_token="###STOP###", enders=["user"],
                   tools={"assistant": make_tools},               # make_tools(world) -> [tools]
                   score=lambda transcript, world: {"refunded": float("A100" in world["refunds"])})
```

The simulated user gets a hidden goal and stops the chat when the goal is met or
clearly cannot be. Tool factories receive the run's own `world` dict, so parallel
runs never share state, and `score` reads the same world at the end (as in
tau-bench). See the [support agent recipe](../recipes/support-agent.md).

## TaskSequence

Each agent works through tasks in order, and its memory carries across tasks:
the standard setup for memory experiments (early tasks plant information, later
tasks test it).

```python
from simsim import Task, TaskSequence
tasks = [
    Task("For future trips: I always want an aisle seat.", id="plant"),
    Task("Book me a flight to Lisbon.", id="probe",
         score=lambda turn, world: {"success": float(world.get("seat") == "aisle")}),
]
env = TaskSequence(tasks, tools=make_tools, world={})     # make_tools(world) -> [tools]
```

Each agent gets its own deep copy of `world`. Metrics: `mean_<score>` over tasks
and a per-task table.

## Feed

A minimal social network: agents see a timeline (posts from accounts they follow,
or everyone if they follow nobody) and an optional news headline, and act through
`post`, `reply`, `repost`, `like`, `follow`, `unfollow`.

```python
from simsim import Feed
env = Feed(news=["Gas prices rise again."], timeline_size=8, ranking="popular",
           follows={"ana": ["bo"]})
env.inject_post("newsbot", "BREAKING: ...")     # from an intervention, e.g. misinformation
```

`ranking` is `"recent"` or `"popular"`; override `Feed.rank` for your own
recommender. Metrics: `n_posts`, `posts` (with likes, replies, reposts), `follows`.

## Writing an environment

Most studies need their own environment; this is the intended way to extend
SimSim. See the [custom environment recipe](../recipes/custom-environment.md) for
a complete public-goods game in about 40 lines.
