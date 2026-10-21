from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agent import Agent, Turn
from ..tools import tool
from .base import Environment


@dataclass
class Post:
    id: int
    author: str
    text: str
    t: int
    likes: set[str] = field(default_factory=set)
    reply_to: int | None = None
    repost_of: int | None = None


class Feed(Environment):
    """A minimal social-media platform: post, reply, repost, like, follow.

    Each turn an agent sees a timeline (recent posts from accounts it follows,
    or from everyone if it follows nobody) plus an optional news headline, and
    acts through tools.

    Args:
        follows: initial follow graph {agent: [agents it follows]}.
        news: headlines; one is shown per turn (cycled) if given.
        timeline_size: posts shown per turn.
        ranking: "recent" or "popular" (likes, then recency). Swap in your own
            by overriding `rank`.
    """

    def __init__(
        self,
        follows: dict[str, list[str]] | None = None,
        news: list[str] | None = None,
        timeline_size: int = 10,
        ranking: str = "recent",
    ):
        self.initial_follows = follows or {}
        self.news = news or []
        self.timeline_size = timeline_size
        self.ranking = ranking

    def reset(self, agents, rng):
        super().reset(agents, rng)
        self.posts: list[Post] = []
        self.follows = {a.name: set(self.initial_follows.get(a.name, [])) for a in agents}
        self._news_i = 0

    def rank(self, posts: list[Post], agent: Agent) -> list[Post]:
        if self.ranking == "popular":
            return sorted(posts, key=lambda p: (len(p.likes), p.t, p.id), reverse=True)
        return sorted(posts, key=lambda p: (p.t, p.id), reverse=True)

    def timeline(self, agent: Agent) -> list[Post]:
        following = self.follows[agent.name]
        pool = [p for p in self.posts if p.author != agent.name and (not following or p.author in following)]
        return self.rank(pool, agent)[: self.timeline_size]

    def _fmt(self, p: Post) -> str:
        tag = f" (reply to #{p.reply_to})" if p.reply_to is not None else ""
        tag += f" (repost of #{p.repost_of})" if p.repost_of is not None else ""
        return f"#{p.id} @{p.author}{tag}: {p.text}  [{len(p.likes)} likes]"

    def observe(self, agent: Agent, t: int) -> str:
        parts = []
        if self.news:
            parts.append(f"News: {self.news[self._news_i % len(self.news)]}")
            self._news_i += 1
        tl = self.timeline(agent)
        parts.append("Your timeline:\n" + ("\n".join(self._fmt(p) for p in tl) if tl else "(empty)"))
        parts.append(
            "You are browsing the platform. Use the tools to post, reply, repost, like, or follow, "
            "as you naturally would. You may also do nothing. Then briefly say what you did."
        )
        return "\n\n".join(parts)

    def tools_for(self, agent: Agent, t: int):
        me = agent.name

        def new_post(text: str, **kw) -> str:
            p = Post(len(self.posts), me, text, t, **kw)
            self.posts.append(p)
            return f"posted #{p.id}"

        def get(post_id: int) -> Post:
            if not 0 <= post_id < len(self.posts):
                raise ValueError(f"no post #{post_id}")
            return self.posts[post_id]

        @tool
        def post(text: str) -> str:
            """Write a new post."""
            return new_post(text)

        @tool
        def reply(post_id: int, text: str) -> str:
            """Reply to a post by its id."""
            get(post_id)
            return new_post(text, reply_to=post_id)

        @tool
        def repost(post_id: int) -> str:
            """Share someone else's post with your followers."""
            return new_post(get(post_id).text, repost_of=post_id)

        @tool
        def like(post_id: int) -> str:
            """Like a post."""
            get(post_id).likes.add(me)
            return "liked"

        @tool
        def follow(user: str) -> str:
            """Follow another user by name."""
            if user not in self.agents or user == me:
                return f"Error: no user {user!r}"
            self.follows[me].add(user)
            return f"now following {user}"

        @tool
        def unfollow(user: str) -> str:
            """Stop following a user."""
            self.follows[me].discard(user)
            return f"unfollowed {user}"

        return [post, reply, repost, like, follow, unfollow]

    def metrics(self) -> dict[str, Any]:
        return {
            "n_posts": len(self.posts),
            "posts": [
                {"id": p.id, "author": p.author, "t": p.t, "text": p.text, "likes": sorted(p.likes),
                 "reply_to": p.reply_to, "repost_of": p.repost_of}
                for p in self.posts
            ],
            "follows": {a: sorted(f) for a, f in self.follows.items()},
        }
