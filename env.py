import asyncio
import multiprocessing as mp
import os
import re
from typing import List

import textarena as ta
from pydantic import BaseModel, field_validator
from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool


# --- Level generation is CPU-bound and can hang; keep it OFF the event loop ----
#
# The env-server runs a single-worker uvicorn event loop. TextArena's Sokoban
# `reset()` procedurally generates a level via a retry loop over an exhaustive
# reverse-play search (utils.generate_room -> depth_first_search, up to 300k
# states per attempt, up to 1000 attempts, wrapped in another 50-attempt retry).
# For the `-medium` variants (8x8, 5 boxes) this measures at 1-20s and for some
# seeds never completes. Two consequences shaped this code:
#
#  1. `get_prompt` must NOT do this work inline. It is `async def`, and the SDK
#     awaits `async def` callbacks directly on the event loop (see
#     openreward/environments/utils.py::run_user_callable) -- there is no
#     to_thread guard rail for coroutines. A synchronous CPU-bound body here
#     freezes every other session on the pod ("async in name only").
#  2. `asyncio.to_thread` alone is not enough: the generator holds the GIL
#     (degrading the loop) and, worse, uses module-level globals in
#     textarena.envs.Sokoban.utils (explored_states / best_room / ...), so
#     concurrent generation in threads would corrupt shared state. And a thread
#     cannot be killed, so seeds that never finish would leak forever with no
#     enforceable timeout.
#
# So generation runs in a short-lived child PROCESS (isolated globals, killable)
# with a hard timeout. On timeout the process is terminated and THIS task fails
# (the SDK's ErrorHandlingMiddleware turns the exception into a 500 for this one
# session); the pod keeps serving everyone else. The child returns only the
# picklable board arrays (the wrapped TextArena env itself is NOT picklable --
# unpickling recurses infinitely through the wrapper chain), which are then
# injected into the parent's env so its `reset()` skips generation entirely.

# Wall-clock budget for a single level generation. Legitimate `-medium` seeds
# were measured up to ~20s; the default leaves margin while still killing the
# seeds that never terminate. Override via env var for tuning.
RESET_TIMEOUT_S = float(os.environ.get("SOKOBAN_RESET_TIMEOUT_S", "30"))

# fork inherits the already-imported textarena (child start ~0.1-0.3s) vs
# ~0.8s for spawn re-importing it. The child only touches textarena/numpy and
# then exits, so it does not run any asyncio/uvicorn machinery.
_MP_CTX = mp.get_context("fork")


def _unwrap_to_base(env):
    """Descend the TextArena wrapper chain to the underlying SokobanEnv."""
    while hasattr(env, "env"):
        env = env.env
    return env


def _generate_board_worker(env_id: str, seed: int, q) -> None:
    """Child-process entry point: build+reset a Sokoban env and return only the
    picklable board arrays produced by the (expensive) generator."""
    try:
        env = ta.make(env_id=env_id)
        env.reset(num_players=1, seed=seed)
        base = _unwrap_to_base(env)
        q.put(("ok", (base.room_fixed, base.room_state, base.box_mapping)))
    except BaseException as exc:  # noqa: BLE001 - surface any failure to the parent
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def _generate_board_blocking(env_id: str, seed: int, timeout: float):
    """Run level generation in a killable child process with a hard timeout.

    Returns (room_fixed, room_state, box_mapping). Raises TimeoutError if the
    generator does not finish in `timeout` seconds (the child is killed), or
    RuntimeError if the child fails to produce a board. Blocks the calling
    thread on os-level joins (GIL released), so it is safe to run under
    asyncio.to_thread without degrading the event loop.
    """
    q = _MP_CTX.Queue()
    proc = _MP_CTX.Process(
        target=_generate_board_worker, args=(env_id, seed, q), daemon=True
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise TimeoutError(
            f"Sokoban level generation for env_id={env_id!r} seed={seed} "
            f"exceeded {timeout:.0f}s; killed."
        )
    try:
        status, payload = q.get_nowait()
    except Exception as exc:  # queue.Empty and friends
        raise RuntimeError(
            f"Sokoban level generation for env_id={env_id!r} seed={seed} "
            f"produced no board (worker exit code {proc.exitcode})."
        ) from exc
    if status == "error":
        raise RuntimeError(
            f"Sokoban level generation for env_id={env_id!r} seed={seed} "
            f"failed: {payload}"
        )
    return payload


class TaskSpec(BaseModel):
    id: str
    env_id: str
    seed: int
    variant: str = ""


class MoveParams(BaseModel, extra="forbid"):
    direction: str

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v):
        v = v.strip().lower()
        if v not in ("up", "down", "left", "right"):
            raise ValueError("direction must be one of: up, down, left, right")
        return v


class SokobanEnvironment(Environment):
    GAME_NAME = "Sokoban"
    VARIANTS = [
        "Sokoban-v0",
        "Sokoban-v0-medium",
        "Sokoban-v0-medium-raw",
        "Sokoban-v0-medium-train",
        "Sokoban-v0-raw",
        "Sokoban-v0-train"
    ]
    NUM_TASKS_PER_VARIANT = 50

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.ta_env = ta.make(env_id=self.config.env_id)
        self.game_done = False
        self.turn_count = 0

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        tasks = []
        for variant_id in cls.VARIANTS:
            for seed_idx in range(cls.NUM_TASKS_PER_VARIANT):
                seed = seed_idx if split == "train" else seed_idx + 10000
                tasks.append({
                    "id": f"{variant_id}_seed{seed}",
                    "env_id": variant_id,
                    "seed": seed,
                    "variant": variant_id
                })
        return tasks

    def _format_observation(self, observation) -> str:
        if isinstance(observation, str):
            match = None
            for m in re.finditer(r'^\[(?!GAME\])[^\]]+\].*$', observation, re.MULTILINE):
                match = m
            if match:
                return observation[match.end():].lstrip('\n')
            return observation
        if isinstance(observation, list):
            if not observation:
                return ""
            last = observation[-1]
            if isinstance(last, tuple) and len(last) >= 2:
                return str(last[1])
            return str(last)
        return str(observation)

    def _map_reward(self, raw_reward: float) -> float:
        return max(0.0, min(1.0, (raw_reward + 1.0) / 2.0))

    def _reset_with_board(self, board) -> None:
        """Run the env's reset() but short-circuit level generation to a board
        already computed off-loop.

        TextArena's SokobanEnv.reset() looks up `generate_room` in its own
        module namespace, so we bind it there to return the precomputed board
        for the duration of this (fast, generation-free) reset. This runs on the
        event loop, but with generation removed it is sub-millisecond. It is
        concurrency-safe on the single-threaded loop: there is no `await`
        between patch and restore, so no other coroutine can observe the patch.
        """
        import textarena.envs.Sokoban.env as sokoban_module

        called = {"hit": False}

        def _prebuilt_generate_room(*args, **kwargs):
            called["hit"] = True
            return board

        original = sokoban_module.generate_room
        sokoban_module.generate_room = _prebuilt_generate_room
        try:
            self.ta_env.reset(num_players=1, seed=self.config.seed)
        finally:
            sokoban_module.generate_room = original

        if not called["hit"]:
            # Upstream changed how reset() obtains its room; rather than silently
            # fall back to running the slow generator on the event loop, fail
            # this task loudly so the regression is caught.
            raise RuntimeError(
                "Sokoban level injection did not take effect; TextArena's "
                "reset() no longer calls generate_room as expected. Refusing to "
                "run level generation on the event loop."
            )

    async def get_prompt(self) -> List[TextBlock]:
        # Generate the level in a killable child process with a hard timeout so
        # a slow/never-terminating seed cannot freeze the event loop (see the
        # module-level comment). The child returns only the board arrays; we
        # then run the parent's reset() with generation short-circuited to those
        # arrays, so the on-loop work is microseconds, not seconds.
        board = await asyncio.to_thread(
            _generate_board_blocking,
            self.config.env_id,
            self.config.seed,
            RESET_TIMEOUT_S,
        )
        self._reset_with_board(board)
        _, observation = self.ta_env.get_observation()
        obs_text = self._format_observation(observation)
        prompt = (
            f"You are playing Sokoban.\n\n"
            f"{obs_text}\n\n"
            f"Use the move tool with direction (up, down, left, right) to push boxes onto target positions.\n"
            f"Push all boxes to the target locations to win."
        )
        return [TextBlock(text=prompt)]

    @tool
    async def move(self, params: MoveParams) -> ToolOutput:
        """Move in the given direction (up, down, left, right). Push boxes by moving into them."""
        if self.game_done:
            return ToolOutput(
                blocks=[TextBlock(text="Game is already over.")],
                metadata={"error": "game_finished"},
                reward=0.0,
                finished=True
            )

        action = f"[{params.direction}]"
        done, info = self.ta_env.step(action=action)
        self.turn_count += 1

        if done:
            self.game_done = True
            rewards, game_info = self.ta_env.close()
            raw = rewards.get(0, 0.0) if isinstance(rewards, dict) else float(rewards)
            reward = self._map_reward(raw)
            reason = ""
            if isinstance(game_info, dict) and 0 in game_info:
                reason = game_info[0].get("reason", "")
            summary = f"Game Over! Reward: {reward:.2f}"
            if reason:
                summary += f"\n{reason}"
            return ToolOutput(
                blocks=[TextBlock(text=summary)],
                metadata={"turn": self.turn_count, "reward": reward},
                reward=reward,
                finished=True
            )

        _, observation = self.ta_env.get_observation()
        obs_text = self._format_observation(observation)
        return ToolOutput(
            blocks=[TextBlock(text=obs_text)],
            metadata={"turn": self.turn_count},
            reward=0.0,
            finished=False
        )
