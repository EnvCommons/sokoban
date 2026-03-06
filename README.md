# Sokoban

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/Sokoban)

## Description

**Sokoban** is an environment for evaluating agents on spatial planning and sequential puzzle solving. This environment wraps the Sokoban implementation from [TextArena](https://github.com/LeonGuertler/TextArena), a framework for text-based game environments.

## Capabilities

- Long-term spatial planning and reasoning
- Understanding irreversible actions and dead states
- Multi-step goal decomposition
- Strategic path planning with obstacles

## Compute Requirements

Sokoban does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://github.com/LeonGuertler/TextArena/blob/main/LICENSE).

## Tasks

There are two splits: train (300 tasks) and test (300 tasks). Each split contains 50 tasks across each of 6 variants:

- **Sokoban-v0**
- **Sokoban-v0-medium**
- **Sokoban-v0-medium-raw**
- **Sokoban-v0-medium-train**
- **Sokoban-v0-raw**
- **Sokoban-v0-train**

Each task is seeded for reproducibility.

## Reward Structure

This is a sparse reward environment. Rewards are mapped from TextArena's native range of {-1, 0, 1} to {0.0, 0.5, 1.0} via `(raw + 1) / 2`.

We do not use LLM graders for this environment; reward is determined programmatically.

## Data

Game state is generated procedurally by the TextArena engine using seeded randomness. No external data files are required.

## Tools

Agents are given a single tool:

- `move(direction)`: Move in the given direction (up, down, left, right). Push boxes by moving into them.

## Time Horizon

Sokoban is a multi-turn environment.

## Environment Difficulty

This environment ranges from moderate to challenging depending on the variant. The medium variant presents increased difficulty with more complex level layouts and additional boxes to manage.

## Other Environment Requirements

There are no further environment requirements; Sokoban works out of the box without any secrets or API keys.

## Safety

Agents in Sokoban interact only with a puzzle game and have no access to external systems, the internet, or sensitive data. The environment does not present safety risks.

## Citations

```bibtex
@software{textarena2024,
  author    = {Guertler, Leon and Banting, Wilfried and Pignatelli, Eduardo},
  title     = {TextArena},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/LeonGuertler/TextArena}
}
```
