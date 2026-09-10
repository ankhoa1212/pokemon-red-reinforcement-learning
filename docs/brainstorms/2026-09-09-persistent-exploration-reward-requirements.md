---
date: 2026-09-09
topic: persistent-exploration-reward
---

# Persistent, Generalized Exploration Reward

## Summary

Replace the exploration reward's frame-comparison mechanism with a persistent, count-based novelty signal: each screen frame is hashed into a state "cell," and reward comes from how rarely that cell has been seen across the entire training run, not just the current episode. The mechanism reads only rendered screen pixels, so it works the same way for any game with visual output, not only Pokemon Red.

## Problem Frame

The current reward in `pokemon_red_env.py`'s `calculate_fitness` compares each new frame against every frame stored in `self.memory` via pairwise MSE, and `self.memory` is rebuilt from empty on every `reset()`. Two consequences follow: the agent gets full novelty reward for revisiting a state it has already seen in a prior episode, and the comparison cost grows linearly with the number of distinct states seen, which does not scale toward the project's actual goal — exploring the game's full, finite state space until it is complete.

The project's stated design thesis is that games have a finite, enumerable set of states, and full exploration of that set corresponds to completing the game. A reward mechanism that forgets what it has already found on every episode boundary cannot embody that thesis, regardless of how well it behaves within a single episode.

## Key Decisions

- **State identity comes from screen pixels only, never emulator memory.** This keeps the mechanism usable for any game that produces visual output, matching the project's generalization goal, at the cost of the extra precision a direct memory-state hash could give. Chosen explicitly over reading PyBoy's internal RAM.
- **Count-based hashing over a persistent visited-cell table, not Random Network Distillation (RND) or a persistent nearest-neighbor index.** It is the cheapest mechanism to run given the project's CPU-only, multi-process training setup, and its "count visited cells, reward rarity" shape is the same core mechanism published as state of the art for this exact exploration philosophy (see Sources). RND is the documented fallback if hash coarseness or animation noise turns out to be a real ceiling rather than a tuning problem.
- **This reward mechanism replaces the current pixel-comparison logic entirely; it does not run alongside it.** `self.memory`, `calculate_fitness`'s pairwise-MSE loop, and its distance thresholds are retired, not kept as a fallback path.
- **The one-PyBoy-per-CPU training architecture (`SubprocVecEnv` in `main.py`) stays as is.** It matches Stable-Baselines3's own guidance for computationally heavy, non-IO-bound environments, and PPO's on-policy rollout collection benefits directly from wall-clock parallelism. This confirms the cross-process count-sharing requirement below is real and necessary, not something a simpler single-process design could avoid.

## Requirements

- R1. Reward for a step is derived from a persistent count of how many times the agent's current screen state has been observed, not from comparing the current frame against a list of stored past frames.
- R2. The visit-count state survives across episode boundaries (`env.reset()`) for the life of a training run; it does not reset per episode.
- R3. The visit-count state is shared across all parallel training workers (each running its own PyBoy instance), so a state visited by one worker counts as visited for all workers.
- R4. State identity is derived only from the rendered screen; no emulator memory or RAM read is used to compute or influence the reward.
- R5. The existing per-episode, pairwise pixel-comparison reward path (`self.memory`, `calculate_fitness`'s MSE-based scan, and its distance thresholds) is removed as part of this change, not preserved alongside the new mechanism.

## Scope Boundaries

**Deferred for later:**
- Random Network Distillation (RND) as a learned, graded novelty signal — revisit only if count-based hashing's collision or noise behavior proves to be a real ceiling.
- A persistent approximate-nearest-neighbor index over frame embeddings.
- Go-Explore's "return to a promising state, then explore from it" step. This project already has save-state infrastructure (`start_states/`, `PokemonRedEnv.save_state`) that could support it, but it is out of scope until persistent counting itself is validated.

**Outside this feature:**
- Any reward component derived from emulator memory or game-specific state (badges, party levels, map IDs, event flags) — rejected outright, not just deferred, to keep the mechanism generalized across games.
- The existing map-stitching code (`stitching/stitch.py`) — unrelated to this change and untouched by it.

## Dependencies / Assumptions

- `main.py` currently sets `NUM_CPU = os.cpu_count()` with no core held back for the main process. Once the main process also has to coordinate a shared count table, this may oversubscribe cores; addressing it is a tuning concern for planning, not a blocker for this doc.
- The specific hash/state-discretization function (downsample resolution, quantization, tolerance to animation and menu noise) is left to planning. Published results (Tang et al., 2017) identify hash granularity as the primary lever for whether count-based exploration works well or degenerates.

## Outstanding Questions

**Deferred to Planning:**
- What hash or state-discretization scheme to use, and how to make it robust to visual noise (tile animation, text scroll, menu overlays) without collapsing genuinely distinct states together.
- The exact reward formula (e.g., `1/sqrt(count)`) and whether any decay or normalization is applied.
- The concrete mechanism for sharing the count table across `SubprocVecEnv` worker processes (e.g., a `multiprocessing.Manager` dict, shared memory, or a lightweight coordinator process).

## Sources / Research

- [Go-Explore: a New Approach for Hard-Exploration Problems](https://arxiv.org/abs/1901.10995) (Ecoffet et al., 2019) — the direct precedent for this doc's mechanism: an archive of visited state "cells," each defined by a downsampled visual observation, used to drive exploration toward uncovered states. State of the art on Montezuma's Revenge and Pitfall.
- [#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717) (Tang et al., 2017) — establishes that simple hash-based counting is competitive with more complex methods, and that hash granularity is the main design lever for quality.
- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) (Burda et al., 2018) — the fallback approach if hashing proves insufficient; first method to exceed average human performance on Montezuma's Revenge using only pixel input.
- [Peter Whidden's Pokemon Red RL project](https://qlawk.medium.com/how-one-youtuber-trained-ai-to-play-video-games-with-reinforcement-learning-f37ba07133a4) — this repository's stated inspiration. Documents the exact failure mode raw pixel-novelty reward can produce (the agent fixating on animated water tiles), which motivates R4's need for a robust, not naive, notion of screen-state identity.
- [Stable-Baselines3: Vectorized Environments](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html) — guidance behind the Key Decision confirming `SubprocVecEnv` as the correct architecture for this project's CPU-bound PyBoy environments.
- `pokemon_red_env.py` (`calculate_fitness`, `self.memory`) — the current implementation being replaced.
- `main.py` (`NUM_CPU`, `SubprocVecEnv`/`DummyVecEnv`) — the parallel training setup this reward mechanism must integrate with.
