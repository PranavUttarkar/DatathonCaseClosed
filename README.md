# Case Closed Agent Template

### Explanation of Files

This template provides a few key files to get you started. Here's what each one does:

#### `agent.py`
**This is the most important file. This is your starter code, where you will write your agent's logic.**

*   DO NOT RENAME THIS FILE! Our pipeline will only recognize your agent as `agent.py`.
*   It contains a fully functional, Flask-based web server that is already compatible with the Judge Engine's API.
*   It has all the required endpoints (`/`, `/send-state`, `/send-move`, `/end`). You do not need to change the structure of these.
*   Look for the `send_move` function. Inside, you will find a section marked with comments: `# --- YOUR CODE GOES HERE ---`. This is where you should add your code to decide which move to make based on the current game state.
*   Your agent can return moves in the format `"DIRECTION"` (e.g., `"UP"`, `"DOWN"`, `"LEFT"`, `"RIGHT"`) or `"DIRECTION:BOOST"` (e.g., `"UP:BOOST"`) to use a speed boost.

## Our Agent Strategy (Hamiltonian + Flood-Fill Heuristic)

This repository’s `agent.py` implements a modular, high-coverage strategy that combines a Hamiltonian serpentine cycle with a safety-first heuristic fallback. The goal is to systematically traverse the board and fill as many cells as possible while minimizing crash risk and head-on collisions.

### Key Ideas
- Hamiltonian serpentine cycle: A precomputed path that visits every cell exactly once in a snaking pattern across rows. If the next cell on this cycle is available and not the opposite of our current direction, we follow it. This tends to maximize coverage and keep our trail orderly.
- Flood-fill fallback: When the cycle is blocked or following it would be risky, we evaluate the remaining safe directions using a scoring function based on reachable empty area after the move, distance from the opponent’s head, a bonus for continuing straight (to avoid jitter), and a cycle-continuity bonus.
- Selective boost usage: We only use a BOOST if it safely and clearly increases our score beyond a threshold; otherwise, we conserve boosts.

### Components at a Glance
- Torus normalization: All coordinates wrap around the edges (`width` and `height`).
- Direction inference: We infer current direction from the last two trail positions accounting for wrap.
- Hamiltonian successor map: For each cell, compute its “next” neighbor in a serpentine cycle.
- Reachable-area estimator: A fast BFS (flood fill) to count empty cells reachable after a candidate move.
- Scoring: `score = area*10 + distance_to_opponent*2 + straight_bonus + cycle_bonus + boost_penalty`. Deaths are very heavily penalized.

### Pseudocode
1. Read state, locate our head, compute `current_dir` and the Hamiltonian successor map.
2. If the next cell on the cycle is free and not opposite to `current_dir`, follow it (no boost by default).
3. Else, evaluate all non-opposite directions:
    - Simulate occupying that next cell (and second step if boosting),
    - If safe, compute reachable area via BFS and distance to opponent head (with torus),
    - Add small straight and cycle continuity bonuses; penalize boost slightly.
4. Choose the move with the highest score. Use BOOST only if its score beats the best non-boost by a clear margin.

### Why This Works
- The cycle provides a low-computation, high-coverage plan when uncontested.
- The heuristic safely deviates when necessary to avoid traps and preserve future mobility.
- Conservative boost policy reduces accidental over-commit into tight spaces.

## SWOT Analysis

### Strengths
- High coverage: Serpentine cycle tends to visit all cells without crossing the trail.
- Safe fallback: Flood-fill ensures we prefer moves that keep future options open.
- Torus-aware: Correct handling of wrap-around for direction, distances, and reachability.
- Modular design: Easy to tune scoring weights and swap components.

### Weaknesses
- Determinism can be exploited: A strong opponent may learn our cycle-following pattern.
- Local heuristic: Flood-fill looks only one (or two with boost) steps ahead before area estimation; it’s not a full game tree search.
- Boost conservatism: May miss rare opportunities where aggressive boosts create winning partitions.

### Opportunities
- Opponent modeling: Predict opponent head trajectory and penalize risky head-on lines more precisely.
- Territory partitioning: Intentionally steer to split the board and secure a larger isolated region.
- Adaptive boost policy: Detect long safe corridors and spend boosts to capitalize.
- Caching: Memoize repeated flood-fill evaluations for identical local configurations.

### Threats
- Adversarial agents designed to break cycles early or force narrow corridors.
- Time constraints: More sophisticated lookahead or RL policies risk timeouts under strict per-move limits.
- Platform constraints: CPU-only libraries and Docker image limits restrict heavier models.

#### `requirements.txt`
**This file lists your agent's Python dependencies.**

*   Don't rename this file either.
*   It comes pre-populated with `Flask` and `requests`.
*   If your agent's logic requires other libraries (like `numpy`, `scipy`, or any other package from PyPI), you **must** add them to this file.
*   When you submit, our build pipeline will run `pip install -r requirements.txt` to install these libraries for your agent.

#### `judge_engine.py`
**A copy of the runner of matches.**

*   The judge engine is the heart of a match in Case Closed. It can be used to simulate a match.
*   The judge engine can be run only when two agents are running on ports `5008` and `5009`.
*   We provide a sample agent that can be used to train your agent and evaluate its performance.

#### `case_closed_game.py`
**A copy of the official game state logic.**

*   Don't rename this file either.
*   This file contains the complete state of the match played, including the `Game`, `GameBoard`, and `Agent` classes.
*   While your agent will receive the game state as a JSON object, you can read this file to understand the exact mechanics of the game: how collisions are detected, how trails work, how boosts function, and what ends a match. This is the "source of truth" for the game rules.
*   Key mechanics:
    - Agents leave permanent trails behind them
    - Hitting any trail (including your own) causes death
    - Head-on collisions: the agent with the longer trail survives
    - Each agent has 3 speed boosts (moves twice instead of once)
    - The board has torus (wraparound) topology
    - Game ends after 500 turns or when one/both agents die

#### `sample_agent.py`
**A simple agent that you can play against.**

*   The sample agent is provided to help you evaluate your own agent's performance. 
*   In conjunction with `judge_engine.py`, you should be able to simulate a match against this agent.

#### `local-tester.py`
**A local tester to verify your agent's API compliance.**

*   This script tests whether your agent correctly implements all required endpoints.
*   Run this to ensure your agent can communicate with the judge engine before submitting.

#### `Dockerfile`
**A copy of the Dockerfile your agent will be containerized with.**

*   This is a copy of a Dockerfile. This same Dockerfile will be used to containerize your agent so we can run it on our evaluation platform.
*   It is **HIGHLY** recommended that you try Dockerizing your agent once you're done. We can't run your agent if it can't be containerized.
*   There are a lot of resources at your disposal to help you with this. We recommend you recruit a teammate that doesn't run Windows for this. 

#### `.dockerignore`
**A .dockerignore file doesn't include its contents into the Docker image**

*   This `.dockerignore` file will be useful for ensuring unwanted files do not get bundled in your Docker image.
*   You have a 5GB image size restriction, so you are given this file to help reduce image size and avoid unnecessary files in the image.

#### `.gitignore`
*   A standard configuration file that tells Git which files and folders (like the `venv` virtual environment directory) to ignore. You shouldn't need to change this.


### Testing your agent:
**Both `agent.py` and `sample_agent.py` come ready to run out of the box!**

*   To test your agent, you will likely need to create a `venv`. Look up how to do this. 
*   Next, you'll need to `pip install` any required libraries. `Flask` is one of these.
*   Finally, in separate terminals, run both `agent.py` and `sample_agent.py`, and only then can you run `judge_engine.py`.
*   You can also run `local-tester.py` to verify your agent's API compliance before testing against another agent.


### Disclaimers:
* There is a 5GB limit on Docker image size, to keep competition fair and timely.
* Due to platform and build-time constraints, participants are limited to **CPU-only PyTorch**; GPU-enabled versions, including CUDA builds, are disallowed. Any other heavy-duty GPU or large ML frameworks (like Tensorflow, JAX) will not be allowed.
* Ensure your agent's `requirements.txt` is complete before pushing changes.
* If you run into any issues, take a look at your own agent first before asking for help.
