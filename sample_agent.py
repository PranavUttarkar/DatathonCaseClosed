import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentSample"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining

    # -----------------your code here-------------------
    # Territorial control strategy using Voronoi flood-fill scoring.
    # Goal: choose a safe move (optionally with boost) that maximizes
    # (my_reachable_cells - opp_reachable_cells), while avoiding self-traps.

    state_board = state.get("board")
    my_trail = state.get("agent1_trail", []) if player_number == 1 else state.get("agent2_trail", [])
    opp_trail = state.get("agent2_trail", []) if player_number == 1 else state.get("agent1_trail", [])

    # Basic sanity fallbacks
    if not state_board or not my_trail or not opp_trail:
        return jsonify({"move": "RIGHT"}), 200

    # Create a local copy of the board grid (0 empty, 1 occupied)
    board = [row[:] for row in state_board]
    H = len(board)
    W = len(board[0]) if H > 0 else 0

    head_x, head_y = my_trail[-1]
    opp_head_x, opp_head_y = opp_trail[-1]

    # Helper utilities
    DIRS = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    def wrap(pos):
        x, y = pos
        return (x % W, y % H)

    def empty(x, y):
        return board[y % H][x % W] == 0

    def neighbors(x, y):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = (x + dx) % W, (y + dy) % H
            yield nx, ny

    # Deduce my current direction to avoid 180-degree reversals.
    cur_dir_vec = None
    if len(my_trail) >= 2:
        px, py = my_trail[-2]
        vx, vy = head_x - px, head_y - py
        if (vx, vy) != (0, 0):
            cur_dir_vec = (vx, vy)

    # BFS that allows starting from an occupied cell but only expands into empty cells.
    def bfs_distance(start_x, start_y, grid):
        from collections import deque as _dq
        dist = [[-1 for _ in range(W)] for _ in range(H)]
        q = _dq()
        dist[start_y % H][start_x % W] = 0
        q.append((start_x % W, start_y % H))
        while q:
            x, y = q.popleft()
            d = dist[y][x]
            for nx, ny in neighbors(x, y):
                if dist[ny][nx] != -1:
                    continue
                if grid[ny][nx] == 1:  # can only expand into empty cells
                    continue
                dist[ny][nx] = d + 1
                q.append((nx, ny))
        return dist

    def count_component_size(start_x, start_y, grid):
        # Count size of my connected empty region starting from my head.
        from collections import deque as _dq
        if grid[start_y % H][start_x % W] == 1:
            # We allow standing on an occupied cell (our head), but component is the empty neighbors region.
            comp = 0
            seen = set()
            q = _dq()
            for nx, ny in neighbors(start_x, start_y):
                if grid[ny][nx] == 0 and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
            while q:
                x, y = q.popleft()
                comp += 1
                for mx, my in neighbors(x, y):
                    if grid[my][mx] == 0 and (mx, my) not in seen:
                        seen.add((mx, my))
                        q.append((mx, my))
            return comp
        else:
            seen = set([(start_x % W, start_y % H)])
            q = _dq([(start_x % W, start_y % H)])
            comp = 0
            while q:
                x, y = q.popleft()
                comp += 1
                for nx, ny in neighbors(x, y):
                    if grid[ny][nx] == 0 and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            return comp

    def liberties(x, y, grid):
        return sum(1 for nx, ny in neighbors(x, y) if grid[ny][nx] == 0)

    # Generate candidate moves (with/without boost) that are immediately safe.
    candidates = []  # list of tuples: (dir_name, use_boost, path_cells, landing_cell)

    dir_order = ["UP", "RIGHT", "DOWN", "LEFT"]
    # Mild preference to keep going straight if possible
    if cur_dir_vec is not None:
        # bring the straight direction to front
        for name in list(dir_order):
            if DIRS[name] == cur_dir_vec:
                dir_order.remove(name)
                dir_order.insert(0, name)
                break

    for name in dir_order:
        dx, dy = DIRS[name]
        # avoid instant reversal
        if cur_dir_vec is not None and (dx, dy) == (-cur_dir_vec[0], -cur_dir_vec[1]):
            continue
        n1x, n1y = wrap((head_x + dx, head_y + dy))
        if not empty(n1x, n1y):
            continue
        # no-boost candidate
        candidates.append((name, False, [(n1x, n1y)], (n1x, n1y)))
        # boost candidate (same direction twice) if we have boosts and the second step is safe
        n2x, n2y = wrap((n1x + dx, n1y + dy))
        if boosts_remaining > 0 and empty(n2x, n2y):
            candidates.append((name, True, [(n1x, n1y), (n2x, n2y)], (n2x, n2y)))

    # If nothing is safe, pick any direction that doesn't immediately crash (fallback).
    if not candidates:
        for name in dir_order:
            dx, dy = DIRS[name]
            n1x, n1y = wrap((head_x + dx, head_y + dy))
            if empty(n1x, n1y):
                return jsonify({"move": name}), 200
        # truly stuck
        return jsonify({"move": "UP"}), 200

    def score_candidate(path_cells, landing, use_boost):
        # Simulate occupying our path on a copy of the board
        sim = [row[:] for row in board]
        for x, y in path_cells:
            sim[y][x] = 1

        lx, ly = landing
        # Distance maps from new head and from opponent head
        my_dist = bfs_distance(lx, ly, sim)
        opp_dist = bfs_distance(opp_head_x, opp_head_y, sim)

        my_cnt = 0
        opp_cnt = 0
        # Voronoi counting
        for y in range(H):
            row = sim[y]
            for x in range(W):
                if row[x] != 0:
                    continue
                d1 = my_dist[y][x]
                d2 = opp_dist[y][x]
                if d1 == -1 and d2 == -1:
                    continue
                if d2 == -1 and d1 != -1:
                    my_cnt += 1
                elif d1 == -1 and d2 != -1:
                    opp_cnt += 1
                elif d1 < d2:
                    my_cnt += 1
                elif d2 < d1:
                    opp_cnt += 1
                # ties are neutral (ignored)

        # Local safety/shape features
        comp = count_component_size(lx, ly, sim)
        libs = liberties(lx, ly, sim)

        # Heuristic scoring
        score = (my_cnt - opp_cnt)
        # Encourage local freedom
        score += 0.25 * libs
        score += 0.02 * comp

        # Discourage using a boost unless it materially helps
        if use_boost:
            score -= 0.5

        # Strongly penalize moves into dead pockets
        if libs == 0 or comp <= 2:
            score -= 10_000

        # Mild preference to keep going straight
        if cur_dir_vec is not None and DIRS[name] == cur_dir_vec:
            score += 0.1

        return score

    # (no additional helpers needed)

    # Score all candidates
    best = None
    best_score = -1e18
    for name, use_boost, path, landing in candidates:
        sc = score_candidate(path, landing, use_boost)
        if sc > best_score:
            best_score = sc
            best = (name, use_boost)

    if best is None:
        return jsonify({"move": "UP"}), 200

    move_name, use_boost = best
    move = move_name + (":BOOST" if use_boost else "")
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)
