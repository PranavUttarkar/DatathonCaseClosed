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
OPP_HISTORY = []  # Track opponent head positions for dynamic behavior inference
OPP_BEHAVIOR = {"mode": "unknown"}  # modes: unknown, serpentine, aggressive

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


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
    my_trail_set = set(tuple(p) for p in my_trail)

    # Basic sanity fallbacks
    if not state_board or not my_trail or not opp_trail:
        return jsonify({"move": "RIGHT"}), 200

    # Create a local copy of the board grid (0 empty, 1 occupied)
    board = [row[:] for row in state_board]
    H = len(board)
    W = len(board[0]) if H > 0 else 0

    head_x, head_y = my_trail[-1]
    opp_head_x, opp_head_y = opp_trail[-1]

    # --- Track opponent history for dynamic model ---
    OPP_HISTORY.append((opp_head_x, opp_head_y))
    if len(OPP_HISTORY) > 12:
        OPP_HISTORY.pop(0)

    def infer_opponent_mode(history):
        if len(history) < 5:
            return "unknown"
        # Compute deltas
        deltas = [(history[i+1][0]-history[i][0], history[i+1][1]-history[i][1]) for i in range(len(history)-1)]
        # Normalize wrap effects (approx) - skip large jumps
        cleaned = [d for d in deltas if abs(d[0]) <= 2 and abs(d[1]) <= 2]
        if not cleaned:
            return "unknown"
        horiz = sum(1 for dx, dy in cleaned if dx != 0 and dy == 0)
        vert = sum(1 for dx, dy in cleaned if dy != 0 and dx == 0)
        # Serpentine pattern: horizontal dominance with periodic vertical steps
        if horiz >= 3 and vert >= 1 and vert <= horiz:
            return "serpentine"
        # Aggressive: average distance to us decreasing
        my_head = (head_x, head_y)
        dist_series = []
        for pos in history:
            dx = min(abs(pos[0]-my_head[0]), W-abs(pos[0]-my_head[0]))
            dy = min(abs(pos[1]-my_head[1]), H-abs(pos[1]-my_head[1]))
            dist_series.append(dx+dy)
        if len(dist_series) >= 6:
            first_avg = sum(dist_series[:3])/3
            last_avg = sum(dist_series[-3:])/3
            if last_avg < first_avg - 2:  # approaching
                return "aggressive"
        return "unknown"

    OPP_BEHAVIOR["mode"] = infer_opponent_mode(OPP_HISTORY)

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

    # Phase weighting based on remaining empty cells
    empty_cells = sum(1 for y in range(H) for x in range(W) if board[y][x] == 0)
    total_cells = max(1, H * W)
    empty_ratio = empty_cells / total_cells
    if empty_ratio > 0.6:  # opening
        wTerritory, wLib, wComp, wRisk, boost_pen_base = 1.0, 0.2, 0.01, 2.0, 1.2
    elif empty_ratio > 0.3:  # midgame
        wTerritory, wLib, wComp, wRisk, boost_pen_base = 0.85, 0.3, 0.02, 3.2, 0.8
    else:  # endgame
        wTerritory, wLib, wComp, wRisk, boost_pen_base = 0.45, 0.55, 0.06, 4.8, 0.4

    # Adjust weights based on opponent mode
    mode = OPP_BEHAVIOR["mode"]
    if mode == "aggressive":
        wRisk *= 1.3
        wTerritory *= 0.9
    elif mode == "serpentine":
        wTerritory *= 1.05
        boost_pen_base *= 1.1

    # Precompute current component (for choke detection)
    comp_now = count_component_size(head_x, head_y, board)

    # Opponent immediate reach (one step) and boost reach (two in straight line)
    opp_one_step = set()
    opp_two_step = set()
    # --- Articulation / bridge detection heuristic ---
    def is_articulation(x, y, grid):
        # Rough test: count distinct empty neighbor components if we occupy (x,y)
        if grid[y][x] == 1:
            return False
        temp = [row[:] for row in grid]
        temp[y][x] = 1
        comps = 0
        seen = set()
        for nx, ny in neighbors(x, y):
            if temp[ny][nx] == 0 and (nx, ny) not in seen:
                # BFS this component
                from collections import deque as _dq
                q = _dq([(nx, ny)])
                seen.add((nx, ny))
                while q:
                    cx, cy = q.popleft()
                    for mx, my in neighbors(cx, cy):
                        if temp[my][mx] == 0 and (mx, my) not in seen:
                            seen.add((mx, my))
                            q.append((mx, my))
                comps += 1
                if comps > 1:
                    return True
        return False

    # --- Territory sealing planner ---
    def sealing_bonus(lx, ly, grid):
        # Bonus if landing cell touches >=2 of our own trail (non-opposite) suggesting loop potential.
        adj_my = []
        for nx, ny in neighbors(lx, ly):
            if (nx, ny) in my_trail_set:
                adj_my.append((nx, ny))
        if len(adj_my) >= 2:
            # Estimate enclosed area: count component excluding immediate outside expansion.
            # Simple heuristic: local liberties after occupancy
            temp = [row[:] for row in grid]
            temp[ly][lx] = 1
            libs_local = liberties(lx, ly, temp)
            return 2.0 + 0.3 * libs_local
        return 0.0

    # --- BFS caching structures ---
    bfs_cache_my = {}
    bfs_cache_opp = None  # cache opponent base dist on unchanged board
    bfs_cache_opp = bfs_distance(opp_head_x, opp_head_y, board)
    for name, (odx, ody) in DIRS.items():
        s1x, s1y = wrap((opp_head_x + odx, opp_head_y + ody))
        if empty(s1x, s1y):
            opp_one_step.add((s1x, s1y))
            s2x, s2y = wrap((s1x + odx, s1y + ody))
            if empty(s2x, s2y):
                opp_two_step.add((s2x, s2y))

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
        key_my = (lx, ly, tuple(path_cells))
        if key_my in bfs_cache_my:
            my_dist = bfs_cache_my[key_my]
        else:
            my_dist = bfs_distance(lx, ly, sim)
            bfs_cache_my[key_my] = my_dist
        # Recompute opponent only if candidate path blocked an opp reachable cell earlier
        opp_blocked = any(sim[y][x] == 1 and bfs_cache_opp[y][x] != -1 for x, y in path_cells)
        if opp_blocked:
            opp_dist = bfs_distance(opp_head_x, opp_head_y, sim)
        else:
            opp_dist = bfs_cache_opp

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
        score = wTerritory * (my_cnt - opp_cnt)
        # Encourage local freedom
        score += wLib * libs
        score += wComp * comp

        # Use component delta to discourage big choke-ins
        if comp_now > 0:
            ratio = comp / comp_now
            if ratio < 0.25:
                score -= 15.0
            elif ratio < 0.5:
                score -= 5.0

        # Risk: avoid landing where opponent can be next tick or by boost
        if (lx, ly) in opp_one_step:
            score -= wRisk * 1500.0  # near certain duel risk
        if (lx, ly) in opp_two_step:
            score -= wRisk * 400.0

        # Also avoid moving through a square opp can step into immediately
        if any(cell in opp_one_step for cell in path_cells):
            score -= wRisk * 800.0

        # Prefer to keep distance if opponent is very close
        d_here = opp_dist[ly][lx]
        if d_here != -1 and d_here <= 2:
            score -= wRisk * (3 - d_here) * 50.0

        # Discourage using a boost unless it materially helps; later phases are more lenient
        if use_boost:
            score -= boost_pen_base

        # Strongly penalize moves into dead pockets
        if libs == 0 or comp <= 2:
            score -= 10_000

        # Mild preference to keep going straight
        if cur_dir_vec is not None and DIRS[name] == cur_dir_vec:
            score += 0.1

        # Articulation penalty/bonus
        if is_articulation(lx, ly, sim):
            # Occupying articulation may be good if we split opponent area significantly; approximate by opp_cnt decrease potential
            score += 0.5  # slight positive (can refine later)
        else:
            score += 0.0

        # Sealing bonus
        score += sealing_bonus(lx, ly, sim)

        return score

        return score

    # (no additional helpers needed)

    # First pass scoring
    scored = []
    for name, use_boost, path, landing in candidates:
        sc = score_candidate(path, landing, use_boost)
        scored.append((sc, name, use_boost, path, landing))
    scored.sort(reverse=True)

    # Beam search depth-2 (simulate our second move without opponent interference, optimistic)
    beam_width = min(3, len(scored))
    second_layer_scores = {}
    for i in range(beam_width):
        sc1, name1, use_boost1, path1, land1 = scored[i]
        lx, ly = land1
        # Generate second-step candidates from landing position (no boost expansion for simplicity)
        for name2, (dx2, dy2) in DIRS.items():
            prev_vec = DIRS[name1]
            if DIRS[name2] == (-prev_vec[0], -prev_vec[1]):
                continue
            sx, sy = wrap((lx + dx2, ly + dy2))
            # Build a mini sim board
            sim2 = [row[:] for row in board]
            for x, y in path1:
                sim2[y][x] = 1
            if sim2[sy][sx] == 1:
                continue
            sim2[sy][sx] = 1
            # Count liberties and component size for second landing
            comp2 = count_component_size(sx, sy, sim2)
            libs2 = liberties(sx, sy, sim2)
            bonus = 0.1 * libs2 + 0.01 * comp2
            key = (name1, use_boost1)
            second_layer_scores[key] = max(second_layer_scores.get(key, -1e18), sc1 + bonus)

    # Monte Carlo rollouts for top K candidates
    import random as _r
    rollout_count = 6 if empty_ratio > 0.3 else 10  # more in later tighter phases
    top_for_rollout = scored[:beam_width]
    for sc1, name1, use_boost1, path1, land1 in top_for_rollout:
        roll_score_accum = 0.0
        for r in range(rollout_count):
            # Simulate a few random steps (3) with random safe moves to estimate stability
            simr = [row[:] for row in board]
            for x, y in path1:
                simr[y][x] = 1
            curx, cury = land1
            for step in range(3):
                # pick random safe move
                moves = []
                for nm, (dxr, dyr) in DIRS.items():
                    nxr, nyr = wrap((curx + dxr, cury + dyr))
                    if simr[nyr][nxr] == 0:
                        moves.append((nxr, nyr))
                if not moves:
                    break
                nxr, nyr = _r.choice(moves)
                simr[nyr][nxr] = 1
                curx, cury = nxr, nyr
            # After rollout steps, quick liberty measure
            roll_libs = liberties(curx, cury, simr)
            roll_score_accum += roll_libs
        avg_roll = roll_score_accum / max(1, rollout_count)
        key = (name1, use_boost1)
        second_layer_scores[key] = second_layer_scores.get(key, sc1) + 0.05 * avg_roll

    # Combine scores selecting best final move
    best = None
    best_score = -1e18
    for sc1, name1, use_boost1, path1, land1 in scored:
        final_sc = second_layer_scores.get((name1, use_boost1), sc1)
        if final_sc > best_score:
            best_score = final_sc
            best = (name1, use_boost1)

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
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
