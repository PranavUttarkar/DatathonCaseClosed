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
    # Modular, smarter agent combining Hamiltonian serpentine cycle following
    # with flood-fill fallback and selective boost usage.

    board = state.get("board")
    my_trail = state.get("agent1_trail", []) if player_number == 1 else state.get("agent2_trail", [])
    opp_trail = state.get("agent2_trail", []) if player_number == 1 else state.get("agent1_trail", [])
    turn_count = state.get("turn_count", 0)

    if not board or not my_trail:
        return jsonify({"move": "RIGHT"}), 200

    H = len(board)
    W = len(board[0]) if H > 0 else 0

    DIRS = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }
    OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    def norm(x, y):
        return (x % W, y % H)

    def cell_empty(x, y, grid):
        nx, ny = norm(x, y)
        try:
            return grid[ny][nx] == 0
        except Exception:
            return False

    def current_direction(trail):
        if len(trail) < 2:
            return "RIGHT"
        (x2, y2) = trail[-1]
        (x1, y1) = trail[-2]
        dx = x2 - x1
        dy = y2 - y1
        if dx > 1:
            dx -= W
        if dx < -1:
            dx += W
        if dy > 1:
            dy -= H
        if dy < -1:
            dy += H
        if (dx, dy) == (1, 0):
            return "RIGHT"
        if (dx, dy) == (-1, 0):
            return "LEFT"
        if (dx, dy) == (0, 1):
            return "DOWN"
        if (dx, dy) == (0, -1):
            return "UP"
        return "RIGHT"

    def build_hamiltonian_successor(w, h):
        succ = {}
        for y in range(h):
            if y % 2 == 0:
                # even row: left to right
                for x in range(w - 1):
                    succ[(x, y)] = (x + 1, y)
                # end of row, go down
                succ[(w - 1, y)] = (w - 1, (y + 1) % h)
            else:
                # odd row: right to left
                for x in range(w - 1, 0, -1):
                    succ[(x, y)] = (x - 1, y)
                # start of row, go down
                succ[(0, y)] = (0, (y + 1) % h)
        return succ

    def reachable_area(start_xy, grid):
        from collections import deque
        sx, sy = norm(start_xy[0], start_xy[1])
        if not cell_empty(sx, sy, grid):
            return 0
        seen = set([(sx, sy)])
        q = deque([(sx, sy)])
        cnt = 0
        while q:
            x, y = q.popleft()
            cnt += 1
            for dx, dy in DIRS.values():
                nx, ny = norm(x + dx, y + dy)
                if (nx, ny) not in seen and cell_empty(nx, ny, grid):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return cnt

    def score_move(dir_name, use_boost_flag):
        dx, dy = DIRS[dir_name]
        hx, hy = my_trail[-1]
        # Step 1
        n1x, n1y = norm(hx + dx, hy + dy)
        if not cell_empty(n1x, n1y, board):
            return -1_000_000
        g = [row[:] for row in board]
        g[n1y][n1x] = 1
        end_pos = (n1x, n1y)
        # Step 2 if boosting
        if use_boost_flag:
            n2x, n2y = norm(n1x + dx, n1y + dy)
            if not cell_empty(n2x, n2y, g):
                return -900_000
            g[n2y][n2x] = 1
            end_pos = (n2x, n2y)

        # Reachable space after move
        area = reachable_area(end_pos, g)

        # Distance from opponent head (torus manhattan) to reduce risky proximity
        opp_head = tuple(opp_trail[-1]) if opp_trail else None
        if opp_head:
            ox, oy = opp_head
            dxwrap = min((end_pos[0] - ox) % W, (ox - end_pos[0]) % W)
            dywrap = min((end_pos[1] - oy) % H, (oy - end_pos[1]) % H)
            dist = dxwrap + dywrap
        else:
            dist = 0

        # Cycle adherence bonus: prefer steps that go toward Hamiltonian successor
        cyc_bonus = 0
        if end_pos in hsucc:
            # If our move puts us on a node whose successor is empty, encourage
            next_on_cycle = hsucc[end_pos]
            if cell_empty(*next_on_cycle, g):
                cyc_bonus = 8

        straight_bonus = 3 if dir_name == cur_dir else 0
        boost_penalty = -2 if use_boost_flag else 0

        return area * 10 + dist * 2 + straight_bonus + cyc_bonus + boost_penalty

    # Build Hamiltonian successor mapping once per request
    hsucc = build_hamiltonian_successor(W, H)
    cur_dir = current_direction(my_trail)

    # 1) Try to follow Hamiltonian cycle if the next cell is free
    head = tuple(my_trail[-1])
    if head in hsucc:
        target = hsucc[head]
        tx, ty = target
        if cell_empty(tx, ty, board):
            # Translate to direction (adjacent by construction)
            dx = (tx - head[0]) % W
            dy = (ty - head[1]) % H
            # Convert wrap to -1/0/1 step
            if dx == W - 1:
                dx = -1
            if dy == H - 1:
                dy = -1
            dir_from_vec = {(0, -1): "UP", (0, 1): "DOWN", (1, 0): "RIGHT", (-1, 0): "LEFT"}
            dname = dir_from_vec.get((dx, dy))
            if dname and dname != OPPOSITE.get(cur_dir, ""):
                # Mostly avoid BOOST when on cycle to maintain coverage rhythm
                return jsonify({"move": dname}), 200

    # 2) Otherwise, evaluate candidate safe directions via scoring
    candidates = [d for d in DIRS.keys() if d != OPPOSITE.get(cur_dir, "")]
    best = ("RIGHT", False, -1_000_000)
    for d in candidates:
        s = score_move(d, use_boost_flag=False)
        if s > best[2]:
            best = (d, False, s)

    # Consider BOOST only if it clearly improves score and is safe
    if boosts_remaining > 0:
        for d in candidates:
            s = score_move(d, use_boost_flag=True)
            # Require a margin to compensate for risk and to avoid skipping cycle cells needlessly
            if s > best[2] + 20:
                best = (d, True, s)

    move = best[0] + (":BOOST" if best[1] else "")
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
