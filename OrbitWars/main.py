import math

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_ACTIONS = 8
ORBIT_LIMIT = 50.0


def _get(obs, key, default):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    r = min(1.0, math.log(max(2.0, float(ships))) / math.log(1000.0))
    return 1.0 + 5.0 * r ** 1.5


def _planet_pos_at(planet, eta, omega):
    if not planet["_orb"]:
        return planet["x"], planet["y"]
    theta = planet["_th"] + omega * eta
    return CENTER_X + planet["_r"] * math.cos(theta), CENTER_Y + planet["_r"] * math.sin(theta)


def _hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    sl2 = dx * dx + dy * dy
    if sl2 < 1e-9:
        return False
    t = max(0.0, min(1.0, ((CENTER_X - x1) * dx + (CENTER_Y - y1) * dy) / sl2))
    cx, cy = x1 + t * dx, y1 + t * dy
    return (cx - CENTER_X) ** 2 + (cy - CENTER_Y) ** 2 <= SUN_RADIUS ** 2


def _aim(sx, sy, dst, ships, omega, iters=7):
    speed = _fleet_speed(max(1, ships))
    tx, ty = dst["x"], dst["y"]
    for _ in range(iters):
        d = math.hypot(tx - sx, ty - sy)
        if d < 1e-6:
            return None, None
        eta = d / speed
        tx, ty = _planet_pos_at(dst, eta, omega)
    if _hits_sun(sx, sy, tx, ty):
        return None, None
    return math.atan2(ty - sy, tx - sx), math.hypot(tx - sx, ty - sy) / speed


def _eta_approx(sx, sy, dst, ships, omega):
    _, t = _aim(sx, sy, dst, ships, omega)
    if t is not None:
        return t
    return math.hypot(dst["x"] - sx, dst["y"] - sy) / _fleet_speed(max(1, ships))


def _build_world(obs):
    player = int(_get(obs, "player", 0))
    planets = []
    for p in _get(obs, "planets", []):
        px, py, pr = float(p[2]), float(p[3]), float(p[4])
        dx, dy = px - CENTER_X, py - CENTER_Y
        r_orb = math.hypot(dx, dy)
        orb = r_orb + pr < ORBIT_LIMIT
        planets.append({
            "id": int(p[0]), "owner": int(p[1]),
            "x": px, "y": py, "radius": pr,
            "ships": int(p[5]), "production": int(p[6]),
            "_r": r_orb, "_th": math.atan2(dy, dx) if r_orb > 1e-6 else 0.0, "_orb": orb,
        })
    fleets = []
    for f in _get(obs, "fleets", []):
        fleets.append({
            "id": int(f[0]), "owner": int(f[1]),
            "x": float(f[2]), "y": float(f[3]),
            "angle": float(f[4]), "ships": int(f[6]),
        })
    my_pl = [p for p in planets if p["owner"] == player]
    en_pl = [p for p in planets if p["owner"] not in (-1, player)]
    ne_pl = [p for p in planets if p["owner"] == -1]
    my_fl = [f for f in fleets if f["owner"] == player]
    en_fl = [f for f in fleets if f["owner"] != player]
    return player, planets, my_pl, en_pl, ne_pl, my_fl, en_fl


def _threat(my_planets, enemy_fleets, omega):
    thr = {p["id"]: 0.0 for p in my_planets}
    for f in enemy_fleets:
        spd = _fleet_speed(f["ships"])
        ca, sa = math.cos(f["angle"]), math.sin(f["angle"])
        for p in my_planets:
            eta = math.hypot(f["x"] - p["x"], f["y"] - p["y"]) / spd
            if eta > 80:
                continue
            if p["_orb"]:
                px, py = _planet_pos_at(p, eta, omega)
            else:
                px, py = p["x"], p["y"]
            fx, fy = f["x"], f["y"]
            along = (px - fx) * ca + (py - fy) * sa
            if along <= 0:
                continue
            perp2 = (px - fx - along * ca) ** 2 + (py - fy - along * sa) ** 2
            if perp2 <= (p["radius"] + 1.5) ** 2:
                thr[p["id"]] += f["ships"] / (1.0 + 0.06 * eta)
    return thr


def _avail(planet, thr_map, enemy_planets, step):
    thr = thr_map.get(planet["id"], 0.0)
    is_fl = bool(enemy_planets) and min(
        math.hypot(planet["x"] - e["x"], planet["y"] - e["y"]) for e in enemy_planets
    ) <= 32.0
    if step < 80:
        reserve = max(2, int(0.12 * thr))
    elif step < 200:
        reserve = max(3, planet["production"] + (4 if is_fl else 0) + int(0.20 * thr))
    else:
        reserve = max(4, planet["production"] + 2 + (5 if is_fl else 0) + int(0.25 * thr))
    return max(0, planet["ships"] - reserve)


def _capture_cost(dst, eta, committed):
    garrison = max(0, dst["ships"] - committed.get(dst["id"], 0))
    if dst["owner"] != -1:
        garrison += dst["production"] * max(0, int(eta))
    return max(1, garrison + 2)


def _pair_score(src, dst, remaining, omega):
    t = _eta_approx(src["x"], src["y"], dst, 40, omega)
    prod_gain = dst["production"] * max(0.0, remaining - t)
    if dst["owner"] != -1:
        prod_gain *= 2.2
    raw_cost = max(1.0, float(dst["ships"] + 1))
    return prod_gain / raw_cost - 0.008 * t


def _defend_moves(my_planets, thr_map, avail_map, omega):
    moves = []
    threatened = [
        (thr_map.get(p["id"], 0.0) - p["ships"], p)
        for p in my_planets
        if thr_map.get(p["id"], 0.0) > p["ships"]
    ]
    if not threatened:
        return moves
    threatened.sort(key=lambda x: x[0])
    donors = sorted(my_planets, key=lambda p: -avail_map.get(p["id"], 0))
    for deficit, tgt in threatened:
        need = int(-deficit) + 2
        for donor in donors:
            if need <= 0 or len(moves) >= MAX_ACTIONS:
                break
            a = avail_map.get(donor["id"], 0)
            if a <= 0 or donor["id"] == tgt["id"]:
                continue
            send = min(a, need)
            angle, _ = _aim(donor["x"], donor["y"], tgt, send, omega)
            if angle is None:
                continue
            moves.append([donor["id"], angle, int(send)])
            avail_map[donor["id"]] -= int(send)
            need -= int(send)
    return moves


def _finalize(my_planets, moves):
    final = []
    sent = {}
    caps = {p["id"]: p["ships"] for p in my_planets}
    for pid, angle, ships in moves:
        if len(final) >= MAX_ACTIONS:
            break
        if pid not in caps:
            continue
        room = caps[pid] - sent.get(pid, 0)
        s = int(max(0, min(int(ships), room)))
        if s <= 0:
            continue
        sent[pid] = sent.get(pid, 0) + s
        final.append([int(pid), float(angle), s])
    return final


def agent(obs):
    player, planets, my_pl, en_pl, ne_pl, my_fl, en_fl = _build_world(obs)
    if not my_pl:
        return []

    step = int(_get(obs, "step", 0))
    omega = float(_get(obs, "angular_velocity", 0.0))
    comet_ids = set(_get(obs, "comet_planet_ids", []))
    remaining = max(1, 500 - step)

    thr_map = _threat(my_pl, en_fl, omega)
    avail_map = {p["id"]: _avail(p, thr_map, en_pl, step) for p in my_pl}

    # Endgame: defend and consolidate
    if step >= 460:
        moves = _defend_moves(my_pl, thr_map, avail_map, omega)
        if step < 480 and len(moves) < MAX_ACTIONS:
            best = max(my_pl, key=lambda p: p["ships"] + 8 * p["production"])
            for src in my_pl:
                if src["id"] == best["id"] or len(moves) >= MAX_ACTIONS:
                    continue
                a = avail_map.get(src["id"], 0)
                if a < 5:
                    continue
                angle, _ = _aim(src["x"], src["y"], best, a, omega)
                if angle is not None:
                    moves.append([src["id"], angle, int(a)])
        return _finalize(my_pl, moves)

    # Build all feasible attack pairs (only solo-affordable attacks)
    targets = ne_pl + en_pl
    pairs = []
    for src in my_pl:
        a = avail_map.get(src["id"], 0)
        if a < 2:
            continue
        for dst in targets:
            if dst["id"] in comet_ids and dst["ships"] > 20:
                continue
            angle, t = _aim(src["x"], src["y"], dst, a, omega)
            if angle is None:
                continue
            cost = int(_capture_cost(dst, t, {})) + 1
            if cost > a:
                continue  # cannot solo-capture; skip (no pooling)
            sc = _pair_score(src, dst, remaining, omega)
            pairs.append((sc, src["id"], dst, angle, cost))

    pairs.sort(key=lambda x: -x[0])

    # Greedy bipartite matching: each planet attacks at most one target
    moves = []
    used_src = set()
    used_dst = set()
    committed = {}

    for sc, sid, dst, angle, cost in pairs:
        if len(moves) >= MAX_ACTIONS:
            break
        if sid in used_src or dst["id"] in used_dst:
            continue
        moves.append([sid, angle, cost])
        avail_map[sid] -= cost
        used_src.add(sid)
        used_dst.add(dst["id"])
        committed[dst["id"]] = committed.get(dst["id"], 0) + cost

    # Second pass: unused planets grab any still-affordable target
    if len(moves) < MAX_ACTIONS:
        for src in my_pl:
            if src["id"] in used_src or len(moves) >= MAX_ACTIONS:
                continue
            a = avail_map.get(src["id"], 0)
            if a < 2:
                continue
            best_pair = None
            best_sc = -1e9
            for dst in targets:
                angle, t = _aim(src["x"], src["y"], dst, a, omega)
                if angle is None:
                    continue
                cost = int(_capture_cost(dst, t, committed)) + 1
                if cost > a:
                    continue
                sc = _pair_score(src, dst, remaining, omega)
                if sc > best_sc:
                    best_sc = sc
                    best_pair = (dst, angle, cost)
            if best_pair is None:
                continue
            dst, angle, cost = best_pair
            moves.append([src["id"], angle, cost])
            avail_map[src["id"]] -= cost
            used_src.add(src["id"])
            committed[dst["id"]] = committed.get(dst["id"], 0) + cost
            if len(moves) >= MAX_ACTIONS:
                break

    # Defense last
    if len(moves) < MAX_ACTIONS:
        for m in _defend_moves(my_pl, thr_map, avail_map, omega):
            if len(moves) >= MAX_ACTIONS:
                break
            moves.append(m)

    return _finalize(my_pl, moves)
