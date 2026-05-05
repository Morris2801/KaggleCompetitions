import math


CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_ACTIONS_PER_TURN = 8

STATE_DEFEND = "defend"
STATE_EXPAND = "expand"
STATE_CONSOLIDATE = "consolidate"
STATE_RAID = "raid"


def _get(obs, key, default):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(max(2.0, float(ships))) / math.log(1000.0)
    ratio = min(1.0, max(0.0, ratio))
    return 1.0 + 5.0 * (ratio ** 1.5)


def _travel_time(src, dst, ships_guess=40):
    return _distance(src, dst) / max(1.0, _fleet_speed(max(1, ships_guess)))


def _line_hits_sun(src, dst):
    x1, y1 = src["x"], src["y"]
    x2, y2 = dst["x"], dst["y"]
    dx = x2 - x1
    dy = y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-9:
        return False

    t = ((CENTER_X - x1) * dx + (CENTER_Y - y1) * dy) / seg_len2
    t = min(1.0, max(0.0, t))
    px = x1 + t * dx
    py = y1 + t * dy
    return (px - CENTER_X) ** 2 + (py - CENTER_Y) ** 2 <= SUN_RADIUS ** 2


def _likely_hits_planet(fleet, planet):
    dx = math.cos(fleet["angle"])
    dy = math.sin(fleet["angle"])
    vx = planet["x"] - fleet["x"]
    vy = planet["y"] - fleet["y"]
    along = vx * dx + vy * dy
    if along <= 0.0:
        return False, float("inf")

    perp = abs(vx * dy - vy * dx)
    if perp > planet["radius"] + 1.25:
        return False, float("inf")

    eta = along / _fleet_speed(fleet["ships"])
    return True, eta


def _effective_inbound_ships(raw_ships, eta):
    return raw_ships / (1.0 + 0.16 * eta)


def _build_world(obs):
    planets_raw = _get(obs, "planets", [])
    fleets_raw = _get(obs, "fleets", [])
    player = _get(obs, "player", 0)

    planets = []
    for p in planets_raw:
        planets.append(
            {
                "id": int(p[0]),
                "owner": int(p[1]),
                "x": float(p[2]),
                "y": float(p[3]),
                "radius": float(p[4]),
                "ships": int(p[5]),
                "production": int(p[6]),
            }
        )

    fleets = []
    for f in fleets_raw:
        fleets.append(
            {
                "id": int(f[0]),
                "owner": int(f[1]),
                "x": float(f[2]),
                "y": float(f[3]),
                "angle": float(f[4]),
                "from_planet_id": int(f[5]),
                "ships": int(f[6]),
            }
        )

    my_planets = [p for p in planets if p["owner"] == player]
    enemy_planets = [p for p in planets if p["owner"] not in (-1, player)]
    neutral_planets = [p for p in planets if p["owner"] == -1]
    my_fleets = [f for f in fleets if f["owner"] == player]
    enemy_fleets = [f for f in fleets if f["owner"] != player]

    return player, planets, my_planets, enemy_planets, neutral_planets, my_fleets, enemy_fleets


def _estimate_pressure(my_planets, my_fleets, enemy_fleets):
    enemy_pressure = {p["id"]: 0.0 for p in my_planets}
    friendly_inbound = {p["id"]: 0.0 for p in my_planets}

    for fleet in enemy_fleets:
        best_pid = None
        best_eta = float("inf")
        for p in my_planets:
            hits, eta = _likely_hits_planet(fleet, p)
            if hits and eta < best_eta:
                best_eta = eta
                best_pid = p["id"]
        if best_pid is not None:
            enemy_pressure[best_pid] += fleet["ships"] * (1.0 + 4.0 / (best_eta + 4.0))

    for fleet in my_fleets:
        best_pid = None
        best_eta = float("inf")
        for p in my_planets:
            hits, eta = _likely_hits_planet(fleet, p)
            if hits and eta < best_eta:
                best_eta = eta
                best_pid = p["id"]
        if best_pid is not None:
            friendly_inbound[best_pid] += fleet["ships"]

    return enemy_pressure, friendly_inbound


def _estimate_inbound_by_planet(planets, fleets, player):
    my_inbound = {p["id"]: 0.0 for p in planets}
    enemy_inbound = {p["id"]: 0.0 for p in planets}

    for fleet in fleets:
        best_pid = None
        best_eta = float("inf")
        for planet in planets:
            hits, eta = _likely_hits_planet(fleet, planet)
            if hits and eta < best_eta:
                best_eta = eta
                best_pid = planet["id"]

        if best_pid is None:
            continue

        eff = _effective_inbound_ships(fleet["ships"], best_eta)
        if fleet["owner"] == player:
            my_inbound[best_pid] += eff
        else:
            enemy_inbound[best_pid] += eff

    return my_inbound, enemy_inbound


def _classify_frontline(my_planets, enemy_planets):
    frontline = {}
    if not enemy_planets:
        for p in my_planets:
            frontline[p["id"]] = False
        return frontline

    for p in my_planets:
        nearest_enemy = min(_distance(p, e) for e in enemy_planets)
        frontline[p["id"]] = nearest_enemy <= 24.0
    return frontline


def _pick_state(step, my_planets, enemy_planets, neutral_planets, enemy_pressure):
    my_ships = sum(p["ships"] for p in my_planets)
    enemy_ships = max(1, sum(p["ships"] for p in enemy_planets))
    pressure = sum(enemy_pressure.values())

    if pressure > 0.5 * my_ships:
        return STATE_DEFEND
    if neutral_planets and step < 260:
        return STATE_EXPAND
    if my_ships > 1.12 * enemy_ships:
        return STATE_RAID
    if step > 420 and my_ships >= 0.95 * enemy_ships:
        return STATE_DEFEND
    return STATE_CONSOLIDATE


def _available_ships(planet, pressure_here, is_frontline, step):
    reserve = 5 + 2 * planet["production"] + int(0.33 * pressure_here)
    if is_frontline:
        reserve += 8
    if step < 80:
        reserve -= 2
    reserve = max(3, reserve)
    return max(0, planet["ships"] - reserve)


def _angle(src, dst):
    return math.atan2(dst["y"] - src["y"], dst["x"] - src["x"])


def _projected_target_requirement(dst, my_inbound, enemy_inbound, eta_guess, mode):
    net = dst["ships"] + enemy_inbound.get(dst["id"], 0.0) - my_inbound.get(dst["id"], 0.0)

    if dst["owner"] == -1:
        growth = 0.0
        safety = 1.0
    else:
        growth = dst["production"] * eta_guess
        safety = 2.0 if mode == STATE_RAID else 1.2

    return int(max(1.0, math.floor(net + growth + safety)))


def _target_value(dst, mode, is_comet):
    if dst["owner"] == -1:
        val = 12.0 * dst["production"] + 5.0
    else:
        val = 18.0 * dst["production"] + 12.0
        if mode == STATE_RAID:
            val += 8.0

    if is_comet:
        val -= 10.0
    return val


def _defensive_moves(my_planets, enemy_pressure, friendly_inbound, frontline, step):
    moves = []
    threatened = []
    donors = []

    for p in my_planets:
        threat = enemy_pressure.get(p["id"], 0.0)
        support = friendly_inbound.get(p["id"], 0.0)
        net = threat - (p["ships"] + support)
        if net > 0:
            threatened.append((net, p))

        spare = _available_ships(p, threat, frontline.get(p["id"], False), step)
        if spare > 0:
            donors.append((spare, p))

    threatened.sort(key=lambda x: x[0], reverse=True)
    donors.sort(key=lambda x: x[0], reverse=True)

    for deficit, target in threatened:
        need = int(deficit) + 1
        for i, donor_pair in enumerate(donors):
            spare, donor = donor_pair
            if spare <= 0 or donor["id"] == target["id"]:
                continue
            send = min(spare, need)
            if send <= 0:
                continue
            if _line_hits_sun(donor, target):
                continue
            moves.append([donor["id"], _angle(donor, target), int(send)])
            donors[i] = (spare - send, donor)
            need -= send
            if need <= 0 or len(moves) >= MAX_ACTIONS_PER_TURN:
                break
        if len(moves) >= MAX_ACTIONS_PER_TURN:
            break

    return moves


def _coordinated_capture_moves(
    my_planets,
    targets,
    mode,
    enemy_pressure,
    my_inbound,
    enemy_inbound,
    frontline,
    step,
    comet_ids,
):
    moves = []
    donors = []
    for src in my_planets:
        available = _available_ships(src, enemy_pressure.get(src["id"], 0.0), frontline.get(src["id"], False), step)
        if available > 0:
            donors.append({"planet": src, "available": int(available)})

    if not donors:
        return moves

    scored_targets = []
    for dst in targets:
        best_eta = float("inf")
        valid_source_exists = False
        for d in donors:
            src = d["planet"]
            if src["id"] == dst["id"]:
                continue
            if _line_hits_sun(src, dst):
                continue
            valid_source_exists = True
            best_eta = min(best_eta, _travel_time(src, dst, max(1, d["available"] // 2)))

        if not valid_source_exists:
            continue

        req = _projected_target_requirement(dst, my_inbound, enemy_inbound, best_eta, mode)
        val = _target_value(dst, mode, dst["id"] in comet_ids)
        score = val - 0.45 * req - 0.35 * best_eta
        scored_targets.append((score, req, dst))

    scored_targets.sort(key=lambda x: x[0], reverse=True)

    for score, req, dst in scored_targets:
        if len(moves) >= MAX_ACTIONS_PER_TURN:
            break
        if score < -6.0:
            continue

        donors.sort(key=lambda d: (_distance(d["planet"], dst), -d["available"]))
        remaining = req

        for d in donors:
            if len(moves) >= MAX_ACTIONS_PER_TURN:
                break
            if remaining <= 0:
                break

            src = d["planet"]
            if d["available"] <= 0:
                continue
            if src["id"] == dst["id"]:
                continue
            if _line_hits_sun(src, dst):
                continue

            send = min(d["available"], remaining)
            if send <= 0:
                continue

            moves.append([src["id"], _angle(src, dst), int(send)])
            d["available"] -= int(send)
            remaining -= int(send)

            if send >= req * 0.75:
                break

    return moves


def _expansion_or_raid_moves(
    my_planets,
    targets,
    mode,
    enemy_pressure,
    my_inbound,
    enemy_inbound,
    frontline,
    step,
    comet_ids,
):
    return _coordinated_capture_moves(
        my_planets,
        targets,
        mode,
        enemy_pressure,
        my_inbound,
        enemy_inbound,
        frontline,
        step,
        comet_ids,
    )


def _consolidation_moves(
    my_planets,
    neutral_planets,
    enemy_planets,
    enemy_pressure,
    my_inbound,
    enemy_inbound,
    frontline,
    step,
    comet_ids,
):
    neutrals = [p for p in neutral_planets if p["ships"] <= 30]
    neutrals.sort(key=lambda p: (p["ships"] - 3 * p["production"], p["ships"]))

    moves = _expansion_or_raid_moves(
        my_planets,
        neutrals,
        STATE_EXPAND,
        enemy_pressure,
        my_inbound,
        enemy_inbound,
        frontline,
        step,
        comet_ids,
    )
    if len(moves) >= MAX_ACTIONS_PER_TURN:
        return moves

    weak_enemies = [p for p in enemy_planets if p["ships"] <= 60]
    weak_enemies.sort(key=lambda p: (p["ships"] - 2 * p["production"], -p["production"]))

    raid_moves = _expansion_or_raid_moves(
        my_planets,
        weak_enemies,
        STATE_RAID,
        enemy_pressure,
        my_inbound,
        enemy_inbound,
        frontline,
        step,
        comet_ids,
    )

    for m in raid_moves:
        if len(moves) >= MAX_ACTIONS_PER_TURN:
            break
        moves.append(m)
    return moves


def agent(obs):
    player, planets, my_planets, enemy_planets, neutral_planets, my_fleets, enemy_fleets = _build_world(obs)
    if not my_planets:
        return []

    step = int(_get(obs, "step", 0))
    comet_ids = set(_get(obs, "comet_planet_ids", []))

    enemy_pressure, friendly_inbound = _estimate_pressure(my_planets, my_fleets, enemy_fleets)
    my_inbound, enemy_inbound = _estimate_inbound_by_planet(planets, my_fleets + enemy_fleets, player)
    frontline = _classify_frontline(my_planets, enemy_planets)

    state = _pick_state(step, my_planets, enemy_planets, neutral_planets, enemy_pressure)

    if state == STATE_DEFEND:
        moves = _defensive_moves(my_planets, enemy_pressure, friendly_inbound, frontline, step)
    elif state == STATE_EXPAND:
        moves = _expansion_or_raid_moves(
            my_planets,
            neutral_planets,
            STATE_EXPAND,
            enemy_pressure,
            my_inbound,
            enemy_inbound,
            frontline,
            step,
            comet_ids,
        )
    elif state == STATE_RAID:
        moves = _expansion_or_raid_moves(
            my_planets,
            enemy_planets,
            STATE_RAID,
            enemy_pressure,
            my_inbound,
            enemy_inbound,
            frontline,
            step,
            comet_ids,
        )
    else:
        moves = _consolidation_moves(
            my_planets,
            neutral_planets,
            enemy_planets,
            enemy_pressure,
            my_inbound,
            enemy_inbound,
            frontline,
            step,
            comet_ids,
        )

    final_moves = []
    sent_by_source = {}
    ship_lookup = {p["id"]: p["ships"] for p in my_planets}

    for from_id, angle, ships in moves:
        if len(final_moves) >= MAX_ACTIONS_PER_TURN:
            break
        if from_id not in ship_lookup:
            continue
        already = sent_by_source.get(from_id, 0)
        remaining = ship_lookup[from_id] - already
        if remaining <= 0:
            continue

        s = int(max(0, min(int(ships), remaining)))
        if s <= 0:
            continue

        sent_by_source[from_id] = already + s
        final_moves.append([int(from_id), float(angle), s])

    return final_moves
