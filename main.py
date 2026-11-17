"""GTA VI Twitch CCU forecast: a size-conditioned launch-shape model.

Approach (one specific, data-grounded idea, not a generic median curve).

Every launch day has the same skeleton -- a pre-dawn dip, a US-unlock jump,
a midday sag, a US-evening peak -- but the *scale* of that skeleton depends
strongly on how big hour 1 already was. Across the 37 launch comparables in
the case file, log(peak viewers / hour-1 viewers) correlates -0.94 with
log(hour-1 viewers): a launch whose hour 1 is already huge (Elden Ring,
Cyberpunk 2077) peaks barely above hour 1, while a launch with a small hour 1
(Marvel's Wolverine) grows ~9x by US evening. A single median ratio curve
ignores this and is wrong for both ends; the house persistence baseline
ignores it entirely.

So, per hour h, we fit on the comparable launches:

    log(viewers_h / viewers_1) = a_h + b_h * log(viewers_1)

with the slope split the way the data splits it: ~0 for hours 2-5 (pre-US,
the night dip is size-independent), one shared slope for hours 6-11 (US
unlock and morning), one shared slope for hours 12-24 (US daytime/evening,
where the size effect is strongest, about -0.5). The pool is restricted to
launches that share GTA VI's unlock shape -- midnight-local rolling unlock,
no early access -- because simultaneous-unlock and early-access launches
have a different skeleton (their hour 1 is not a launch hour at all). The
residual spread around the fit (pooled across hours) sets the quantile
bands, scaled by 0.9.

Two guards against the one way this can go wrong:
  - the size anchor is clamped to the pool's observed hour-1 range. The
    log-log trend is estimated on launches up to ~875k; extrapolating it
    linearly to a 1.5M hour 1 would invert the day shape (US-evening peak
    below hour 1), which no observed launch does. Clamping keeps a
    mega-launch anchored to the largest observed launches.
  - if the clean pool is too small (< 5 launches) we fall back to all launch
    comparables; if there is no usable pool at all we fall back to the
    persistence baseline. The output is always a valid 23-row forecast.

Validation (local, on the kit's own practice cases, answers fetched from
SullyGnome, the same public source the case data comes from; the 4th
practice launch, Spider-Man 2, could not be fetched, so n = 3):
  house persistence baseline  mean pinball loss 0.415
  median-ratio model          mean pinball loss 0.302
  this model (main.py)        mean pinball loss 0.145
    per case: Wolverine 0.199, Red Dead Redemption 2 0.110,
    God of War Ragnarok 0.126
Leave-one-out over the 8 similar comparables: baseline 0.418, this model
0.301. In the final case the pool also carries the 4 practice launches
(12 clean launches), which can only help. Deterministic, standard library
only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.15          # fallback persistence spread (log units per sqrt hour)
SPREAD = 0.9          # multiplier on the residual SD for the quantile bands
MIN_CLEAN_POOL = 5    # below this, widen the pool to all launch comparables
MIN_HOURS = 12        # a comparable must have this many of hours 2..24


def round2(x):
    """x rounded to 2 decimals, half up (same as the starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def day_hours(comp):
    """Map a comparable's hourly points to hour -> viewers, hours 1..24."""
    ds = _parse(comp["day_start"])
    pts = {}
    for p in comp.get("hourly", []):
        h = int((_parse(p["from"]) - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and p.get("viewers") is not None:
            pts[h] = p["viewers"]
    return pts


def _unlock(comp):
    return (comp.get("unlock") or "").lower()


def is_rolling_midnight(comp):
    return "midnight local" in _unlock(comp)


def has_early_access(comp):
    u = _unlock(comp)
    return "early access" in u or "early-access" in u


def is_target_prelaunch(comp):
    """The GTA VI category's own pre-release history is not a launch day."""
    name = (comp.get("name") or "").lower()
    return "grand theft auto vi" in name or "before release" in name


def _usable(comp):
    hrs = day_hours(comp)
    if not hrs.get(1) or hrs[1] <= 0:
        return None
    n = sum(1 for h in range(2, 25) if h in hrs and hrs[h] > 0)
    if n < MIN_HOURS:
        return None
    return (comp, hrs)


def clean_pool(case):
    """Rolling-midnight, no-early-access launch comparables; widen if thin."""
    pool = []
    for c in case.get("comparables", []):
        if is_target_prelaunch(c):
            continue
        if not is_rolling_midnight(c) or has_early_access(c):
            continue
        u = _usable(c)
        if u is not None:
            pool.append(u)
    if len(pool) < MIN_CLEAN_POOL:
        pool = []
        for c in case.get("comparables", []):
            if is_target_prelaunch(c):
                continue
            u = _usable(c)
            if u is not None:
                pool.append(u)
    pool.sort(key=lambda ch: ch[0].get("name", ""))
    return pool


def _pooled_slope(pool, hlo, hhi):
    xs, ys = [], []
    for _c, hrs in pool:
        lp = math.log(hrs[1])
        for h in range(hlo, hhi + 1):
            if h in hrs and hrs[h] > 0:
                xs.append(lp)
                ys.append(math.log(hrs[h] / hrs[1]))
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    vx = sum((x - mx) ** 2 for x in xs) / len(xs)
    return cov / vx if vx > 0 else 0.0


def fit(pool):
    """Per-hour intercepts and the two shared slopes; pooled residual SD."""
    be = _pooled_slope(pool, 6, 11)
    bl = _pooled_slope(pool, 12, 24)

    def b_of(h):
        if h < 6:
            return 0.0
        return be if h <= 11 else bl

    a = {}
    for h in range(2, 25):
        vals = []
        for _c, hrs in pool:
            if h in hrs and hrs[h] > 0:
                vals.append(math.log(hrs[h] / hrs[1]) - b_of(h) * math.log(hrs[1]))
        a[h] = sum(vals) / len(vals) if vals else 0.0
    res = []
    for _c, hrs in pool:
        lp = math.log(hrs[1])
        for h in range(2, 25):
            if h in hrs and hrs[h] > 0:
                res.append(math.log(hrs[h] / hrs[1]) - (a[h] + b_of(h) * lp))
    sd = (sum(r * r for r in res) / len(res)) ** 0.5 if res else 0.5
    return a, b_of, sd


def persistence_quantiles(p1, h):
    return [round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    pool = clean_pool(case)
    rows = []
    if not pool:
        for row in case["forecast"]:
            rows.append({"hour": row["hour"], "q": persistence_quantiles(p1, row["hour"])})
        return {"hours": rows}

    a, b_of, sd = fit(pool)
    lo = min(hrs[1] for _c, hrs in pool)
    hi = max(hrs[1] for _c, hrs in pool)
    # Clamp the size anchor to the observed range (see module docstring).
    lp = math.log(min(max(p1, lo), hi))
    for row in case["forecast"]:
        h = row["hour"]
        med = a[h] + b_of(h) * lp
        qs = []
        prev = 0.0
        for z in Z:
            v = p1 * math.exp(med + z * sd * SPREAD)
            v = max(1.0, min(1e9, v))
            v = max(v, prev)
            prev = v
            qs.append(round2(v))
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
