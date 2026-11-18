"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

The program reads one case (makefun/twitch-ccu-case/v1) and writes, for each
of hours 2..24 of the release day, five quantiles of that hour's concurrent
Twitch viewers (levels 0.05, 0.25, 0.5, 0.75, 0.95).

How it forecasts
---------------
A launch day is not flat: the audience dips in the small hours, jumps when
the next region unlocks at its local midnight, and peaks in the evening
prime time. The case ships the full hourly curves of past big launches
(`comparables`), so the forecast is built from them, not from a guess.

The key fact the comparables show is that a launch's shape depends on how
big its first hour already is: the bigger hour 1, the flatter the rest of
the day (across the comparables, log10(hour-1 viewers) and the evening
log-ratio correlate at -0.94). A launch already at 850k in hour 1 barely
grows; one at 1k grows sevenfold. Averaging the peers in a size band
throws that relation away, so this program fits it instead: for each hour,
a kernel-weighted local linear regression of the peers' log-ratios
ln(viewers_h / viewers_1) on log10(viewers_1), with a Gaussian kernel
(bandwidth 0.6 in log10) centred on the target's own hour-1 print. The
fitted line at the target's hour 1 is the centre; the width is the
kernel-weighted residual standard deviation around that line, scaled by
0.8 and floored at 0.25 log units, so the quantiles widen exactly where
past launches of this size disagreed and stay tight where they agreed.
The centre is clamped to the observed range of log-ratios (half a log
unit of slack), so a launch bigger than every comparable -- GTA VI
included -- extrapolates from the biggest peers instead of running away.
The spread factor 0.8 is set by leave-one-out, not by the practice set:
narrowing it keeps improving the four practice launches (0.138 at 0.55)
but steadily worsens leave-one-out (0.457 at 0.55 against 0.437 at 0.8),
so a narrower spread would be overfitting four launches at the cost of
the one that pays.

That is the whole model: no fitted parameters beyond the three constants
below, no network, no clock, no randomness -- the same case always gives
the same bytes. Measured on the kit's four practice launches (truth
reconstructed from SullyGnome's public "Average viewers", hour-1 values
verified to match the case files exactly) it scores a pinball loss of
0.1550, against 0.1843 for the band-average model it replaces and 0.3871
for the house baseline. Leave-one-out over the 26 usable comparables
agrees: 0.4370 against 0.4833 for the band average (big launches,
hour-1 >= 20k: 0.2610 against 0.2638), and on the two largest launches
in the set it wins by a wide margin (ELDEN RING 0.092 vs 0.222,
Cyberpunk 2077 0.114 vs 0.272).

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there);
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, each q five finite numbers above 0 and at most 1e9,
    non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Kernel bandwidth in log10(viewers): a comparable a factor of 10^0.6
# (~4x) away in hour-1 size still carries weight exp(-0.5) ~ 0.6; one a
# factor of 100 away carries almost none.
KERNEL_H = 0.6
# The kernel-weighted residual spread is scaled by SPREAD before it
# becomes the quantile width.
SPREAD = 0.8
# Never claim a narrower spread than this in log units: a small peer set
# must not look overconfident.
SPREAD_FLOOR = 0.25
# How far past the observed range of log-ratios the centre may stray, in
# log units, when the target is bigger (or smaller) than every peer.
CLAMP_SLACK = 0.5

# Markers in a comparable's `unlock` note that disqualify it as a peer.
BAD_TIERS = ("early-access-contaminated", "Tier: weak", "target-history")


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text: str) -> datetime:
    """An ISO timestamp like 2026-09-15T00:00:00Z."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def usable_peer(comp: dict) -> bool:
    """Whether a comparable is a clean past launch to learn from.

    The case's own notes tier each comparable in its `unlock` text. An
    early-access launch's release day is already days into play, a weak
    analogue is not a paid launch at all, and the target-history entry is
    the target game's own pre-release category, not a launch: none of them
    has the shape this forecast is after, and keeping them in costs about
    a sixth of the practice score. A comparable with no tier noted stays.
    """
    note = comp.get("unlock") or ""
    for marker in BAD_TIERS:
        if marker in note:
            return False
    return True


def hour_means(hourly: list, day_start: datetime) -> dict:
    """Mean viewers per UTC hour of the release day, from hourly points.

    A point at `from` covers the hour [from, from + 1h); an hour with no
    point has no print, exactly as the case format defines it.
    """
    by_hour: dict = {}
    for point in hourly:
        at = parse_ts(point["from"])
        if day_start <= at < day_start + timedelta(days=1):
            hour = int((at - day_start).total_seconds() // 3600) + 1
            by_hour.setdefault(hour, []).append(point["viewers"])
    return {h: sum(v) / len(v) for h, v in by_hour.items()}


def _kernel_weights(xs: list, x0: float) -> list:
    """Gaussian kernel weights of the peers' log10 hour-1 around x0."""
    return [math.exp(-0.5 * ((x - x0) / KERNEL_H) ** 2) for x in xs]


def _wols(points: list, weights: list) -> tuple | None:
    """Weighted least squares for y = a + b*x; None when it cannot be fit."""
    n = len(points)
    sw = sum(weights)
    if sw <= 0 or n < 3:
        return None
    mx = sum(w * p[0] for w, p in zip(weights, points)) / sw
    my = sum(w * p[1] for w, p in zip(weights, points)) / sw
    sxx = sum(w * (p[0] - mx) ** 2 for w, p in zip(weights, points))
    if sxx < 1e-12:
        return (my, 0.0)
    b = sum(w * (p[0] - mx) * (p[1] - my) for w, p in zip(weights, points)) / sxx
    return (my - b * mx, b)


def _resid_sd(points: list, a: float, b: float, weights: list) -> float:
    """Kernel-weighted residual SD around y = a + b*x (2 params fitted)."""
    sw = sum(weights)
    if sw <= 0:
        return 0.0
    var = sum(w * (p[1] - (a + b * p[0])) ** 2 for w, p in zip(weights, points)) / sw
    n = len(points)
    if n > 2:
        var *= n / max(1, n - 2)
    return math.sqrt(max(0.0, var))


def _baseline_quantiles(p1: float, hour: int) -> list:
    """The house baseline's flat shape: every hour stays at hour 1."""
    return [round2(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1))) for z in Z]


def hour_quantiles(p1: float, hour: int, curves: list) -> list:
    """The five quantiles for one hour, from a kernel-weighted local fit."""
    x0 = math.log10(p1)
    points = []
    for curve in curves:
        first = curve.get(1)
        if not first or first <= 0 or hour not in curve:
            continue
        points.append((math.log10(first), math.log(curve[hour] / first)))
    if len(points) < 3:
        # No usable peer at all: fall back to the house baseline.
        return _baseline_quantiles(p1, hour)
    weights = _kernel_weights([x for x, _ in points], x0)
    fit = _wols(points, weights)
    if fit is None:
        return _baseline_quantiles(p1, hour)
    a, b = fit
    centre = a + b * x0
    # Clamp to the observed range of log-ratios (with slack): a target
    # beyond every peer must not extrapolate the line into nowhere.
    lo = min(y for _, y in points)
    hi = max(y for _, y in points)
    centre = min(max(centre, lo - CLAMP_SLACK), hi + CLAMP_SLACK)
    width = max(SPREAD_FLOOR, _resid_sd(points, a, b, weights) * SPREAD)
    out = []
    for z in Z:
        value = min(max(p1 * math.exp(centre + z * width), 1.0), 1e9)
        out.append(round2(value))
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is not None:
        p1 = max(1, first["viewers"])
    else:
        # No hour-1 print: fall back to the last history point, else 1.
        history = case.get("history") or []
        p1 = max(1, history[-1]["viewers"]) if history else 1
    curves = []
    for comp in case.get("comparables") or []:
        if not usable_peer(comp):
            continue
        try:
            day_start = parse_ts(comp["day_start"])
            curve = hour_means(comp.get("hourly") or [], day_start)
        except (KeyError, ValueError):
            continue
        if 1 in curve:
            curves.append(curve)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = [{"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], curves)}
             for row in sorted(rows, key=lambda r: r["hour"])]
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
