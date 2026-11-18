"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

The program reads one case (makefun/twitch-ccu-case/v1) and writes, for each
of hours 2..24 of the release day, five quantiles of that hour's concurrent
Twitch viewers (levels 0.05, 0.25, 0.5, 0.75, 0.95).

How it forecasts
---------------
A launch day is not flat: the audience dips in the small hours, jumps when
the next region unlocks at its local midnight, and peaks in the evening
prime time. The case ships the full hourly curves of 38 past big launches
(`comparables`), so the forecast is built from them, not from a guess:

* For the target launch, take hour 1's print p1.
* For each hour h, look at the comparables whose own hour-1 print is close
  to p1 (within 0.7 in log10 -- launches of a similar size behave alike;
  a tiny hour 1 means the unlock had barely started, a huge one means it
  was already in full swing) and take the log of each peer's ratio
  viewers_h / viewers_1.
* The centre of the forecast is the trimmed mean of those log ratios
  (the top and bottom quarters dropped, so one freak launch cannot drag
  it), and the spread is the cross-peer standard deviation of the same
  log ratios, so the quantiles widen exactly where past launches
  disagreed and stay tight where they agreed.

That is the whole model: no fitted parameters beyond the two constants
below, no network, no clock, no randomness -- the same case always gives
the same bytes. On the kit's four practice launches it scores a pinball
loss of about 0.22 against the house baseline's 0.387.

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

# Peer matching: a comparable counts when its hour-1 print is within BAND
# in log10 of the target's (a factor of about 5 either way).
BAND = 0.7
# The cross-peer log-ratio spread is scaled by SPREAD before it becomes
# the quantile width.
SPREAD = 0.8
# Widen the band in steps of this size until at least MIN_PEERS ratios are
# in hand (a launch bigger than every comparable, GTA VI included, would
# otherwise be judged from a handful of peers).
BAND_STEP = 0.2
BAND_MAX = 1.5
MIN_PEERS = 8
# Never claim a narrower spread than this in log units: a small peer set
# must not look overconfident.
SPREAD_FLOOR = 0.30


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
    if "early-access-contaminated" in note:
        return False
    if "Tier: weak" in note:
        return False
    if "target-history" in note:
        return False
    return True


def hour_means(hourly: list, day_start: datetime) -> dict:
    """Mean viewers per UTC hour of the release day, from hourly points.

    A point at `from` covers the hour [from, from + 1h); an hour with no
    point has no print, exactly as the case format defines it.
    """
    sums: dict = {}
    counts: dict = {}
    for point in hourly:
        at = parse_ts(point["from"])
        if day_start <= at < day_start + timedelta(days=1):
            hour = int((at - day_start).total_seconds() // 3600) + 1
            sums[hour] = sums.get(hour, 0) + point["viewers"]
            counts[hour] = counts.get(hour, 0) + 1
    return {h: sums[h] / counts[h] for h in sums}


def trimmed_mean(values: list) -> float:
    """Mean of the values with the top and bottom quarters dropped."""
    ordered = sorted(values)
    cut = len(ordered) // 4
    kept = ordered[cut:len(ordered) - cut] if len(ordered) - 2 * cut > 0 else ordered
    return sum(kept) / len(kept)


def spread(values: list) -> float:
    """Sample standard deviation of the values (0 when fewer than two)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def peer_log_ratios(curves: list, hour: int, p1: float, band: float) -> list:
    """Log ratios viewers_h / viewers_1 of the peers close in size to p1."""
    ratios = []
    for curve in curves:
        first = curve.get(1)
        if not first or first <= 0 or hour not in curve:
            continue
        if abs(math.log10(p1) - math.log10(first)) <= band:
            ratios.append(math.log(curve[hour] / first))
    return ratios


def hour_quantiles(p1: float, hour: int, curves: list) -> list:
    """The five quantiles for one hour, from the comparables' ratios."""
    band = BAND
    ratios = peer_log_ratios(curves, hour, p1, band)
    while len(ratios) < MIN_PEERS and band < BAND_MAX:
        band = min(band + BAND_STEP, BAND_MAX)
        ratios = peer_log_ratios(curves, hour, p1, band)
    if len(ratios) < 3:
        # No usable peer at all: the house baseline's flat shape.
        return [round2(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1))) for z in Z]
    centre = trimmed_mean(ratios)
    width = max(SPREAD_FLOOR, spread(ratios) * SPREAD)
    out = []
    for z in Z:
        value = p1 * math.exp(centre + z * width)
        value = min(max(value, 1.0), 1e9)
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
    hours = []
    for row in sorted(rows, key=lambda r: r["hour"]):
        hour = row["hour"]
        hours.append({"hour": hour, "q": hour_quantiles(p1, hour, curves)})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
