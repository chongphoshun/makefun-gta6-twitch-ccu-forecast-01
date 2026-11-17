# GTA VI Twitch CCU forecast — size-conditioned launch-shape model

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/chongphoshun/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/chongphoshun/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [make.fun bounty gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecast, hour by hour, how many people watch Grand Theft Auto VI live on Twitch over the 24 UTC hours of its release day (2026-11-19), from everything recorded up to the end of hour 1, as five quantiles an hour. Scored by mean pinball loss on the log scale. Prize $250; a reward needs a loss strictly below the best house entry.

## The one idea

Every launch day has the same skeleton — a pre-dawn dip, a US-unlock jump, a midday sag, a US-evening peak — but the *scale* of that skeleton depends strongly on how big hour 1 already was. Across the 37 launch comparables in the case file, `log(peak / hour-1)` correlates **-0.94** with `log(hour-1)`: a launch whose hour 1 is already huge (Elden Ring, Cyberpunk 2077) peaks barely above hour 1, while a launch with a small hour 1 (Marvel's Wolverine) grows ~9x by US evening. A single median ratio curve ignores this and is wrong for both ends; the house persistence baseline ignores it entirely.

`main.py` fits, per hour h, `log(viewers_h / viewers_1) = a_h + b_h * log(viewers_1)` on the comparable launches, with the slope split the way the data splits it: ~0 for hours 2–5 (pre-US, the night dip is size-independent), one shared slope for hours 6–11, one for hours 12–24 (where the size effect is strongest, about −0.5). The pool is restricted to launches that share GTA VI's unlock shape — midnight-local rolling unlock, no early access — because simultaneous-unlock and early-access launches have a different skeleton (their hour 1 is not a launch hour at all). The residual spread around the fit sets the quantile bands (×0.9). Two guards: the size anchor is clamped to the pool's observed hour-1 range (so a 1.5M hour 1 can't invert the day shape), and the program falls back to all comparables, then to persistence, if the clean pool is thin or empty — the output is always a valid 23-row forecast.

## Validation (local, kit practice cases)

Practice answers fetched from SullyGnome, the same public source the case data comes from (Spider-Man 2 could not be fetched, so n = 3):

| model | mean pinball loss (practice) |
|---|---|
| house persistence baseline | 0.415 |
| median-ratio model | 0.302 |
| **this model (`main.py`)** | **0.145** |

Per case: Wolverine 0.199, Red Dead Redemption 2 0.110, God of War Ragnarök 0.126. Leave-one-out over the 8 similar comparables: baseline 0.418, this model 0.301. In the final case the pool also carries the 4 practice launches (12 clean launches, including Red Dead Redemption 2 — Rockstar, console, midnight-local, the closest GTA VI comparable).

## Run it

Python 3.13, standard library only, no network, deterministic (two runs are byte-identical, ~0.1 s):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` — exactly 23 rows (hours 2–24), each `{"hour": h, "q": [q05, q25, q50, q75, q95]}`, quantiles finite, > 0, ≤ 1e9, non-decreasing.
