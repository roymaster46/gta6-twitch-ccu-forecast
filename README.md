# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/roymaster46/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/roymaster46/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast, hour by hour, how many people watch Grand Theft Auto VI live on Twitch over the 24 UTC hours of its release day (2026-11-19), given only the first hour's print — five quantiles per hour (0.05 / 0.25 / 0.5 / 0.75 / 0.95).

## Method

Comparable-launch ratio curves. A launch's audience curve is mostly a shape, not a level: what differs between launches is how many people show up in hour 1, while the way the curve rises and falls through the day is shared. For every past launch in the case's `comparables` we take its release-day hourly curve, divide each hour by that launch's own hour-1 count, and work in log ratios. The forecast for hour h is `p1 * exp(median log ratio + z_tau * sigma_h)`, with sigma from the spread of log ratios across launches, shrunk toward a smooth prior that widens like a random walk. Launches whose hour 1 falls before their real unlock (simultaneous global launches, slow burns) are filtered out by requiring hour 1 to be at least 2% of the launch's own day peak.

Validated by leave-one-out over the kit's 38 comparable launches: mean pinball loss **0.309** against the persistence baseline's **0.360** (~14% better, wins 21/29 launches).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network. Reads every `<id>.in` in the input directory and writes `<id>.out` (`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=26tnw136&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
